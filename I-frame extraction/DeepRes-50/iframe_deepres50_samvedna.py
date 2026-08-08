import os
import random
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import cv2
import dlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.layers import (
    Activation, Add, BatchNormalization, Conv2D, Dense,
    GlobalAveragePooling2D, Input, MaxPooling2D, ZeroPadding2D,
)
from tensorflow.keras.optimizers import Adam


# ---- paths, change these -----------------------------------------------
SAMVEDNA_VIDEO_DIR = Path(r"C:\PATH\TO\SAMVEDNA\VIDEOS")
EXTRACTED_IFRAME_DIR = Path(r"C:\PATH\TO\SAMVEDNA_IFRAME\extracted_frames")
ALIGNED_FRAME_DIR = Path(r"C:\PATH\TO\SAMVEDNA_IFRAME\aligned_frames")
DLIB_LANDMARK_MODEL = Path(r"C:\PATH\TO\shape_predictor_68_face_landmarks.dat")

# leave both False once extraction/preprocessing has been run once
RUN_IFRAME_EXTRACTION = False
RUN_FACE_PREPROCESSING = False

OVERWRITE_EXISTING_IFRAMES = False
OVERWRITE_EXISTING_ALIGNED = False

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg", ".m4v"}

CLASS_NAMES = ["Anger", "Disgust", "Fear", "Happy", "Sad"]
NUM_CLASSES = len(CLASS_NAMES)
CLASS_TO_INDEX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
CLASS_LOOKUP = {name.lower(): name for name in CLASS_NAMES}

IMG_SIZE = (48, 48)
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
VAL_SPLIT = 0.20
EARLY_STOPPING_PATIENCE = 5
SEED = 123

FACE_CHIP_PADDING = 0.25  # this is dlib's own default for get_face_chip


# reproducibility - not perfect on GPU but good enough
os.environ.setdefault("PYTHONHASHSEED", str(SEED))
random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except (AttributeError, RuntimeError):
    pass

print("TensorFlow version:", tf.__version__)
print("GPU(s) detected:", tf.config.list_physical_devices("GPU"))


def infer_class_from_relative_path(relative_path: Path):
    """Look for one of the five class names somewhere in the video's path.
    Works whether videos sit directly under a class folder or under
    something like Subject01/Anger/video.mp4. Anything outside the 5
    retained classes (Neutral, etc.) just gets skipped."""
    matches = []
    for part in relative_path.parts[:-1]:
        canonical = CLASS_LOOKUP.get(part.lower())
        if canonical is not None:
            matches.append((part, canonical))

    if not matches:
        return None, None

    canonical_classes = {c for _, c in matches}
    if len(canonical_classes) > 1:
        raise ValueError(f"Ambiguous emotion folders in path '{relative_path}': {sorted(canonical_classes)}")

    matched_part, canonical_class = matches[0]
    return matched_part, canonical_class


def make_video_id(relative_path: Path, matched_class_part: str):
    """Builds a video id with the class folder stripped out, e.g.
    Subject01/Anger/video01.mp4 -> Subject01__video01"""
    parts_without_suffix = list(relative_path.with_suffix("").parts)
    removed = False
    kept_parts = []
    for part in parts_without_suffix:
        if not removed and part == matched_class_part:
            removed = True
            continue
        kept_parts.append(part)
    return "__".join(kept_parts) or relative_path.stem


# ---------------------------------------------------------------------
# I-frame extraction
# ---------------------------------------------------------------------

def check_ffmpeg_available():
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg not found on PATH — install FFmpeg 5.1 (what's used in the paper) "
            "or add it to PATH."
        )


def extract_iframes_from_video(video_path: Path, output_dir: Path, overwrite: bool = False):
    """Pulls the codec-level I-frames out of one video with ffmpeg.
    select=eq(pict_type,I) grabs only intra-coded frames, -vsync vfr
    stops ffmpeg from padding out to a constant frame rate. Frames come
    out as lossless PNGs."""
    check_ffmpeg_available()

    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    existing_frames = sorted(output_dir.glob("iframe_*.png"))
    if existing_frames and not overwrite:
        print(f"skipping {video_path.name}, already has {len(existing_frames)} frames")
        return existing_frames

    if overwrite:
        for frame_path in existing_frames:
            frame_path.unlink()

    output_pattern = str(output_dir / "iframe_%05d.png")
    select_filter = r"select=eq(pict_type\,I)"

    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(video_path),
        "-map", "0:v:0",
        "-vf", select_filter,
        "-vsync", "vfr",
        "-start_number", "1",
        output_pattern,
    ]

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffmpeg failed on {video_path}") from exc

    extracted_frames = sorted(output_dir.glob("iframe_*.png"))
    if not extracted_frames:
        print(f"warning: no i-frames written for {video_path}")
    else:
        print(f"{video_path.name}: got {len(extracted_frames)} i-frame(s)")

    return extracted_frames


def extract_iframes_from_dataset(video_root: Path, output_root: Path, overwrite: bool = False):
    """Walks the dataset root, pulls I-frames for every video that falls
    into one of the 5 retained classes. Output ends up as
    output_root/<class>/<video_id>/iframe_00001.png etc."""
    video_root = Path(video_root)
    output_root = Path(output_root)

    if not video_root.exists():
        raise FileNotFoundError(f"video folder not found: {video_root} — check SAMVEDNA_VIDEO_DIR")

    all_video_files = sorted(
        p for p in video_root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not all_video_files:
        raise FileNotFoundError(f"no video files under: {video_root}")

    selected_videos, ignored_videos = [], []
    for video_path in all_video_files:
        relative_path = video_path.relative_to(video_root)
        matched_part, class_name = infer_class_from_relative_path(relative_path)
        if class_name is None:
            ignored_videos.append(relative_path)
            continue
        video_id = make_video_id(relative_path, matched_part)
        selected_videos.append((video_path, relative_path, class_name, video_id))

    if not selected_videos:
        raise RuntimeError(f"none of the found videos matched classes: {CLASS_NAMES}")

    print(f"total videos found: {len(all_video_files)}")
    print(f"in the 5 retained classes: {len(selected_videos)}")
    print(f"skipped (other classes): {len(ignored_videos)}")

    total_iframes = 0
    for index, (video_path, relative_path, class_name, video_id) in enumerate(selected_videos, start=1):
        destination = output_root / class_name / video_id
        print(f"[{index}/{len(selected_videos)}] {relative_path} -> {class_name}/{video_id}")
        frames = extract_iframes_from_video(video_path, destination, overwrite=overwrite)
        total_iframes += len(frames)

    print("\ndone extracting i-frames")
    print(f"videos processed: {len(selected_videos)}, total frames: {total_iframes}")
    print(f"output dir: {output_root}")


# ---------------------------------------------------------------------
# face detection / landmarks / alignment
# ---------------------------------------------------------------------

_face_detector = dlib.get_frontal_face_detector()
_landmark_predictor = None


def get_landmark_predictor():
    global _landmark_predictor
    if _landmark_predictor is None:
        if not DLIB_LANDMARK_MODEL.exists():
            raise FileNotFoundError(
                f"can't find the landmark model at {DLIB_LANDMARK_MODEL} — "
                "download shape_predictor_68_face_landmarks.dat and point DLIB_LANDMARK_MODEL at it"
            )
        _landmark_predictor = dlib.shape_predictor(str(DLIB_LANDMARK_MODEL))
    return _landmark_predictor


def detect_face_and_shape(image_bgr):
    """Finds the biggest face in the frame and its 68 landmarks.
    Returns (None, None, None) if nothing was detected."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = _face_detector(gray, 1)
    if len(faces) == 0:
        return None, None, None

    # in case of multiple faces just take the biggest one
    face_rect = max(faces, key=lambda r: max(0, r.width()) * max(0, r.height()))
    predictor = get_landmark_predictor()
    shape = predictor(gray, face_rect)
    landmarks = np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)], dtype=np.float32)
    return face_rect, shape, landmarks


def align_crop_resize_face(image_bgr, shape, output_size=IMG_SIZE, padding=FACE_CHIP_PADDING):
    """dlib.get_face_chip does the actual alignment/crop for us — just
    have to feed it RGB and convert back after."""
    if output_size[0] != output_size[1]:
        raise ValueError("output_size needs to be square for get_face_chip")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    aligned_rgb = dlib.get_face_chip(image_rgb, shape, size=int(output_size[0]), padding=float(padding))
    aligned_bgr = cv2.cvtColor(aligned_rgb, cv2.COLOR_RGB2BGR)

    if aligned_bgr.shape[:2] != output_size:
        aligned_bgr = cv2.resize(aligned_bgr, output_size, interpolation=cv2.INTER_LINEAR)

    return aligned_bgr


def preprocess_face(image_path: Path, output_size=IMG_SIZE):
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        return None
    _, shape, _ = detect_face_and_shape(image_bgr)
    if shape is None:
        return None
    return align_crop_resize_face(image_bgr, shape, output_size=output_size)


def preprocess_dataset(iframe_root: Path, aligned_root: Path, output_size=IMG_SIZE, overwrite=False):
    """Runs face detect+align+crop over every extracted I-frame, keeping
    the same class/video folder layout so we can still group by video
    later for majority voting. Frames with no detectable face get
    skipped."""
    iframe_root = Path(iframe_root)
    aligned_root = Path(aligned_root)

    if not iframe_root.exists():
        raise FileNotFoundError(f"i-frame root doesn't exist: {iframe_root}")

    frame_paths = sorted(iframe_root.rglob("iframe_*.png"))
    if not frame_paths:
        raise FileNotFoundError(f"no i-frames found under: {iframe_root}")

    kept = skipped_no_face = failed_write = 0

    for index, frame_path in enumerate(frame_paths, start=1):
        relative_path = frame_path.relative_to(iframe_root)
        dest_path = aligned_root / relative_path
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        if dest_path.exists() and not overwrite:
            kept += 1
            continue

        aligned = preprocess_face(frame_path, output_size=output_size)
        if aligned is None:
            skipped_no_face += 1
            continue

        if not cv2.imwrite(str(dest_path), aligned):
            failed_write += 1
            continue

        kept += 1
        if index % 200 == 0:
            print(f"{index}/{len(frame_paths)} done (kept={kept}, no_face={skipped_no_face}, write_fail={failed_write})")

    print("\nface preprocessing done")
    print(f"input frames: {len(frame_paths)}, kept: {kept}, no face: {skipped_no_face}, write failures: {failed_write}")
    print(f"output dir: {aligned_root}")


if RUN_IFRAME_EXTRACTION:
    extract_iframes_from_dataset(SAMVEDNA_VIDEO_DIR, EXTRACTED_IFRAME_DIR, overwrite=OVERWRITE_EXISTING_IFRAMES)

if RUN_FACE_PREPROCESSING:
    preprocess_dataset(EXTRACTED_IFRAME_DIR, ALIGNED_FRAME_DIR, output_size=IMG_SIZE, overwrite=OVERWRITE_EXISTING_ALIGNED)


# ---------------------------------------------------------------------
# build frame index + video-level split
# ---------------------------------------------------------------------

def build_frame_index(aligned_root: Path):
    """expects aligned_root/class_name/video_id/iframe_*.png, returns a
    list of (frame_path, class_idx, video_id) tuples"""
    aligned_root = Path(aligned_root)
    if not aligned_root.exists():
        raise FileNotFoundError(f"aligned frames dir missing: {aligned_root} — run extraction/preprocessing first")

    records = []
    videos_without_frames = []

    for class_name in CLASS_NAMES:
        class_dir = aligned_root / class_name
        if not class_dir.exists():
            print(f"warning: no folder for class {class_name}")
            continue

        class_idx = CLASS_TO_INDEX[class_name]
        for video_dir in sorted(p for p in class_dir.iterdir() if p.is_dir()):
            frame_files = sorted(video_dir.glob("iframe_*.png"))
            if not frame_files:
                videos_without_frames.append(f"{class_name}/{video_dir.name}")
                continue
            video_id = f"{class_name}/{video_dir.name}"
            for frame_path in frame_files:
                records.append((frame_path, class_idx, video_id))

    if not records:
        raise FileNotFoundError(f"no preprocessed frames under: {aligned_root}")
    if videos_without_frames:
        print(f"warning: {len(videos_without_frames)} videos ended up with zero usable frames")

    return records


records = build_frame_index(ALIGNED_FRAME_DIR)
frame_paths = np.array([str(r[0]) for r in records])
frame_labels = np.array([r[1] for r in records], dtype=np.int32)
frame_video_ids = np.array([r[2] for r in records])


def split_videos_stratified(video_ids, labels_per_frame, validation_fraction=VAL_SPLIT, seed=SEED):
    """splits by video, not by frame — otherwise you'd get frames from
    the same video leaking into both train and val"""
    video_to_label = {}
    for vid, label in zip(video_ids, labels_per_frame):
        label = int(label)
        if vid in video_to_label and video_to_label[vid] != label:
            raise ValueError(f"video '{vid}' has conflicting labels")
        video_to_label[vid] = label

    unique_video_ids = np.array(list(video_to_label.keys()))
    unique_video_labels = np.array([video_to_label[v] for v in unique_video_ids], dtype=np.int32)

    class_counts = Counter(unique_video_labels.tolist())
    missing = [CLASS_NAMES[i] for i in range(NUM_CLASSES) if class_counts.get(i, 0) == 0]
    if missing:
        raise ValueError(f"no videos at all for class(es): {', '.join(missing)}")

    too_small = {CLASS_NAMES[i]: c for i, c in class_counts.items() if c < 2}
    if too_small:
        raise ValueError(f"need at least 2 videos per class for a stratified split, too small: {too_small}")

    train_ids, val_ids = train_test_split(
        unique_video_ids, test_size=validation_fraction, random_state=seed,
        shuffle=True, stratify=unique_video_labels,
    )
    train_ids, val_ids = np.array(train_ids), np.array(val_ids)

    if len(np.intersect1d(train_ids, val_ids)) != 0:
        raise RuntimeError("train/val video overlap — this shouldn't happen")

    return train_ids, val_ids, video_to_label


train_video_ids, val_video_ids_unique, video_to_label = split_videos_stratified(frame_video_ids, frame_labels)

train_mask = np.isin(frame_video_ids, train_video_ids)
val_mask = np.isin(frame_video_ids, val_video_ids_unique)

train_paths, train_labels, train_frame_video_ids = frame_paths[train_mask], frame_labels[train_mask], frame_video_ids[train_mask]
val_paths, val_labels, val_video_ids = frame_paths[val_mask], frame_labels[val_mask], frame_video_ids[val_mask]

if len(train_paths) == 0 or len(val_paths) == 0:
    raise RuntimeError("train or val ended up empty after the split")

print("\nsplit summary")
print(f"total frames: {len(frame_paths)}, total videos: {len(video_to_label)}")
print(f"train videos: {len(train_video_ids)}, val videos: {len(val_video_ids_unique)}")
print(f"train frames: {len(train_paths)}, val frames: {len(val_paths)}")


def print_video_class_distribution(split_name, video_ids_for_split):
    counts = Counter(video_to_label[v] for v in video_ids_for_split)
    text = ", ".join(f"{CLASS_NAMES[i]}={counts.get(i, 0)}" for i in range(NUM_CLASSES))
    print(f"{split_name}: {text}")


print_video_class_distribution("train", train_video_ids)
print_video_class_distribution("val", val_video_ids_unique)


# ---------------------------------------------------------------------
# tf.data pipeline
# ---------------------------------------------------------------------

def load_and_preprocess_image(path, label):
    image_bytes = tf.io.read_file(path)
    image = tf.image.decode_png(image_bytes, channels=3)
    image = tf.image.resize(image, IMG_SIZE, method="bilinear")  # already 48x48 but just in case
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


def make_dataset(paths, labels, shuffle, batch_size=BATCH_SIZE, seed=SEED):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


train_ds = make_dataset(train_paths, train_labels, shuffle=True)
# val stays unshuffled so predictions line up with val_labels / val_video_ids later
val_ds = make_dataset(val_paths, val_labels, shuffle=False)


# ---------------------------------------------------------------------
# DeepRes-50
# ---------------------------------------------------------------------

def identity_block(X, f, filters, stage, block):
    conv_base_name, bn_base_name = f"res{stage}{block}_", f"bn{stage}{block}_"
    f1, f2, f3 = filters
    bn_axis = 3
    X_shortcut = X

    X = Conv2D(f1, (1, 1), strides=(1, 1), padding="valid", name=conv_base_name + "first_component",
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=bn_axis, name=bn_base_name + "first_component")(X)
    X = Activation("relu")(X)

    X = Conv2D(f2, (f, f), strides=(1, 1), padding="same", name=conv_base_name + "second_component",
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=bn_axis, name=bn_base_name + "second_component")(X)
    X = Activation("relu")(X)

    X = Conv2D(f3, (1, 1), strides=(1, 1), padding="valid", name=conv_base_name + "third_component",
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=bn_axis, name=bn_base_name + "third_component")(X)

    X = Add()([X, X_shortcut])
    X = Activation("relu")(X)
    return X


def convolutional_block(X, f, filters, stage, block, s=2):
    conv_base_name, bn_base_name = f"res{stage}{block}_", f"bn{stage}{block}_"
    f1, f2, f3 = filters
    bn_axis = 3
    X_shortcut = X

    X = Conv2D(f1, (1, 1), strides=(s, s), padding="valid", name=conv_base_name + "first_component",
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=bn_axis, name=bn_base_name + "first_component")(X)
    X = Activation("relu")(X)

    X = Conv2D(f2, (f, f), strides=(1, 1), padding="same", name=conv_base_name + "second_component",
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=bn_axis, name=bn_base_name + "second_component")(X)
    X = Activation("relu")(X)

    X = Conv2D(f3, (1, 1), strides=(1, 1), padding="valid", name=conv_base_name + "third_component",
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=bn_axis, name=bn_base_name + "third_component")(X)

    X_shortcut = Conv2D(f3, (1, 1), strides=(s, s), padding="valid", name=conv_base_name + "merge",
                         kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=bn_axis, name=bn_base_name + "merge")(X_shortcut)

    X = Add()([X, X_shortcut])
    X = Activation("relu")(X)
    return X


def DeepRes50(input_shape=(48, 48, 3), classes=NUM_CLASSES):
    """the 50-layer residual net used for frame-level classification —
    standard bottleneck resnet50, just with a 5-way softmax head at the
    end and sized for 48x48 input instead of the usual 224x224"""
    X_input = Input(shape=input_shape, name="input_image")

    X = ZeroPadding2D((3, 3), name="zero_padding")(X_input)
    X = Conv2D(64, (7, 7), strides=(2, 2), name="conv_1", kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name="bn_1")(X)
    X = Activation("relu")(X)
    X = MaxPooling2D((3, 3), strides=(2, 2), name="max_pool")(X)

    X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block="a", s=1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block="b")
    X = identity_block(X, 3, [64, 64, 256], stage=2, block="c")

    X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block="a", s=2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block="b")
    X = identity_block(X, 3, [128, 128, 512], stage=3, block="c")
    X = identity_block(X, 3, [128, 128, 512], stage=3, block="d")

    X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block="a", s=2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block="b")
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block="c")
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block="d")
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block="e")
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block="f")

    X = convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block="a", s=2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block="b")
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block="c")

    X = GlobalAveragePooling2D(name="global_average_pool")(X)
    X = Dense(classes, activation="softmax", name=f"fc{classes}", kernel_initializer=glorot_uniform(seed=0))(X)

    return Model(inputs=X_input, outputs=X, name="DeepRes-50")


model = DeepRes50(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), classes=NUM_CLASSES)
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

early_stopping = EarlyStopping(monitor="val_loss", patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True, verbose=1)

history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=[early_stopping])


# ---------------------------------------------------------------------
# frame-level eval
# ---------------------------------------------------------------------

LABEL_IDS = np.arange(NUM_CLASSES)

y_pred_probs = model.predict(val_ds, verbose=1)
y_pred_frame = np.argmax(y_pred_probs, axis=1).astype(np.int32)
y_true_frame = val_labels.astype(np.int32)

if not (len(y_pred_frame) == len(y_true_frame) == len(val_video_ids)):
    raise RuntimeError("predictions/labels/video ids are out of sync somehow")

frame_accuracy = accuracy_score(y_true_frame, y_pred_frame)
print(f"\nframe-level accuracy: {frame_accuracy * 100:.4f}%")
print(classification_report(y_true_frame, y_pred_frame, labels=LABEL_IDS, target_names=CLASS_NAMES, digits=4, zero_division=0))

cm_frame = confusion_matrix(y_true_frame, y_pred_frame, labels=LABEL_IDS)
disp_frame = ConfusionMatrixDisplay(confusion_matrix=cm_frame, display_labels=CLASS_NAMES)
fig, ax = plt.subplots(figsize=(8, 6))
disp_frame.plot(ax=ax, xticks_rotation=45, values_format="d")
plt.title("Frame-Level Confusion Matrix\nI-Frame Extraction + DeepRes-50")
plt.tight_layout()
plt.show()


# ---------------------------------------------------------------------
# majority voting -> video-level label
# ---------------------------------------------------------------------

def majority_vote_per_video(video_ids, frame_probs, frame_true_labels):
    """most-voted class wins; ties get broken by whichever tied class has
    the higher average softmax score across that video's frames"""
    video_ids = np.asarray(video_ids)
    frame_probs = np.asarray(frame_probs)
    frame_true_labels = np.asarray(frame_true_labels, dtype=np.int32)
    frame_preds = np.argmax(frame_probs, axis=1).astype(np.int32)

    unique_videos = np.unique(video_ids)
    video_true, video_pred = [], []

    for video_id in unique_videos:
        mask = video_ids == video_id
        probs_for_video = frame_probs[mask]
        preds_for_video = frame_preds[mask]
        true_for_video = frame_true_labels[mask]

        true_classes = np.unique(true_for_video)
        if len(true_classes) != 1:
            raise ValueError(f"video '{video_id}' has inconsistent true labels: {true_classes.tolist()}")

        votes = np.bincount(preds_for_video, minlength=NUM_CLASSES)
        max_votes = votes.max()
        tied_classes = np.flatnonzero(votes == max_votes)

        if len(tied_classes) == 1:
            final_label = int(tied_classes[0])
        else:
            mean_probabilities = probs_for_video.mean(axis=0)
            tied_scores = mean_probabilities[tied_classes]
            final_label = int(tied_classes[np.argmax(tied_scores)])

        video_true.append(int(true_classes[0]))
        video_pred.append(final_label)

    return np.array(video_true, dtype=np.int32), np.array(video_pred, dtype=np.int32), unique_videos


video_true, video_pred, video_ids_ordered = majority_vote_per_video(val_video_ids, y_pred_probs, y_true_frame)

video_accuracy = accuracy_score(video_true, video_pred)
print(f"\nvideo-level accuracy after majority voting: {video_accuracy * 100:.4f}%  ({len(video_ids_ordered)} videos)")
print(classification_report(video_true, video_pred, labels=LABEL_IDS, target_names=CLASS_NAMES, digits=4, zero_division=0))

cm_video = confusion_matrix(video_true, video_pred, labels=LABEL_IDS)
disp_video = ConfusionMatrixDisplay(confusion_matrix=cm_video, display_labels=CLASS_NAMES)
fig, ax = plt.subplots(figsize=(8, 6))
disp_video.plot(ax=ax, xticks_rotation=45, values_format="d")
plt.title("Video-Level Confusion Matrix After Majority Voting\nI-Frame Extraction + DeepRes-50")
plt.tight_layout()
plt.show()

print("\nper-video predictions")
for video_id, true_idx, pred_idx in zip(video_ids_ordered, video_true, video_pred):
    print(f"{video_id:50s} | true: {CLASS_NAMES[true_idx]:8s} | pred: {CLASS_NAMES[pred_idx]}")
