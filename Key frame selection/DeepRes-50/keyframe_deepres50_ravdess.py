# =====================================================================
# 1. IMPORTS
# =====================================================================

import csv
import json
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
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.layers import (
    Activation,
    Add,
    AveragePooling2D,
    BatchNormalization,
    Conv2D,
    Dense,
    Flatten,
    Input,
    MaxPooling2D,
    ZeroPadding2D,
)
from tensorflow.keras.optimizers import Adam


# =====================================================================
# 2. CONFIGURATION
# =====================================================================

RAVDESS_VIDEO_DIR = Path(r"C:\PATH\TO\RAVDESS\VIDEOS")
KEYFRAME_DIR = Path(r"C:\PATH\TO\RAVDESS_KEYFRAME\selected_keyframes")
ALIGNED_KEYFRAME_DIR = Path(r"C:\PATH\TO\RAVDESS_KEYFRAME\aligned_keyframes")
DLIB_LANDMARK_MODEL = Path(r"C:\PATH\TO\shape_predictor_68_face_landmarks.dat")
OUTPUT_DIR = Path(r"C:\PATH\TO\RAVDESS_KEYFRAME\deepres50_outputs")

# True only when key frames / aligned faces need to be generated.
RUN_KEYFRAME_PIPELINE = False
OVERWRITE_PREPARED_DATA = False

# Save raw selected frames in addition to aligned facial crops.
# Set False to save disk space when only aligned frames are needed.
SAVE_RAW_KEYFRAMES = True

VIDEO_EXTENSIONS = {
    ".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg", ".m4v"
}

CLASS_NAMES = ["Anger", "Disgust", "Fear", "Happy", "Sad"]
NUM_CLASSES = len(CLASS_NAMES)
CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}

# RAVDESS filename format:
# modality-vocal_channel-emotion-intensity-statement-repetition-actor
#
# Emotion code (third field):
# 01 = neutral, 02 = calm, 03 = happy, 04 = sad,
# 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised.
#
# Only the five common emotions retained in the paper are used.
RAVDESS_EMOTION_CODE_TO_CLASS = {
    "03": "Happy",
    "04": "Sad",
    "05": "Anger",
    "06": "Fear",
    "07": "Disgust",
}

IMG_SIZE = (48, 48)
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
VAL_SPLIT = 0.20
EARLY_STOPPING_PATIENCE = 5
SEED = 123

# Mean Euclidean displacement across the 68 corresponding landmarks.
LANDMARK_DISPLACEMENT_THRESHOLD_PX = 5.0

# DLib parameters.
DLIB_UPSAMPLE = 1
FACE_CHIP_PADDING = 0.25

# Training speed optimizations.
# Mixed precision is useful on modern NVIDIA GPUs such as RTX cards.
USE_MIXED_PRECISION = True
STEPS_PER_EXECUTION = 16


# =====================================================================
# 3. REPRODUCIBILITY / GPU SETUP
# =====================================================================

os.environ.setdefault("PYTHONHASHSEED", str(SEED))
random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except (AttributeError, RuntimeError):
    pass

gpus = tf.config.list_physical_devices("GPU")
print("TensorFlow version:", tf.__version__)
print("GPU(s) detected:", gpus)

if USE_MIXED_PRECISION and gpus:
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy("mixed_float16")
    print("Mixed precision enabled:", mixed_precision.global_policy())
else:
    print("Mixed precision disabled.")


# =====================================================================
# 4. DATASET / VIDEO IDENTIFICATION
# =====================================================================

def infer_class_from_relative_path(relative_path: Path):
    """
    Infer the emotion class from the official RAVDESS filename.

    Example:
        Actor_01/01-01-05-01-01-01-01.mp4
                       ^^
                       05 -> Anger

    Neutral (01), calm (02), and surprised (08) clips are ignored because
    the paper uses only Anger, Disgust, Fear, Happy, and Sad.
    """
    fields = relative_path.stem.split("-")

    if len(fields) < 3:
        return None

    emotion_code = fields[2]
    return RAVDESS_EMOTION_CODE_TO_CLASS.get(emotion_code)


def make_video_id(relative_path: Path):
    """
    Create a stable unique RAVDESS source-video ID.

    Example:
        Actor_01/01-01-05-01-01-01-01.mp4
        -> Actor_01__01-01-05-01-01-01-01
    """
    return "__".join(relative_path.with_suffix("").parts)


def discover_dataset_videos(video_root: Path):
    """Return (video_path, relative_path, class_name, video_id) records."""
    video_root = Path(video_root)

    if not video_root.exists():
        raise FileNotFoundError(
            f"RAVDESS video directory does not exist:\n{video_root}"
        )

    videos = sorted(
        p for p in video_root.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )

    if not videos:
        raise FileNotFoundError(f"No supported videos found under:\n{video_root}")

    selected = []
    ignored = []

    for video_path in videos:
        relative = video_path.relative_to(video_root)
        class_name = infer_class_from_relative_path(relative)

        if class_name is None:
            ignored.append(relative)
            continue

        video_id = make_video_id(relative)
        selected.append((video_path, relative, class_name, video_id))

    if not selected:
        raise RuntimeError(
            f"No videos matched the retained classes: {CLASS_NAMES}"
        )

    print(f"Total video files found         : {len(videos)}")
    print(f"Videos in retained five classes: {len(selected)}")
    print(f"Ignored RAVDESS clips           : {len(ignored)}")

    return selected


# =====================================================================
# 5. FFMPEG DECODER
# =====================================================================

def check_ffmpeg_available():
    missing = [
        exe for exe in ("ffmpeg", "ffprobe")
        if shutil.which(exe) is None
    ]
    if missing:
        raise RuntimeError(
            "Missing executable(s) on PATH: " + ", ".join(missing)
        )


def get_video_dimensions(video_path: Path):
    """Get source-video width and height using ffprobe."""
    check_ffmpeg_available()

    command = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=s=x:p=0",
        str(video_path),
    ]

    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )

    dimensions = result.stdout.strip()
    if "x" not in dimensions:
        raise RuntimeError(f"Could not determine dimensions: {video_path}")

    width, height = map(int, dimensions.split("x"))
    return width, height


def _read_exact(stream, n_bytes):
    chunks = []
    remaining = n_bytes

    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)

    data = b"".join(chunks)
    return data if len(data) == n_bytes else None


def decode_video_frames_ffmpeg(video_path: Path):
    """
    Stream decoded frames directly from FFmpeg as BGR uint8 arrays.

    Frames are not first written to disk, reducing temporary disk I/O.
    """
    width, height = get_video_dimensions(video_path)
    bytes_per_frame = width * height * 3

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-i", str(video_path),
        "-map", "0:v:0",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "pipe:1",
    ]

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=10**7,
    )

    frame_index = 0

    try:
        while True:
            raw = _read_exact(process.stdout, bytes_per_frame)
            if raw is None:
                break

            frame_index += 1
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                height, width, 3
            )
            yield frame_index, frame

    finally:
        if process.stdout is not None:
            process.stdout.close()

        stderr = b""
        if process.stderr is not None:
            stderr = process.stderr.read()
            process.stderr.close()

        return_code = process.wait()

        if return_code != 0:
            message = stderr.decode("utf-8", errors="replace")
            raise RuntimeError(
                f"FFmpeg decoding failed for {video_path}\n{message}"
            )


# =====================================================================
# 6. DLIB FACE DETECTION + 68 LANDMARKS
# =====================================================================

_face_detector = dlib.get_frontal_face_detector()
_landmark_predictor = None


def get_landmark_predictor():
    global _landmark_predictor

    if _landmark_predictor is None:
        if not DLIB_LANDMARK_MODEL.exists():
            raise FileNotFoundError(
                f"DLib predictor not found:\n{DLIB_LANDMARK_MODEL}\n"
                "Download shape_predictor_68_face_landmarks.dat and update "
                "DLIB_LANDMARK_MODEL."
            )

        _landmark_predictor = dlib.shape_predictor(
            str(DLIB_LANDMARK_MODEL)
        )

    return _landmark_predictor


def detect_face_shape_landmarks(image_bgr):
    """
    Detect the largest face once and return:
        rectangle, DLib shape object, 68x2 landmark array.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = _face_detector(gray, DLIB_UPSAMPLE)

    if not faces:
        return None, None, None

    face = max(
        faces,
        key=lambda r: max(0, r.width()) * max(0, r.height())
    )

    shape = get_landmark_predictor()(gray, face)

    landmarks = np.fromiter(
        (
            coordinate
            for i in range(68)
            for coordinate in (shape.part(i).x, shape.part(i).y)
        ),
        dtype=np.float32,
        count=136,
    ).reshape(68, 2)

    return face, shape, landmarks


# =====================================================================
# 7. KEY-FRAME SELECTION
# =====================================================================

def mean_landmark_displacement(current_landmarks, reference_landmarks):
    """
    Mean 68-point Euclidean displacement in pixels:

        D = mean_i ||p_current,i - p_reference,i||_2
    """
    return float(
        np.linalg.norm(
            current_landmarks - reference_landmarks,
            axis=1,
        ).mean()
    )


def align_crop_resize_face(image_bgr, shape):
    """
    Facial preprocessing:
        landmark-guided alignment -> crop -> 48x48 resize.

    Reuses the SAME shape object already obtained during key-frame selection.
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    aligned_rgb = dlib.get_face_chip(
        image_rgb,
        shape,
        size=IMG_SIZE[0],
        padding=FACE_CHIP_PADDING,
    )

    aligned_bgr = cv2.cvtColor(aligned_rgb, cv2.COLOR_RGB2BGR)

    if aligned_bgr.shape[:2] != IMG_SIZE:
        aligned_bgr = cv2.resize(
            aligned_bgr,
            IMG_SIZE,
            interpolation=cv2.INTER_LINEAR,
        )

    return aligned_bgr


def landmark_csv_header():
    header = [
        "selected_index",
        "decoded_frame_index",
        "keyframe_file",
        "mean_landmark_displacement_px",
    ]

    for i in range(68):
        header.extend([f"x_{i}", f"y_{i}"])

    return header


def completion_marker_matches(marker_path: Path):
    """Skip only when an earlier completed run used the same key settings."""
    if not marker_path.exists():
        return False

    try:
        data = json.loads(marker_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False

    return (
        float(data.get("threshold_px", -1))
        == float(LANDMARK_DISPLACEMENT_THRESHOLD_PX)
        and tuple(data.get("img_size", [])) == tuple(IMG_SIZE)
    )


def prepare_video_keyframes(
    video_path: Path,
    raw_output_dir: Path,
    aligned_output_dir: Path,
    overwrite=False,
):
    """
    ONE-PASS optimized key-frame selection + facial preprocessing.

    The DLib face detector and 68-landmark predictor are executed once
    per decoded frame. For a selected key frame, the already computed
    DLib shape is reused for facial alignment/cropping.
    """
    raw_output_dir = Path(raw_output_dir)
    aligned_output_dir = Path(aligned_output_dir)

    raw_output_dir.mkdir(parents=True, exist_ok=True)
    aligned_output_dir.mkdir(parents=True, exist_ok=True)

    marker_path = aligned_output_dir / "_complete.json"
    metadata_path = aligned_output_dir / "keyframe_landmarks.csv"

    if not overwrite and completion_marker_matches(marker_path):
        existing = sorted(aligned_output_dir.glob("keyframe_*.png"))
        if existing:
            print(
                f"Skipping {video_path.name}: "
                f"{len(existing)} prepared key frame(s) already exist."
            )
            return len(existing)

    if overwrite:
        for folder in (raw_output_dir, aligned_output_dir):
            for file_path in folder.glob("keyframe_*.png"):
                file_path.unlink()

        for path in (marker_path, metadata_path):
            if path.exists():
                path.unlink()

    selected_count = 0
    decoded_count = 0
    face_count = 0
    last_selected_landmarks = None

    with open(metadata_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(landmark_csv_header())

        for frame_index, frame_bgr in decode_video_frames_ffmpeg(video_path):
            decoded_count += 1

            _, shape, landmarks = detect_face_shape_landmarks(frame_bgr)

            if landmarks is None:
                continue

            face_count += 1

            if last_selected_landmarks is None:
                displacement = 0.0
                is_keyframe = True
            else:
                displacement = mean_landmark_displacement(
                    landmarks,
                    last_selected_landmarks,
                )
                is_keyframe = (
                    displacement >= LANDMARK_DISPLACEMENT_THRESHOLD_PX
                )

            if not is_keyframe:
                continue

            selected_count += 1

            filename = (
                f"keyframe_{selected_count:05d}"
                f"_frame_{frame_index:06d}.png"
            )

            # Reuse the already computed shape. No second DLib pass.
            aligned_face = align_crop_resize_face(frame_bgr, shape)

            aligned_path = aligned_output_dir / filename
            if not cv2.imwrite(str(aligned_path), aligned_face):
                raise RuntimeError(f"Could not write: {aligned_path}")

            if SAVE_RAW_KEYFRAMES:
                raw_path = raw_output_dir / filename
                if not cv2.imwrite(str(raw_path), frame_bgr):
                    raise RuntimeError(f"Could not write: {raw_path}")

            writer.writerow(
                [
                    selected_count,
                    frame_index,
                    filename,
                    displacement,
                    *landmarks.reshape(-1).tolist(),
                ]
            )

            last_selected_landmarks = landmarks.copy()

    marker = {
        "source_video": str(video_path),
        "threshold_px": LANDMARK_DISPLACEMENT_THRESHOLD_PX,
        "img_size": list(IMG_SIZE),
        "decoded_frames": decoded_count,
        "frames_with_face": face_count,
        "selected_keyframes": selected_count,
    }

    marker_path.write_text(
        json.dumps(marker, indent=2),
        encoding="utf-8",
    )

    print(
        f"{video_path.name}: decoded={decoded_count}, "
        f"face={face_count}, selected={selected_count}, "
        f"threshold={LANDMARK_DISPLACEMENT_THRESHOLD_PX:.2f}px"
    )

    return selected_count


def prepare_keyframe_dataset(video_root: Path):
    """Run optimized one-pass preparation for all retained RAVDESS videos."""
    records = discover_dataset_videos(video_root)
    total_selected = 0

    for i, (video_path, relative, class_name, video_id) in enumerate(
        records,
        start=1,
    ):
        print(f"[{i}/{len(records)}] {relative}")

        raw_dir = KEYFRAME_DIR / class_name / video_id
        aligned_dir = ALIGNED_KEYFRAME_DIR / class_name / video_id

        total_selected += prepare_video_keyframes(
            video_path=video_path,
            raw_output_dir=raw_dir,
            aligned_output_dir=aligned_dir,
            overwrite=OVERWRITE_PREPARED_DATA,
        )

    print("\nKey-frame preparation completed.")
    print(f"Videos processed          : {len(records)}")
    print(f"Total selected key frames : {total_selected}")
    print(f"Threshold                 : {LANDMARK_DISPLACEMENT_THRESHOLD_PX}px")
    print(f"Aligned output            : {ALIGNED_KEYFRAME_DIR}")


if RUN_KEYFRAME_PIPELINE:
    prepare_keyframe_dataset(RAVDESS_VIDEO_DIR)


# =====================================================================
# 8. BUILD ALIGNED KEY-FRAME INDEX
# =====================================================================

def build_keyframe_index(aligned_root: Path):
    records = []

    if not aligned_root.exists():
        raise FileNotFoundError(
            f"Aligned key-frame directory not found:\n{aligned_root}\n"
            "Set RUN_KEYFRAME_PIPELINE=True for the first run."
        )

    for class_name in CLASS_NAMES:
        class_dir = aligned_root / class_name

        if not class_dir.exists():
            print(f"WARNING: missing class directory: {class_dir}")
            continue

        class_index = CLASS_TO_INDEX[class_name]

        for video_dir in sorted(p for p in class_dir.iterdir() if p.is_dir()):
            frames = sorted(video_dir.glob("keyframe_*.png"))

            if not frames:
                continue

            video_id = f"{class_name}/{video_dir.name}"

            for frame_path in frames:
                records.append((str(frame_path), class_index, video_id))

    if not records:
        raise FileNotFoundError(
            f"No aligned key frames found under:\n{aligned_root}"
        )

    return records


records = build_keyframe_index(ALIGNED_KEYFRAME_DIR)

frame_paths = np.asarray([r[0] for r in records])
frame_labels = np.asarray([r[1] for r in records], dtype=np.int32)
frame_video_ids = np.asarray([r[2] for r in records])


# =====================================================================
# 9. VIDEO-LEVEL STRATIFIED 80:20 SPLIT
# =====================================================================

def split_videos_stratified(video_ids, labels):
    video_to_label = {}

    for video_id, label in zip(video_ids, labels):
        label = int(label)

        previous = video_to_label.get(video_id)
        if previous is not None and previous != label:
            raise ValueError(f"Conflicting labels for video: {video_id}")

        video_to_label[video_id] = label

    unique_videos = np.asarray(list(video_to_label.keys()))
    unique_labels = np.asarray(
        [video_to_label[v] for v in unique_videos],
        dtype=np.int32,
    )

    class_counts = Counter(unique_labels.tolist())

    missing = [
        CLASS_NAMES[i]
        for i in range(NUM_CLASSES)
        if class_counts.get(i, 0) == 0
    ]
    if missing:
        raise ValueError("Missing usable class(es): " + ", ".join(missing))

    too_small = {
        CLASS_NAMES[i]: count
        for i, count in class_counts.items()
        if count < 2
    }
    if too_small:
        raise ValueError(
            "At least two videos/class are required: "
            f"{too_small}"
        )

    train_videos, val_videos = train_test_split(
        unique_videos,
        test_size=VAL_SPLIT,
        random_state=SEED,
        shuffle=True,
        stratify=unique_labels,
    )

    if np.intersect1d(train_videos, val_videos).size:
        raise RuntimeError("Video-level train/validation leakage detected.")

    return np.asarray(train_videos), np.asarray(val_videos), video_to_label


train_video_ids, val_video_ids_unique, video_to_label = (
    split_videos_stratified(frame_video_ids, frame_labels)
)

train_mask = np.isin(frame_video_ids, train_video_ids)
val_mask = np.isin(frame_video_ids, val_video_ids_unique)

train_paths = frame_paths[train_mask]
train_labels = frame_labels[train_mask]

val_paths = frame_paths[val_mask]
val_labels = frame_labels[val_mask]
val_video_ids = frame_video_ids[val_mask]

print("\nDataset split")
print("-------------")
print(f"Total videos      : {len(video_to_label)}")
print(f"Training videos   : {len(train_video_ids)}")
print(f"Validation videos : {len(val_video_ids_unique)}")
print(f"Training frames   : {len(train_paths)}")
print(f"Validation frames : {len(val_paths)}")


# =====================================================================
# 10. OPTIMIZED tf.data PIPELINE
# =====================================================================

def load_image(path, label):
    image = tf.io.decode_png(
        tf.io.read_file(path),
        channels=3,
    )

    # Images are already 48x48; resize is a defensive consistency check.
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0

    return image, label


def make_dataset(paths, labels, training):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        ds = ds.shuffle(
            buffer_size=len(paths),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    ds = ds.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=not training,
    )

    ds = ds.batch(
        BATCH_SIZE,
        drop_remainder=False,
    )

    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


train_ds = make_dataset(train_paths, train_labels, training=True)
val_ds = make_dataset(val_paths, val_labels, training=False)


# =====================================================================
# 11. DEEPRES-50 BLOCKS
# =====================================================================

def identity_block(X, f, filters, stage, block):
    conv_name = f"res{stage}{block}_"
    bn_name = f"bn{stage}{block}_"
    f1, f2, f3 = filters

    shortcut = X

    X = Conv2D(
        f1, 1,
        kernel_initializer=glorot_uniform(seed=0),
        name=conv_name + "first_component",
    )(X)
    X = BatchNormalization(axis=3, name=bn_name + "first_component")(X)
    X = Activation("relu")(X)

    X = Conv2D(
        f2, f,
        padding="same",
        kernel_initializer=glorot_uniform(seed=0),
        name=conv_name + "second_component",
    )(X)
    X = BatchNormalization(axis=3, name=bn_name + "second_component")(X)
    X = Activation("relu")(X)

    X = Conv2D(
        f3, 1,
        kernel_initializer=glorot_uniform(seed=0),
        name=conv_name + "third_component",
    )(X)
    X = BatchNormalization(axis=3, name=bn_name + "third_component")(X)

    X = Add()([X, shortcut])
    return Activation("relu")(X)


def convolutional_block(X, f, filters, stage, block, s=2):
    conv_name = f"res{stage}{block}_"
    bn_name = f"bn{stage}{block}_"
    f1, f2, f3 = filters

    shortcut = X

    X = Conv2D(
        f1, 1,
        strides=s,
        kernel_initializer=glorot_uniform(seed=0),
        name=conv_name + "first_component",
    )(X)
    X = BatchNormalization(axis=3, name=bn_name + "first_component")(X)
    X = Activation("relu")(X)

    X = Conv2D(
        f2, f,
        padding="same",
        kernel_initializer=glorot_uniform(seed=0),
        name=conv_name + "second_component",
    )(X)
    X = BatchNormalization(axis=3, name=bn_name + "second_component")(X)
    X = Activation("relu")(X)

    X = Conv2D(
        f3, 1,
        kernel_initializer=glorot_uniform(seed=0),
        name=conv_name + "third_component",
    )(X)
    X = BatchNormalization(axis=3, name=bn_name + "third_component")(X)

    shortcut = Conv2D(
        f3, 1,
        strides=s,
        kernel_initializer=glorot_uniform(seed=0),
        name=conv_name + "merge",
    )(shortcut)
    shortcut = BatchNormalization(axis=3, name=bn_name + "merge")(shortcut)

    X = Add()([X, shortcut])
    return Activation("relu")(X)


# =====================================================================
# 12. DEEPRES-50
# =====================================================================

def DeepRes50(input_shape=(48, 48, 3), classes=NUM_CLASSES):
    X_input = Input(shape=input_shape, name="input_image")

    X = ZeroPadding2D((3, 3))(X_input)
    X = Conv2D(
        64, 7,
        strides=2,
        kernel_initializer=glorot_uniform(seed=0),
        name="conv_1",
    )(X)
    X = BatchNormalization(axis=3, name="bn_1")(X)
    X = Activation("relu")(X)
    X = MaxPooling2D(3, strides=2)(X)

    X = convolutional_block(X, 3, [64, 64, 256], 2, "a", s=1)
    X = identity_block(X, 3, [64, 64, 256], 2, "b")
    X = identity_block(X, 3, [64, 64, 256], 2, "c")

    X = convolutional_block(X, 3, [128, 128, 512], 3, "a", s=2)
    for block in ("b", "c", "d"):
        X = identity_block(X, 3, [128, 128, 512], 3, block)

    X = convolutional_block(X, 3, [256, 256, 1024], 4, "a", s=2)
    for block in ("b", "c", "d", "e", "f"):
        X = identity_block(X, 3, [256, 256, 1024], 4, block)

    X = convolutional_block(X, 3, [512, 512, 2048], 5, "a", s=2)
    X = identity_block(X, 3, [512, 512, 2048], 5, "b")
    X = identity_block(X, 3, [512, 512, 2048], 5, "c")

    X = AveragePooling2D((2, 2), name="avg_pool")(X)
    X = Flatten(name="flatten")(X)

    # float32 output is safer when mixed precision is active.
    X = Dense(
        classes,
        activation="softmax",
        dtype="float32",
        kernel_initializer=glorot_uniform(seed=0),
        name=f"fc{classes}",
    )(X)

    return Model(X_input, X, name="DeepRes-50")


# =====================================================================
# 13. TRAINING
# =====================================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

model = DeepRes50(
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    classes=NUM_CLASSES,
)

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
    steps_per_execution=STEPS_PER_EXECUTION,
)

checkpoint_path = OUTPUT_DIR / "best_deepres50.keras"

callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1,
    ),
    ModelCheckpoint(
        filepath=str(checkpoint_path),
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    ),
]

model.summary()

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
)


# =====================================================================
# 14. TRAINING CURVES
# =====================================================================

def save_training_curves(history, output_dir: Path):
    plt.figure(figsize=(8, 6))
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("DeepRes-50 Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "training_accuracy.png")
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("DeepRes-50 Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "training_loss.png")
    plt.show()


save_training_curves(history, OUTPUT_DIR)


# =====================================================================
# 15. FRAME-LEVEL EVALUATION
# =====================================================================

LABEL_IDS = np.arange(NUM_CLASSES)

frame_probabilities = model.predict(val_ds, verbose=1)
frame_predictions = np.argmax(frame_probabilities, axis=1).astype(np.int32)
frame_true = val_labels.astype(np.int32)

if not (
    len(frame_predictions)
    == len(frame_true)
    == len(val_video_ids)
):
    raise RuntimeError(
        "Validation predictions, labels, and video IDs are misaligned."
    )

frame_accuracy = accuracy_score(frame_true, frame_predictions)

print("\nFrame-Level Evaluation")
print("----------------------")
print(f"Accuracy: {frame_accuracy * 100:.4f}%")

frame_report = classification_report(
    frame_true,
    frame_predictions,
    labels=LABEL_IDS,
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0,
)

print(frame_report)

(OUTPUT_DIR / "frame_level_classification_report.txt").write_text(
    frame_report,
    encoding="utf-8",
)

frame_cm = confusion_matrix(
    frame_true,
    frame_predictions,
    labels=LABEL_IDS,
)

fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay(
    frame_cm,
    display_labels=CLASS_NAMES,
).plot(
    ax=ax,
    xticks_rotation=45,
    values_format="d",
)

plt.title("Frame-Level Confusion Matrix")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "frame_level_confusion_matrix.png")
plt.show()


# =====================================================================
# 16. VIDEO-LEVEL MAJORITY VOTING
# =====================================================================

def majority_vote_per_video(video_ids, probabilities, true_labels):
    video_ids = np.asarray(video_ids)
    probabilities = np.asarray(probabilities)
    true_labels = np.asarray(true_labels, dtype=np.int32)

    frame_predictions = np.argmax(probabilities, axis=1).astype(np.int32)

    video_true = []
    video_pred = []
    ordered_video_ids = np.unique(video_ids)

    for video_id in ordered_video_ids:
        mask = video_ids == video_id

        preds = frame_predictions[mask]
        probs = probabilities[mask]
        truth = np.unique(true_labels[mask])

        if len(truth) != 1:
            raise ValueError(
                f"Inconsistent true labels for video '{video_id}': "
                f"{truth.tolist()}"
            )

        votes = np.bincount(preds, minlength=NUM_CLASSES)
        tied = np.flatnonzero(votes == votes.max())

        if len(tied) == 1:
            final_prediction = int(tied[0])
        else:
            mean_probs = probs.mean(axis=0)
            final_prediction = int(
                tied[np.argmax(mean_probs[tied])]
            )

        video_true.append(int(truth[0]))
        video_pred.append(final_prediction)

    return (
        np.asarray(video_true, dtype=np.int32),
        np.asarray(video_pred, dtype=np.int32),
        ordered_video_ids,
    )


video_true, video_pred, ordered_video_ids = majority_vote_per_video(
    val_video_ids,
    frame_probabilities,
    frame_true,
)


# =====================================================================
# 17. VIDEO-LEVEL EVALUATION
# =====================================================================

video_accuracy = accuracy_score(video_true, video_pred)

print("\nVideo-Level Evaluation After Majority Voting")
print("--------------------------------------------")
print(f"Videos evaluated: {len(ordered_video_ids)}")
print(f"Accuracy        : {video_accuracy * 100:.4f}%")

video_report = classification_report(
    video_true,
    video_pred,
    labels=LABEL_IDS,
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0,
)

print(video_report)

(OUTPUT_DIR / "video_level_classification_report.txt").write_text(
    video_report,
    encoding="utf-8",
)

video_cm = confusion_matrix(
    video_true,
    video_pred,
    labels=LABEL_IDS,
)

fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay(
    video_cm,
    display_labels=CLASS_NAMES,
).plot(
    ax=ax,
    xticks_rotation=45,
    values_format="d",
)

plt.title("Video-Level Confusion Matrix After Majority Voting")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "video_level_confusion_matrix.png")
plt.show()


# =====================================================================
# 18. SAVE FINAL VIDEO PREDICTIONS
# =====================================================================

prediction_csv = OUTPUT_DIR / "video_level_predictions.csv"

with open(prediction_csv, "w", newline="", encoding="utf-8") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["video_id", "true_label", "predicted_label"])

    print("\nFinal Per-Video Predictions")
    print("---------------------------")

    for video_id, true_idx, pred_idx in zip(
        ordered_video_ids,
        video_true,
        video_pred,
    ):
        true_label = CLASS_NAMES[true_idx]
        pred_label = CLASS_NAMES[pred_idx]

        print(
            f"{video_id:50s} | "
            f"True: {true_label:8s} | "
            f"Predicted: {pred_label}"
        )

        writer.writerow([video_id, true_label, pred_label])
