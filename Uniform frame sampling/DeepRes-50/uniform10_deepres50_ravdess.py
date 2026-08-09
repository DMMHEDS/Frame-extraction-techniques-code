# =====================================================================
# 1. IMPORTS
# =====================================================================

import csv
import os
import random
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

# RAVDESS root containing Actor_01 ... Actor_24 folders.
RAVDESS_VIDEO_DIR = Path(
    r"D:\IIT mandi\Facial datasets\RAVDESS\RAVDESS"
)

# Preprocessed uniformly sampled frames will be stored here.
ALIGNED_FRAME_DIR = Path(
    r"D:\IIT mandi\Facial datasets\RAVDESS_UNIFORM10\aligned_frames"
)

# DLib 68-point facial landmark model.
DLIB_LANDMARK_MODEL = Path(
    r"C:\PATH\TO\shape_predictor_68_face_landmarks.dat"
)

# Model/results output.
OUTPUT_DIR = Path(
    r"D:\IIT mandi\Facial datasets\RAVDESS_UNIFORM10\deepres50_outputs"
)

# First run:
#     RUN_FRAME_PREPARATION = True
# Later training-only runs:
#     RUN_FRAME_PREPARATION = False
RUN_FRAME_PREPARATION = False
OVERWRITE_PREPARED_DATA = False

VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".mpeg",
    ".mpg",
    ".m4v",
}

CLASS_NAMES = [
    "Anger",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
]

NUM_CLASSES = len(CLASS_NAMES)

CLASS_TO_INDEX = {
    class_name: index
    for index, class_name in enumerate(CLASS_NAMES)
}

# Official RAVDESS emotion code -> retained paper class.
RAVDESS_EMOTION_CODE_TO_CLASS = {
    "03": "Happy",
    "04": "Sad",
    "05": "Anger",
    "06": "Fear",
    "07": "Disgust",
}

# Modality 02 = video-only in RAVDESS.
RAVDESS_ALLOWED_MODALITIES = {"02"}

# Uniform Frame Sampling parameter.
FRAME_INTERVAL = 10

# Facial preprocessing.
IMG_SIZE = (48, 48)
FACE_CHIP_PADDING = 0.25
DLIB_UPSAMPLE = 1

# Training settings retained close to the supplied DeepRes-50 code.
VAL_SPLIT = 0.20
SEED = 42
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001


# =====================================================================
# 3. REPRODUCIBILITY / GPU SETUP
# =====================================================================

os.environ.setdefault(
    "PYTHONHASHSEED",
    str(SEED),
)

random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except (AttributeError, RuntimeError):
    pass

print("TensorFlow version:", tf.__version__)
print("GPU(s) detected:", tf.config.list_physical_devices("GPU"))


# =====================================================================
# 4. RAVDESS DATASET DISCOVERY
# =====================================================================

def parse_ravdess_filename(video_path: Path):
    """
    Parse the official RAVDESS filename:

        modality-vocal_channel-emotion-intensity-statement-repetition-actor

    Example:
        02-01-05-01-01-01-01.mp4

    Returns
    -------
    class_name, modality

    Only emotion codes 03-07 are retained:
        03 = Happy
        04 = Sad
        05 = Anger
        06 = Fear
        07 = Disgust
    """
    parts = video_path.stem.split("-")

    if len(parts) != 7:
        return None, None

    modality = parts[0]
    emotion_code = parts[2]

    class_name = RAVDESS_EMOTION_CODE_TO_CLASS.get(
        emotion_code
    )

    return class_name, modality


def make_video_id(relative_path: Path):
    """
    Build a stable RAVDESS source-video ID.

    Example:
        Actor_01/02-01-05-01-01-01-01.mp4
        -> Actor_01__02-01-05-01-01-01-01
    """
    return "__".join(
        relative_path
        .with_suffix("")
        .parts
    )


def discover_ravdess_videos(
    video_root: Path,
):
    """
    Find RAVDESS videos belonging to the five retained emotion classes.
    """
    video_root = Path(
        video_root
    )

    if not video_root.exists():
        raise FileNotFoundError(
            "RAVDESS video directory does not exist:\n"
            f"{video_root}"
        )

    videos = sorted(
        path
        for path in video_root.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower()
            in VIDEO_EXTENSIONS
        )
    )

    if not videos:
        raise FileNotFoundError(
            f"No supported videos were found under:\n{video_root}"
        )

    selected = []
    skipped_modality = 0
    skipped_emotion = 0
    malformed = 0

    for video_path in videos:
        class_name, modality = parse_ravdess_filename(
            video_path
        )

        if modality is None:
            malformed += 1
            continue

        if modality not in RAVDESS_ALLOWED_MODALITIES:
            skipped_modality += 1
            continue

        if class_name is None:
            skipped_emotion += 1
            continue

        relative_path = video_path.relative_to(
            video_root
        )

        video_id = make_video_id(
            relative_path
        )

        selected.append(
            (
                video_path,
                relative_path,
                class_name,
                video_id,
            )
        )

    if not selected:
        raise RuntimeError(
            "No RAVDESS videos matched the selected five-class protocol."
        )

    print(
        f"Total video files found : {len(videos)}"
    )

    print(
        f"Five-class videos       : {len(selected)}"
    )

    print(
        f"Skipped by modality     : {skipped_modality}"
    )

    print(
        f"Skipped other emotions  : {skipped_emotion}"
    )

    print(
        f"Malformed filenames     : {malformed}"
    )

    return selected


# =====================================================================
# 5. DLIB FACE DETECTION + 68 FACIAL LANDMARKS
# =====================================================================

_face_detector = (
    dlib.get_frontal_face_detector()
)

_landmark_predictor = None


def get_landmark_predictor():
    """
    Load the DLib 68-point facial landmark predictor only once.
    """
    global _landmark_predictor

    if _landmark_predictor is None:
        if not DLIB_LANDMARK_MODEL.exists():
            raise FileNotFoundError(
                "DLib predictor not found:\n"
                f"{DLIB_LANDMARK_MODEL}\n"
                "Update DLIB_LANDMARK_MODEL."
            )

        _landmark_predictor = (
            dlib.shape_predictor(
                str(
                    DLIB_LANDMARK_MODEL
                )
            )
        )

    return _landmark_predictor


def detect_face_shape_landmarks(
    frame_bgr,
):
    """
    Detect the largest face and localize its 68 facial landmarks.

    Returns
    -------
    face_rectangle, shape, landmarks

    landmarks:
        numpy array of shape (68, 2)
    """
    gray = cv2.cvtColor(
        frame_bgr,
        cv2.COLOR_BGR2GRAY,
    )

    faces = _face_detector(
        gray,
        DLIB_UPSAMPLE,
    )

    if len(faces) == 0:
        return (
            None,
            None,
            None,
        )

    face = max(
        faces,
        key=lambda rectangle:
            max(
                0,
                rectangle.width(),
            )
            * max(
                0,
                rectangle.height(),
            ),
    )

    shape = (
        get_landmark_predictor()(
            gray,
            face,
        )
    )

    landmarks = np.array(
        [
            [
                shape.part(index).x,
                shape.part(index).y,
            ]
            for index in range(68)
        ],
        dtype=np.float32,
    )

    return (
        face,
        shape,
        landmarks,
    )


def align_crop_resize_face(
    frame_bgr,
    shape,
):
    """
    Facial preprocessing:
        DLib landmark-guided alignment
        -> facial crop
        -> 48x48x3 resize
    """
    frame_rgb = cv2.cvtColor(
        frame_bgr,
        cv2.COLOR_BGR2RGB,
    )

    aligned_rgb = dlib.get_face_chip(
        frame_rgb,
        shape,
        size=IMG_SIZE[0],
        padding=FACE_CHIP_PADDING,
    )

    aligned_bgr = cv2.cvtColor(
        aligned_rgb,
        cv2.COLOR_RGB2BGR,
    )

    if aligned_bgr.shape[:2] != IMG_SIZE:
        aligned_bgr = cv2.resize(
            aligned_bgr,
            IMG_SIZE,
            interpolation=cv2.INTER_LINEAR,
        )

    return aligned_bgr


# =====================================================================
# 6. UNIFORM FRAME SAMPLING: EVERY 10TH FRAME
# =====================================================================

def prepare_video_uniform_frames(
    video_path: Path,
    aligned_output_dir: Path,
    frame_interval=FRAME_INTERVAL,
    overwrite=False,
):
    """
    Uniformly sample every Nth decoded frame and then perform
    DLib facial preprocessing.

    Selection rule:
        retain frame F_t if t mod N == 0

    For this experiment:
        N = 10
    """
    aligned_output_dir = Path(
        aligned_output_dir
    )

    aligned_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata_path = (
        aligned_output_dir
        / "uniform_sampling_metadata.csv"
    )

    completion_path = (
        aligned_output_dir
        / "_complete.txt"
    )

    existing_frames = sorted(
        aligned_output_dir.glob(
            "frame_*.png"
        )
    )

    if (
        existing_frames
        and completion_path.exists()
        and not overwrite
    ):
        print(
            f"Skipping {video_path.name}: "
            f"{len(existing_frames)} prepared sampled frame(s) exist."
        )

        return len(
            existing_frames
        )

    if overwrite:
        for frame_path in existing_frames:
            frame_path.unlink()

        if metadata_path.exists():
            metadata_path.unlink()

        if completion_path.exists():
            completion_path.unlink()

    cap = cv2.VideoCapture(
        str(
            video_path
        )
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    decoded_count = 0
    uniformly_selected_count = 0
    aligned_count = 0
    no_face_count = 0

    with open(
        metadata_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.writer(
            csv_file
        )

        header = [
            "sample_index",
            "decoded_frame_index",
            "output_file",
        ]

        for landmark_index in range(68):
            header.extend(
                [
                    f"x_{landmark_index}",
                    f"y_{landmark_index}",
                ]
            )

        writer.writerow(
            header
        )

        try:
            while True:
                ret, frame_bgr = (
                    cap.read()
                )

                if not ret:
                    break

                decoded_count += 1

                # -----------------------------------------------------
                # FRAME SELECTION:
                # uniformly retain every 10th decoded frame.
                # -----------------------------------------------------
                if (
                    decoded_count
                    % frame_interval
                    != 0
                ):
                    continue

                uniformly_selected_count += 1

                # -----------------------------------------------------
                # FACIAL PRE-PROCESSING:
                # performed only AFTER frame selection.
                # -----------------------------------------------------
                (
                    _,
                    shape,
                    landmarks,
                ) = detect_face_shape_landmarks(
                    frame_bgr
                )

                if shape is None:
                    no_face_count += 1
                    continue

                aligned_count += 1

                aligned_face = (
                    align_crop_resize_face(
                        frame_bgr,
                        shape,
                    )
                )

                filename = (
                    f"frame_{aligned_count:05d}"
                    f"_source_{decoded_count:06d}.png"
                )

                output_path = (
                    aligned_output_dir
                    / filename
                )

                if not cv2.imwrite(
                    str(
                        output_path
                    ),
                    aligned_face,
                ):
                    raise RuntimeError(
                        f"Could not save frame: {output_path}"
                    )

                writer.writerow(
                    [
                        aligned_count,
                        decoded_count,
                        filename,
                        *landmarks.reshape(-1).tolist(),
                    ]
                )

        finally:
            cap.release()

    completion_path.write_text(
        (
            f"source_video={video_path}\n"
            f"frame_interval={frame_interval}\n"
            f"decoded_frames={decoded_count}\n"
            f"uniformly_selected={uniformly_selected_count}\n"
            f"aligned_faces={aligned_count}\n"
            f"sampled_frames_without_face={no_face_count}\n"
        ),
        encoding="utf-8",
    )

    print(
        f"{video_path.name}: "
        f"decoded={decoded_count}, "
        f"every_{frame_interval}th={uniformly_selected_count}, "
        f"aligned={aligned_count}, "
        f"no_face={no_face_count}"
    )

    return aligned_count


def prepare_ravdess_uniform_dataset(
    video_root: Path,
):
    """
    Apply:
        uniform frame sampling (every 10th frame)
        -> facial preprocessing
    to every retained RAVDESS video.
    """
    videos = (
        discover_ravdess_videos(
            video_root
        )
    )

    total_aligned_frames = 0

    for index, (
        video_path,
        relative_path,
        class_name,
        video_id,
    ) in enumerate(
        videos,
        start=1,
    ):
        print(
            f"[{index}/{len(videos)}] "
            f"{relative_path}"
        )

        aligned_dir = (
            ALIGNED_FRAME_DIR
            / class_name
            / video_id
        )

        total_aligned_frames += (
            prepare_video_uniform_frames(
                video_path=video_path,
                aligned_output_dir=aligned_dir,
                frame_interval=FRAME_INTERVAL,
                overwrite=OVERWRITE_PREPARED_DATA,
            )
        )

    print(
        "\nUniform frame preparation completed."
    )

    print(
        f"Videos processed       : {len(videos)}"
    )

    print(
        f"Sampling interval       : every {FRAME_INTERVAL}th frame"
    )

    print(
        f"Aligned sampled frames : {total_aligned_frames}"
    )


if RUN_FRAME_PREPARATION:
    prepare_ravdess_uniform_dataset(
        RAVDESS_VIDEO_DIR
    )


# =====================================================================
# 7. BUILD PREPROCESSED FRAME INDEX
# =====================================================================

def build_frame_index(
    aligned_root: Path,
):
    """
    Read:
        aligned_frames/
            Anger/
                video_id/
                    frame_*.png
            ...
    """
    aligned_root = Path(
        aligned_root
    )

    if not aligned_root.exists():
        raise FileNotFoundError(
            "Aligned frame directory does not exist:\n"
            f"{aligned_root}\n"
            "Set RUN_FRAME_PREPARATION=True for the first run."
        )

    records = []

    for class_name in CLASS_NAMES:
        class_dir = (
            aligned_root
            / class_name
        )

        if not class_dir.exists():
            print(
                f"WARNING: missing class directory: {class_dir}"
            )
            continue

        class_index = (
            CLASS_TO_INDEX[
                class_name
            ]
        )

        for video_dir in sorted(
            path
            for path in class_dir.iterdir()
            if path.is_dir()
        ):
            frame_files = sorted(
                video_dir.glob(
                    "frame_*.png"
                )
            )

            if not frame_files:
                continue

            video_id = (
                f"{class_name}/"
                f"{video_dir.name}"
            )

            for frame_path in frame_files:
                records.append(
                    (
                        str(
                            frame_path
                        ),
                        class_index,
                        video_id,
                    )
                )

    if not records:
        raise FileNotFoundError(
            f"No preprocessed uniform frames found under:\n{aligned_root}"
        )

    return records


records = build_frame_index(
    ALIGNED_FRAME_DIR
)

frame_paths = np.asarray(
    [
        record[0]
        for record in records
    ]
)

frame_labels = np.asarray(
    [
        record[1]
        for record in records
    ],
    dtype=np.int32,
)

frame_video_ids = np.asarray(
    [
        record[2]
        for record in records
    ]
)

print(
    f"\nTotal aligned uniform frames: {len(records)}"
)


# =====================================================================
# 8. VIDEO-LEVEL STRATIFIED 80:20 SPLIT
# =====================================================================

def split_videos_stratified(
    video_ids,
    labels,
):
    """
    Split source videos instead of individual frames.

    This prevents sampled frames from one video appearing in both
    training and validation sets.
    """
    video_to_label = {}

    for video_id, label in zip(
        video_ids,
        labels,
    ):
        label = int(
            label
        )

        previous = (
            video_to_label.get(
                video_id
            )
        )

        if (
            previous is not None
            and previous != label
        ):
            raise ValueError(
                f"Conflicting labels for video: {video_id}"
            )

        video_to_label[
            video_id
        ] = label

    unique_videos = np.asarray(
        list(
            video_to_label.keys()
        )
    )

    unique_labels = np.asarray(
        [
            video_to_label[
                video_id
            ]
            for video_id
            in unique_videos
        ],
        dtype=np.int32,
    )

    counts = Counter(
        unique_labels.tolist()
    )

    missing = [
        CLASS_NAMES[index]
        for index
        in range(NUM_CLASSES)
        if counts.get(
            index,
            0,
        ) == 0
    ]

    if missing:
        raise ValueError(
            "Missing usable class(es): "
            + ", ".join(
                missing
            )
        )

    train_videos, val_videos = (
        train_test_split(
            unique_videos,
            test_size=VAL_SPLIT,
            random_state=SEED,
            shuffle=True,
            stratify=unique_labels,
        )
    )

    if np.intersect1d(
        train_videos,
        val_videos,
    ).size:
        raise RuntimeError(
            "Video-level train/validation leakage detected."
        )

    return (
        np.asarray(
            train_videos
        ),
        np.asarray(
            val_videos
        ),
    )


(
    train_video_ids,
    val_video_ids_unique,
) = split_videos_stratified(
    frame_video_ids,
    frame_labels,
)

train_mask = np.isin(
    frame_video_ids,
    train_video_ids,
)

val_mask = np.isin(
    frame_video_ids,
    val_video_ids_unique,
)

train_paths = frame_paths[
    train_mask
]

train_labels = frame_labels[
    train_mask
]

val_paths = frame_paths[
    val_mask
]

val_labels = frame_labels[
    val_mask
]

val_video_ids = frame_video_ids[
    val_mask
]

print(
    "\nDataset split"
)

print(
    "-------------"
)

print(
    f"Training videos   : {len(train_video_ids)}"
)

print(
    f"Validation videos : {len(val_video_ids_unique)}"
)

print(
    f"Training frames   : {len(train_paths)}"
)

print(
    f"Validation frames : {len(val_paths)}"
)


# =====================================================================
# 9. tf.data IMAGE PIPELINE
# =====================================================================

def load_image(
    path,
    label,
):
    image = tf.io.read_file(
        path
    )

    image = tf.io.decode_png(
        image,
        channels=3,
    )

    image = tf.image.resize(
        image,
        IMG_SIZE,
    )

    image = (
        tf.cast(
            image,
            tf.float32,
        )
        / 255.0
    )

    return (
        image,
        label,
    )


def make_dataset(
    paths,
    labels,
    training,
):
    dataset = (
        tf.data.Dataset
        .from_tensor_slices(
            (
                paths,
                labels,
            )
        )
    )

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(
                paths
            ),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    dataset = dataset.batch(
        BATCH_SIZE
    )

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset


train_ds = make_dataset(
    train_paths,
    train_labels,
    training=True,
)

val_ds = make_dataset(
    val_paths,
    val_labels,
    training=False,
)


# =====================================================================
# 10. DEEPRES-50: IDENTITY BLOCK
# =====================================================================

def identity_block(
    X,
    f,
    filters,
    stage,
    block,
):
    """
    Identity residual block.
    """
    conv_base_name = (
        "res"
        + str(stage)
        + block
        + "_"
    )

    bn_base_name = (
        "bn"
        + str(stage)
        + block
        + "_"
    )

    f1, f2, f3 = (
        filters
    )

    X_skip_connection = X

    X = Conv2D(
        filters=f1,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="valid",
        name=conv_base_name
        + "first_component",
        kernel_initializer=glorot_uniform(
            seed=0
        ),
    )(X)

    X = BatchNormalization(
        axis=3,
        name=bn_base_name
        + "first_component",
    )(X)

    X = Activation(
        "relu"
    )(X)

    X = Conv2D(
        filters=f2,
        kernel_size=(f, f),
        strides=(1, 1),
        padding="same",
        name=conv_base_name
        + "second_component",
        kernel_initializer=glorot_uniform(
            seed=0
        ),
    )(X)

    X = BatchNormalization(
        axis=3,
        name=bn_base_name
        + "second_component",
    )(X)

    X = Activation(
        "relu"
    )(X)

    X = Conv2D(
        filters=f3,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="valid",
        name=conv_base_name
        + "third_component",
        kernel_initializer=glorot_uniform(
            seed=0
        ),
    )(X)

    X = BatchNormalization(
        axis=3,
        name=bn_base_name
        + "third_component",
    )(X)

    X = Add()(
        [
            X,
            X_skip_connection,
        ]
    )

    X = Activation(
        "relu"
    )(X)

    return X


# =====================================================================
# 11. DEEPRES-50: CONVOLUTIONAL BLOCK
# =====================================================================

def convolutional_block(
    X,
    f,
    filters,
    stage,
    block,
    s=2,
):
    """
    Convolutional residual block.
    """
    conv_base_name = (
        "res"
        + str(stage)
        + block
        + "_"
    )

    bn_base_name = (
        "bn"
        + str(stage)
        + block
        + "_"
    )

    f1, f2, f3 = (
        filters
    )

    X_skip_connection = X

    X = Conv2D(
        f1,
        (1, 1),
        strides=(s, s),
        padding="valid",
        name=conv_base_name
        + "first_component",
        kernel_initializer=glorot_uniform(
            seed=0
        ),
    )(X)

    X = BatchNormalization(
        axis=3,
        name=bn_base_name
        + "first_component",
    )(X)

    X = Activation(
        "relu"
    )(X)

    X = Conv2D(
        f2,
        kernel_size=(f, f),
        strides=(1, 1),
        padding="same",
        name=conv_base_name
        + "second_component",
        kernel_initializer=glorot_uniform(
            seed=0
        ),
    )(X)

    X = BatchNormalization(
        axis=3,
        name=bn_base_name
        + "second_component",
    )(X)

    X = Activation(
        "relu"
    )(X)

    X = Conv2D(
        f3,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="valid",
        name=conv_base_name
        + "third_component",
        kernel_initializer=glorot_uniform(
            seed=0
        ),
    )(X)

    X = BatchNormalization(
        axis=3,
        name=bn_base_name
        + "third_component",
    )(X)

    X_skip_connection = (
        Conv2D(
            f3,
            (1, 1),
            strides=(s, s),
            padding="valid",
            name=conv_base_name
            + "merge",
            kernel_initializer=glorot_uniform(
                seed=0
            ),
        )(
            X_skip_connection
        )
    )

    X_skip_connection = (
        BatchNormalization(
            axis=3,
            name=bn_base_name
            + "merge",
        )(
            X_skip_connection
        )
    )

    X = Add()(
        [
            X,
            X_skip_connection,
        ]
    )

    X = Activation(
        "relu"
    )(X)

    return X


# =====================================================================
# 12. DEEPRES-50 MODEL
# =====================================================================

def DeepRes50(
    input_shape=(48, 48, 3),
    classes=NUM_CLASSES,
):
    """
    DeepRes-50 / ResNet-50 architecture used in the supplied code.
    """
    X_input = Input(
        input_shape
    )

    X = ZeroPadding2D(
        (3, 3)
    )(
        X_input
    )

    X = Conv2D(
        64,
        (7, 7),
        strides=(2, 2),
        name="conv_1",
        kernel_initializer=glorot_uniform(
            seed=0
        ),
    )(X)

    X = BatchNormalization(
        axis=3,
        name="bn_1",
    )(X)

    X = Activation(
        "relu"
    )(X)

    X = MaxPooling2D(
        (3, 3),
        strides=(2, 2),
    )(X)

    # Stage 2
    X = convolutional_block(
        X,
        f=3,
        filters=[
            64,
            64,
            256,
        ],
        stage=2,
        block="a",
        s=1,
    )

    X = identity_block(
        X,
        3,
        [
            64,
            64,
            256,
        ],
        stage=2,
        block="b",
    )

    X = identity_block(
        X,
        3,
        [
            64,
            64,
            256,
        ],
        stage=2,
        block="c",
    )

    # Stage 3
    X = convolutional_block(
        X,
        f=3,
        filters=[
            128,
            128,
            512,
        ],
        stage=3,
        block="a",
        s=2,
    )

    for block_name in (
        "b",
        "c",
        "d",
    ):
        X = identity_block(
            X,
            3,
            [
                128,
                128,
                512,
            ],
            stage=3,
            block=block_name,
        )

    # Stage 4
    X = convolutional_block(
        X,
        f=3,
        filters=[
            256,
            256,
            1024,
        ],
        stage=4,
        block="a",
        s=2,
    )

    for block_name in (
        "b",
        "c",
        "d",
        "e",
        "f",
    ):
        X = identity_block(
            X,
            3,
            [
                256,
                256,
                1024,
            ],
            stage=4,
            block=block_name,
        )

    # Stage 5
    X = convolutional_block(
        X,
        f=3,
        filters=[
            512,
            512,
            2048,
        ],
        stage=5,
        block="a",
        s=2,
    )

    X = identity_block(
        X,
        3,
        [
            512,
            512,
            2048,
        ],
        stage=5,
        block="b",
    )

    X = identity_block(
        X,
        3,
        [
            512,
            512,
            2048,
        ],
        stage=5,
        block="c",
    )

    X = AveragePooling2D(
        (2, 2),
        name="avg_pool",
    )(X)

    X = Flatten()(
        X
    )

    X = Dense(
        classes,
        activation="softmax",
        name="fc"
        + str(
            classes
        ),
        kernel_initializer=glorot_uniform(
            seed=0
        ),
    )(X)

    return Model(
        inputs=X_input,
        outputs=X,
        name="DeepRes-50",
    )


# =====================================================================
# 13. COMPILE / TRAIN
# =====================================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

model = DeepRes50(
    input_shape=(
        IMG_SIZE[0],
        IMG_SIZE[1],
        3,
    ),
    classes=NUM_CLASSES,
)

model.compile(
    optimizer=Adam(
        learning_rate=LEARNING_RATE
    ),
    loss="sparse_categorical_crossentropy",
    metrics=[
        "accuracy"
    ],
)

model.summary()

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
)

model.save(
    OUTPUT_DIR
    / "ravdess_uniform10_deepres50.keras"
)


# =====================================================================
# 14. SAVE TRAINING HISTORY / CURVES
# =====================================================================

history_csv = (
    OUTPUT_DIR
    / "training_history.csv"
)

with open(
    history_csv,
    "w",
    newline="",
    encoding="utf-8",
) as csv_file:
    writer = csv.writer(
        csv_file
    )

    writer.writerow(
        [
            "epoch",
            "train_loss",
            "train_accuracy",
            "val_loss",
            "val_accuracy",
        ]
    )

    for epoch_index in range(
        len(
            history.history[
                "loss"
            ]
        )
    ):
        writer.writerow(
            [
                epoch_index + 1,
                history.history[
                    "loss"
                ][
                    epoch_index
                ],
                history.history[
                    "accuracy"
                ][
                    epoch_index
                ],
                history.history[
                    "val_loss"
                ][
                    epoch_index
                ],
                history.history[
                    "val_accuracy"
                ][
                    epoch_index
                ],
            ]
        )

plt.figure(
    figsize=(8, 6)
)

plt.plot(
    history.history[
        "accuracy"
    ],
    label="Train Accuracy",
)

plt.plot(
    history.history[
        "val_accuracy"
    ],
    label="Validation Accuracy",
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Accuracy"
)

plt.title(
    "DeepRes-50 Accuracy - RAVDESS Uniform Sampling"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "training_accuracy.png"
)

plt.show()


plt.figure(
    figsize=(8, 6)
)

plt.plot(
    history.history[
        "loss"
    ],
    label="Train Loss",
)

plt.plot(
    history.history[
        "val_loss"
    ],
    label="Validation Loss",
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.title(
    "DeepRes-50 Loss - RAVDESS Uniform Sampling"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "training_loss.png"
)

plt.show()


# =====================================================================
# 15. FRAME-LEVEL EVALUATION
# =====================================================================

frame_probabilities = (
    model.predict(
        val_ds,
        verbose=1,
    )
)

frame_predictions = np.argmax(
    frame_probabilities,
    axis=1,
).astype(
    np.int32
)

frame_true = val_labels.astype(
    np.int32
)

if not (
    len(
        frame_predictions
    )
    == len(
        frame_true
    )
    == len(
        val_video_ids
    )
):
    raise RuntimeError(
        "Frame predictions, labels, and video IDs are misaligned."
    )

LABEL_IDS = np.arange(
    NUM_CLASSES
)

frame_accuracy = accuracy_score(
    frame_true,
    frame_predictions,
)

print(
    "\nFrame-Level Evaluation"
)

print(
    "----------------------"
)

print(
    f"Accuracy: {frame_accuracy * 100:.4f}%"
)

frame_report = classification_report(
    frame_true,
    frame_predictions,
    labels=LABEL_IDS,
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0,
)

print(
    frame_report
)

(
    OUTPUT_DIR
    / "frame_level_classification_report.txt"
).write_text(
    frame_report,
    encoding="utf-8",
)

frame_cm = confusion_matrix(
    frame_true,
    frame_predictions,
    labels=LABEL_IDS,
)

fig, ax = plt.subplots(
    figsize=(8, 6)
)

ConfusionMatrixDisplay(
    confusion_matrix=frame_cm,
    display_labels=CLASS_NAMES,
).plot(
    ax=ax,
    xticks_rotation=45,
    values_format="d",
)

plt.title(
    "Frame-Level Confusion Matrix\n"
    "Uniform Sampling Every 10th Frame"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "frame_level_confusion_matrix.png"
)

plt.show()


# =====================================================================
# 16. VIDEO-LEVEL MAJORITY VOTING
# =====================================================================

def majority_vote_per_video(
    video_ids,
    probabilities,
    true_labels,
):
    """
    Aggregate frame-wise predictions into one final prediction per video.

    Primary rule:
        class receiving the highest number of frame votes wins.

    Tie:
        among tied classes, use the class with the highest mean
        softmax probability for the video.
    """
    video_ids = np.asarray(
        video_ids
    )

    probabilities = np.asarray(
        probabilities
    )

    true_labels = np.asarray(
        true_labels,
        dtype=np.int32,
    )

    frame_predictions = np.argmax(
        probabilities,
        axis=1,
    ).astype(
        np.int32
    )

    ordered_video_ids = np.unique(
        video_ids
    )

    video_true = []
    video_pred = []

    for video_id in ordered_video_ids:
        mask = (
            video_ids
            == video_id
        )

        predictions_for_video = (
            frame_predictions[
                mask
            ]
        )

        probabilities_for_video = (
            probabilities[
                mask
            ]
        )

        truth = np.unique(
            true_labels[
                mask
            ]
        )

        if len(
            truth
        ) != 1:
            raise ValueError(
                f"Inconsistent true labels for video "
                f"'{video_id}': {truth.tolist()}"
            )

        votes = np.bincount(
            predictions_for_video,
            minlength=NUM_CLASSES,
        )

        tied_classes = np.flatnonzero(
            votes
            == votes.max()
        )

        if len(
            tied_classes
        ) == 1:
            final_prediction = int(
                tied_classes[
                    0
                ]
            )

        else:
            mean_probabilities = (
                probabilities_for_video.mean(
                    axis=0
                )
            )

            final_prediction = int(
                tied_classes[
                    np.argmax(
                        mean_probabilities[
                            tied_classes
                        ]
                    )
                ]
            )

        video_true.append(
            int(
                truth[
                    0
                ]
            )
        )

        video_pred.append(
            final_prediction
        )

    return (
        np.asarray(
            video_true,
            dtype=np.int32,
        ),
        np.asarray(
            video_pred,
            dtype=np.int32,
        ),
        ordered_video_ids,
    )


(
    video_true,
    video_pred,
    ordered_video_ids,
) = majority_vote_per_video(
    video_ids=val_video_ids,
    probabilities=frame_probabilities,
    true_labels=frame_true,
)


# =====================================================================
# 17. VIDEO-LEVEL EVALUATION
# =====================================================================

video_accuracy = accuracy_score(
    video_true,
    video_pred,
)

print(
    "\nVideo-Level Evaluation After Majority Voting"
)

print(
    "--------------------------------------------"
)

print(
    f"Videos evaluated : {len(ordered_video_ids)}"
)

print(
    f"Accuracy         : {video_accuracy * 100:.4f}%"
)

video_report = classification_report(
    video_true,
    video_pred,
    labels=LABEL_IDS,
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0,
)

print(
    video_report
)

(
    OUTPUT_DIR
    / "video_level_classification_report.txt"
).write_text(
    video_report,
    encoding="utf-8",
)

video_cm = confusion_matrix(
    video_true,
    video_pred,
    labels=LABEL_IDS,
)

fig, ax = plt.subplots(
    figsize=(8, 6)
)

ConfusionMatrixDisplay(
    confusion_matrix=video_cm,
    display_labels=CLASS_NAMES,
).plot(
    ax=ax,
    xticks_rotation=45,
    values_format="d",
)

plt.title(
    "Video-Level Confusion Matrix After Majority Voting\n"
    "RAVDESS Uniform Sampling"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "video_level_confusion_matrix.png"
)

plt.show()


# =====================================================================
# 18. SAVE FINAL VIDEO-LEVEL PREDICTIONS
# =====================================================================

prediction_csv = (
    OUTPUT_DIR
    / "video_level_predictions.csv"
)

with open(
    prediction_csv,
    "w",
    newline="",
    encoding="utf-8",
) as csv_file:
    writer = csv.writer(
        csv_file
    )

    writer.writerow(
        [
            "video_id",
            "true_label",
            "predicted_label",
        ]
    )

    print(
        "\nFinal Per-Video Predictions"
    )

    print(
        "---------------------------"
    )

    for (
        video_id,
        true_index,
        predicted_index,
    ) in zip(
        ordered_video_ids,
        video_true,
        video_pred,
    ):
        true_label = (
            CLASS_NAMES[
                true_index
            ]
        )

        predicted_label = (
            CLASS_NAMES[
                predicted_index
            ]
        )

        print(
            f"{video_id:55s} | "
            f"True: {true_label:8s} | "
            f"Predicted: {predicted_label}"
        )

        writer.writerow(
            [
                video_id,
                true_label,
                predicted_label,
            ]
        )
