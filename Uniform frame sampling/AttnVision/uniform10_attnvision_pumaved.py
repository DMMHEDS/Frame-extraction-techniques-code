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
import torch

from PIL import Image
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import (
    ViTForImageClassification,
    ViTImageProcessor,
)


# =====================================================================
# 2. CONFIGURATION
# =====================================================================

PUMAVED_VIDEO_DIR = Path(
    r"D:\IIT mandi\Facial datasets\PUMAVE-D"
)

ALIGNED_FRAME_DIR = Path(
    r"D:\IIT mandi\Facial datasets\PUMAVE-D_UNIFORM10\aligned_frames"
)

DLIB_LANDMARK_MODEL = Path(
    r"C:\PATH\TO\shape_predictor_68_face_landmarks.dat"
)

OUTPUT_DIR = Path(
    r"D:\IIT mandi\Facial datasets\PUMAVE-D_UNIFORM10\attnvision_outputs"
)

MODEL_NAME = "dima806/face_emotions_image_detection"

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

INDEX_TO_CLASS = {
    index: class_name
    for class_name, index in CLASS_TO_INDEX.items()
}

# PUMAVE-D class discovery is based on emotion-folder names.
# Neutral/unmatched categories are intentionally excluded so that the
# five-class protocol used in the paper is preserved.
PUMAVED_CLASS_LOOKUP = {
    "anger": "Anger",
    "angry": "Anger",
    "disgust": "Disgust",
    "disgusted": "Disgust",
    "fear": "Fear",
    "fearful": "Fear",
    "happy": "Happy",
    "happiness": "Happy",
    "sad": "Sad",
    "sadness": "Sad",
}

# Uniform Frame Sampling parameter.
FRAME_INTERVAL = 10

# Facial preprocessing.
FACE_SIZE = 48
FACE_CHIP_PADDING = 0.25
DLIB_UPSAMPLE = 1

# Dataset split.
VAL_SPLIT = 0.20
SEED = 42

# AttnVision/ViT training.
BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
EARLY_STOPPING_PATIENCE = 5
NUM_WORKERS = 0

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =====================================================================
# 3. REPRODUCIBILITY / DEVICE
# =====================================================================

os.environ.setdefault(
    "PYTHONHASHSEED",
    str(SEED),
)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
print("Using device:", device)


# =====================================================================
# 4. PUMAVE-D DATASET DISCOVERY
# =====================================================================

def infer_pumaved_class(relative_path: Path):
    """
    Infer one retained PUMAVE-D emotion class from parent folders.

    Typical layouts:
        Anger/video01.mp4
        Subject01/Anger/video01.mp4

    Neutral/unmatched videos are ignored because this study retains only
    Anger, Disgust, Fear, Happy, and Sad.
    """
    matches = []

    for part in relative_path.parts[:-1]:
        canonical = PUMAVED_CLASS_LOOKUP.get(
            part.lower()
        )

        if canonical is not None:
            matches.append(
                (part, canonical)
            )

    if not matches:
        return None, None

    canonical_classes = {
        canonical
        for _, canonical in matches
    }

    if len(canonical_classes) > 1:
        raise ValueError(
            f"Ambiguous emotion folders in '{relative_path}': "
            f"{sorted(canonical_classes)}"
        )

    return matches[0]


def make_video_id(
    relative_path: Path,
    matched_class_part: str,
):
    """
    Build a stable PUMAVE-D source-video ID while removing the
    emotion-folder component.

    Example:
        Subject01/Anger/video01.mp4
        -> Subject01__video01
    """
    parts = list(
        relative_path
        .with_suffix("")
        .parts
    )

    kept = []
    removed = False

    for part in parts:
        if (
            not removed
            and part == matched_class_part
        ):
            removed = True
            continue

        kept.append(part)

    return (
        "__".join(kept)
        or relative_path.stem
    )


def discover_pumaved_videos(
    video_root: Path,
):
    """
    Find PUMAVE-D videos belonging to the five retained emotion classes.
    """
    video_root = Path(
        video_root
    )

    if not video_root.exists():
        raise FileNotFoundError(
            "PUMAVE-D video directory does not exist:\n"
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
            f"No supported videos found under:\n{video_root}"
        )

    selected = []
    ignored = 0

    for video_path in videos:
        relative_path = video_path.relative_to(
            video_root
        )

        (
            matched_class_part,
            class_name,
        ) = infer_pumaved_class(
            relative_path
        )

        if class_name is None:
            ignored += 1
            continue

        video_id = make_video_id(
            relative_path,
            matched_class_part,
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
            "No PUMAVE-D videos matched the selected five-class protocol. "
            "This implementation expects emotion names in parent folders."
        )

    print(
        f"Total video files found : {len(videos)}"
    )

    print(
        f"Five-class video files  : {len(selected)}"
    )

    print(
        f"Ignored/unmatched       : {ignored}"
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
    Load DLib's 68-point facial landmark predictor once.
    """
    global _landmark_predictor

    if _landmark_predictor is None:
        if not DLIB_LANDMARK_MODEL.exists():
            raise FileNotFoundError(
                "DLib landmark predictor not found:\n"
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
    Detect the largest face and localize 68 facial landmarks.

    Returns
    -------
    face_rectangle, dlib_shape, landmarks

    landmarks:
        np.ndarray with shape (68, 2)
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
        -> resize to 48x48x3
    """
    frame_rgb = cv2.cvtColor(
        frame_bgr,
        cv2.COLOR_BGR2RGB,
    )

    aligned_rgb = dlib.get_face_chip(
        frame_rgb,
        shape,
        size=FACE_SIZE,
        padding=FACE_CHIP_PADDING,
    )

    aligned_bgr = cv2.cvtColor(
        aligned_rgb,
        cv2.COLOR_RGB2BGR,
    )

    if aligned_bgr.shape[:2] != (
        FACE_SIZE,
        FACE_SIZE,
    ):
        aligned_bgr = cv2.resize(
            aligned_bgr,
            (
                FACE_SIZE,
                FACE_SIZE,
            ),
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
    Select every Nth decoded frame and then apply facial preprocessing.

    Selection rule:
        retain F_t when t mod N = 0

    For this paper:
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
            f"{len(existing_frames)} prepared frame(s) already exist."
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
    sampled_count = 0
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
                (
                    ret,
                    frame_bgr,
                ) = cap.read()

                if not ret:
                    break

                decoded_count += 1

                # -----------------------------------------------------
                # FRAME SELECTION:
                # every 10th decoded frame.
                # -----------------------------------------------------
                if (
                    decoded_count
                    % frame_interval
                    != 0
                ):
                    continue

                sampled_count += 1

                # -----------------------------------------------------
                # FACIAL PRE-PROCESSING:
                # landmarks are extracted only AFTER uniform sampling.
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

                success = cv2.imwrite(
                    str(
                        output_path
                    ),
                    aligned_face,
                )

                if not success:
                    raise RuntimeError(
                        f"Could not save aligned frame: {output_path}"
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
            f"uniformly_sampled_frames={sampled_count}\n"
            f"aligned_faces={aligned_count}\n"
            f"sampled_frames_without_face={no_face_count}\n"
        ),
        encoding="utf-8",
    )

    print(
        f"{video_path.name}: "
        f"decoded={decoded_count}, "
        f"sampled={sampled_count}, "
        f"aligned={aligned_count}, "
        f"no_face={no_face_count}"
    )

    return aligned_count


def prepare_pumaved_uniform_dataset(
    video_root: Path,
):
    """
    PUMAVE-D:
        every 10th frame
        -> DLib face detection
        -> 68 landmarks
        -> alignment/crop
        -> 48x48x3
    """
    videos = discover_pumaved_videos(
        video_root
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
        "\nUniform-frame preparation completed."
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
    prepare_pumaved_uniform_dataset(
        PUMAVED_VIDEO_DIR
    )


# =====================================================================
# 7. BUILD PREPROCESSED FRAME INDEX
# =====================================================================

def build_frame_index(
    aligned_root: Path,
):
    """
    Expected generated structure:

        aligned_frames/
            Anger/
                video_id/
                    frame_*.png
            Disgust/
            Fear/
            Happy/
            Sad/
    """
    aligned_root = Path(
        aligned_root
    )

    if not aligned_root.exists():
        raise FileNotFoundError(
            "Aligned frame directory does not exist:\n"
            f"{aligned_root}\n"
            "Set RUN_FRAME_PREPARATION=True on the first run."
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

        class_index = CLASS_TO_INDEX[
            class_name
        ]

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
    dtype=np.int64,
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
    Split SOURCE VIDEOS rather than individual frames.

    This prevents frames from the same source video from appearing in both
    training and validation sets.
    """
    video_to_label = {}

    for (
        video_id,
        label,
    ) in zip(
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
        dtype=np.int64,
    )

    counts = Counter(
        unique_labels.tolist()
    )

    missing_classes = [
        CLASS_NAMES[index]
        for index
        in range(NUM_CLASSES)
        if counts.get(
            index,
            0,
        ) == 0
    ]

    if missing_classes:
        raise ValueError(
            "Missing usable class(es): "
            + ", ".join(
                missing_classes
            )
        )

    (
        train_videos,
        val_videos,
    ) = train_test_split(
        unique_videos,
        test_size=VAL_SPLIT,
        random_state=SEED,
        shuffle=True,
        stratify=unique_labels,
    )

    overlap = np.intersect1d(
        train_videos,
        val_videos,
    )

    if overlap.size:
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
# 9. ViT IMAGE PROCESSOR
# =====================================================================

processor = (
    ViTImageProcessor
    .from_pretrained(
        MODEL_NAME
    )
)


# =====================================================================
# 10. PYTORCH DATASET / DATALOADERS
# =====================================================================

class FacialFrameDataset(
    Dataset
):
    def __init__(
        self,
        paths,
        labels,
        image_processor,
    ):
        self.paths = list(
            paths
        )

        self.labels = np.asarray(
            labels,
            dtype=np.int64,
        )

        self.image_processor = (
            image_processor
        )

    def __len__(
        self
    ):
        return len(
            self.paths
        )

    def __getitem__(
        self,
        index,
    ):
        image = Image.open(
            self.paths[
                index
            ]
        ).convert(
            "RGB"
        )

        encoded = (
            self.image_processor(
                images=image,
                return_tensors="pt",
            )
        )

        pixel_values = (
            encoded[
                "pixel_values"
            ]
            .squeeze(0)
        )

        label = torch.tensor(
            int(
                self.labels[
                    index
                ]
            ),
            dtype=torch.long,
        )

        return {
            "pixel_values": pixel_values,
            "labels": label,
        }


train_dataset = FacialFrameDataset(
    train_paths,
    train_labels,
    processor,
)

val_dataset = FacialFrameDataset(
    val_paths,
    val_labels,
    processor,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)


# =====================================================================
# 11. ATTNVISION / ViT MODEL
# =====================================================================

id2label = {
    index: class_name
    for index, class_name
    in enumerate(
        CLASS_NAMES
    )
}

label2id = {
    class_name: index
    for index, class_name
    in id2label.items()
}

model = (
    ViTForImageClassification
    .from_pretrained(
        MODEL_NAME,
        num_labels=NUM_CLASSES,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
)

model.to(
    device
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

scheduler = (
    torch.optim.lr_scheduler
    .CosineAnnealingLR(
        optimizer,
        T_max=max(
            1,
            EPOCHS,
        ),
    )
)

BEST_MODEL_PATH = (
    OUTPUT_DIR
    / "best_pumaved_uniform10_attnvision.pt"
)


# =====================================================================
# 12. TRAINING / VALIDATION FUNCTIONS
# =====================================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        pixel_values = (
            batch[
                "pixel_values"
            ]
            .to(
                device,
                non_blocking=True,
            )
        )

        labels = (
            batch[
                "labels"
            ]
            .to(
                device,
                non_blocking=True,
            )
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        outputs = model(
            pixel_values=pixel_values,
            labels=labels,
        )

        loss = outputs.loss

        loss.backward()

        optimizer.step()

        batch_size = labels.size(
            0
        )

        running_loss += (
            loss.item()
            * batch_size
        )

        predictions = (
            outputs.logits
            .argmax(
                dim=1
            )
        )

        correct += (
            predictions
            .eq(
                labels
            )
            .sum()
            .item()
        )

        total += batch_size

    epoch_loss = (
        running_loss
        / max(
            1,
            total,
        )
    )

    epoch_accuracy = (
        correct
        / max(
            1,
            total,
        )
    )

    return (
        epoch_loss,
        epoch_accuracy,
    )


@torch.inference_mode()
def evaluate_epoch(
    model,
    loader,
    device,
):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        pixel_values = (
            batch[
                "pixel_values"
            ]
            .to(
                device,
                non_blocking=True,
            )
        )

        labels = (
            batch[
                "labels"
            ]
            .to(
                device,
                non_blocking=True,
            )
        )

        outputs = model(
            pixel_values=pixel_values,
            labels=labels,
        )

        loss = outputs.loss

        batch_size = labels.size(
            0
        )

        running_loss += (
            loss.item()
            * batch_size
        )

        predictions = (
            outputs.logits
            .argmax(
                dim=1
            )
        )

        correct += (
            predictions
            .eq(
                labels
            )
            .sum()
            .item()
        )

        total += batch_size

    epoch_loss = (
        running_loss
        / max(
            1,
            total,
        )
    )

    epoch_accuracy = (
        correct
        / max(
            1,
            total,
        )
    )

    return (
        epoch_loss,
        epoch_accuracy,
    )


# =====================================================================
# 13. TRAIN ATTNVISION / ViT
# =====================================================================

history = {
    "train_loss": [],
    "train_accuracy": [],
    "val_loss": [],
    "val_accuracy": [],
}

best_val_loss = float(
    "inf"
)

epochs_without_improvement = 0

for epoch in range(
    1,
    EPOCHS + 1,
):
    (
        train_loss,
        train_accuracy,
    ) = train_one_epoch(
        model,
        train_loader,
        optimizer,
        device,
    )

    (
        val_loss,
        val_accuracy,
    ) = evaluate_epoch(
        model,
        val_loader,
        device,
    )

    scheduler.step()

    history[
        "train_loss"
    ].append(
        train_loss
    )

    history[
        "train_accuracy"
    ].append(
        train_accuracy
    )

    history[
        "val_loss"
    ].append(
        val_loss
    )

    history[
        "val_accuracy"
    ].append(
        val_accuracy
    )

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_accuracy:.4f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0

        torch.save(
            model.state_dict(),
            BEST_MODEL_PATH,
        )

        print(
            f"Saved best model: {BEST_MODEL_PATH}"
        )

    else:
        epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):
            print(
                "Early stopping: "
                f"no validation-loss improvement for "
                f"{EARLY_STOPPING_PATIENCE} epoch(s)."
            )
            break


model.load_state_dict(
    torch.load(
        BEST_MODEL_PATH,
        map_location=device,
    )
)

model.to(
    device
)

model.eval()


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

    for index in range(
        len(
            history[
                "train_loss"
            ]
        )
    ):
        writer.writerow(
            [
                index + 1,
                history[
                    "train_loss"
                ][
                    index
                ],
                history[
                    "train_accuracy"
                ][
                    index
                ],
                history[
                    "val_loss"
                ][
                    index
                ],
                history[
                    "val_accuracy"
                ][
                    index
                ],
            ]
        )


plt.figure(
    figsize=(8, 6)
)

plt.plot(
    history[
        "train_accuracy"
    ],
    label="Train Accuracy",
)

plt.plot(
    history[
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
    "AttnVision/ViT Accuracy - PUMAVE-D Uniform Sampling"
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
    history[
        "train_loss"
    ],
    label="Train Loss",
)

plt.plot(
    history[
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
    "AttnVision/ViT Loss - PUMAVE-D Uniform Sampling"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "training_loss.png"
)

plt.show()


# =====================================================================
# 15. FRAME-WISE PREDICTION
# =====================================================================

@torch.inference_mode()
def predict_probabilities(
    model,
    loader,
    device,
):
    model.eval()

    probabilities = []
    true_labels = []

    for batch in loader:
        pixel_values = (
            batch[
                "pixel_values"
            ]
            .to(
                device,
                non_blocking=True,
            )
        )

        labels = (
            batch[
                "labels"
            ]
            .cpu()
            .numpy()
        )

        outputs = model(
            pixel_values=pixel_values
        )

        batch_probabilities = (
            torch.softmax(
                outputs.logits,
                dim=1,
            )
            .cpu()
            .numpy()
        )

        probabilities.append(
            batch_probabilities
        )

        true_labels.append(
            labels
        )

    return (
        np.concatenate(
            probabilities,
            axis=0,
        ),
        np.concatenate(
            true_labels,
            axis=0,
        ),
    )


(
    frame_probabilities,
    frame_true,
) = predict_probabilities(
    model,
    val_loader,
    device,
)

frame_predictions = np.argmax(
    frame_probabilities,
    axis=1,
).astype(
    np.int64
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


# =====================================================================
# 16. FRAME-LEVEL EVALUATION
# =====================================================================

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
    "PUMAVE-D Uniform Sampling Every 10th Frame"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "frame_level_confusion_matrix.png"
)

plt.show()


# =====================================================================
# 17. VIDEO-LEVEL MAJORITY VOTING
# =====================================================================

def majority_vote_per_video(
    video_ids,
    probabilities,
    true_labels,
):
    """
    Convert frame-wise predictions into one final label for each video.

    Main rule:
        class with the largest number of frame votes wins.

    Tie:
        among tied classes, select the class with the highest mean
        softmax probability across frames of that video.
    """
    video_ids = np.asarray(
        video_ids
    )

    probabilities = np.asarray(
        probabilities
    )

    true_labels = np.asarray(
        true_labels,
        dtype=np.int64,
    )

    frame_predictions = np.argmax(
        probabilities,
        axis=1,
    ).astype(
        np.int64
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

        truths_for_video = np.unique(
            true_labels[
                mask
            ]
        )

        if len(
            truths_for_video
        ) != 1:
            raise ValueError(
                f"Inconsistent true labels for video "
                f"'{video_id}': {truths_for_video.tolist()}"
            )

        vote_counts = np.bincount(
            predictions_for_video,
            minlength=NUM_CLASSES,
        )

        tied_classes = np.flatnonzero(
            vote_counts
            == vote_counts.max()
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
                probabilities_for_video
                .mean(
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
                truths_for_video[
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
            dtype=np.int64,
        ),
        np.asarray(
            video_pred,
            dtype=np.int64,
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
# 18. FINAL VIDEO-LEVEL EVALUATION
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
    "PUMAVE-D Uniform Sampling"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "video_level_confusion_matrix.png"
)

plt.show()


# =====================================================================
# 19. SAVE FINAL VIDEO-LEVEL PREDICTIONS
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
        true_label = CLASS_NAMES[
            true_index
        ]

        predicted_label = CLASS_NAMES[
            predicted_index
        ]

        print(
            f"{video_id:60s} | "
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
