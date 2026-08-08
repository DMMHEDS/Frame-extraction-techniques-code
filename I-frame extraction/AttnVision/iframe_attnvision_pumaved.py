# =====================================================================
# 1. IMPORTS
# =====================================================================

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
import torch

from datasets import Dataset
from PIL import Image

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    ViTForImageClassification,
    ViTImageProcessor,
)


# =====================================================================
# 2. CONFIGURATION
# =====================================================================

PUMAVED_VIDEO_DIR = Path(r"C:\PATH\TO\PUMAVE-D\VIDEOS")
EXTRACTED_IFRAME_DIR = Path(r"C:\PATH\TO\PUMAVE-D_IFRAME\extracted_frames")
ALIGNED_FRAME_DIR = Path(r"C:\PATH\TO\PUMAVE-D_IFRAME\aligned_frames")
DLIB_LANDMARK_MODEL = Path(r"C:\PATH\TO\shape_predictor_68_face_landmarks.dat")
OUTPUT_DIR = Path(r"C:\PATH\TO\PUMAVE-D_IFRAME\vit_checkpoints")

RUN_IFRAME_EXTRACTION = False
RUN_FACE_PREPROCESSING = False

OVERWRITE_EXISTING_IFRAMES = False
OVERWRITE_EXISTING_ALIGNED = False

VIDEO_EXTENSIONS = {
    ".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg", ".m4v"
}

CLASS_NAMES = ["Anger", "Disgust", "Fear", "Happy", "Sad"]
NUM_CLASSES = len(CLASS_NAMES)

CLASS_TO_INDEX = {
    class_name: index
    for index, class_name in enumerate(CLASS_NAMES)
}

# PUMAVE-D emotion folders may use noun/adjective variants such as
# Anger/Angry, Fear/Fearful, Happy/Happiness and Sad/Sadness.
# Neutral is intentionally omitted because the paper retains only
# Anger, Disgust, Fear, Happy and Sad.
CLASS_LOOKUP = {
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

IMAGE_SIZE = 224

BATCH_SIZE = 4
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
VAL_SPLIT = 0.20
EARLY_STOPPING_PATIENCE = 5
SEED = 42

MODEL_NAME = "google/vit-base-patch16-224-in21k"

FACE_CHIP_PADDING = 0.25


# =====================================================================
# 3. REPRODUCIBILITY AND DEVICE INFORMATION
# =====================================================================

os.environ.setdefault("PYTHONHASHSEED", str(SEED))

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# =====================================================================
# 4. DATASET CLASS AND VIDEO IDENTIFICATION
# =====================================================================

def infer_class_from_relative_path(relative_path: Path):
    """
    Infer one of the five retained PUMAVE-D emotion classes from the
    video's parent folders.

    The function supports layouts such as:
        PUMAVE-D/Anger/video01.mp4
    and:
        PUMAVE-D/Subject01/Anger/video01.mp4

    Neutral clips are ignored because the paper evaluates only
    Anger, Disgust, Fear, Happy and Sad.
    """
    matches = []
    for part in relative_path.parts[:-1]:
        canonical = CLASS_LOOKUP.get(part.lower())
        if canonical is not None:
            matches.append((part, canonical))

    if not matches:
        return None, None

    canonical_classes = {canonical for _, canonical in matches}

    if len(canonical_classes) > 1:
        raise ValueError(
            f"Ambiguous emotion folders in path '{relative_path}': "
            f"{sorted(canonical_classes)}"
        )

    return matches[0]


def make_video_id(relative_path: Path, matched_class_part: str):
    """
    Build a unique PUMAVE-D video ID while removing the emotion folder.

    Example:
        Subject01/Anger/video01.mp4
        -> Subject01__video01
    """
    path_parts = list(relative_path.with_suffix("").parts)
    kept_parts = []
    class_removed = False

    for part in path_parts:
        if not class_removed and part == matched_class_part:
            class_removed = True
            continue
        kept_parts.append(part)

    return "__".join(kept_parts) or relative_path.stem


# =====================================================================
# 5. I-FRAME EXTRACTION WITH FFMPEG
# =====================================================================

def check_ffmpeg_available():
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "FFmpeg was not found on PATH. Install FFmpeg 5.1 "
            "(the version reported in the paper) or add the FFmpeg "
            "executable folder to PATH."
        )


def extract_iframes_from_video(video_path: Path, output_dir: Path, overwrite: bool = False):
    check_ffmpeg_available()
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    existing_frames = sorted(output_dir.glob("iframe_*.png"))

    if existing_frames and not overwrite:
        print(f"Skipping {video_path.name}: {len(existing_frames)} I-frame(s) already exist.")
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
        raise RuntimeError(f"FFmpeg I-frame extraction failed for: {video_path}") from exc

    extracted_frames = sorted(output_dir.glob("iframe_*.png"))

    if not extracted_frames:
        print(f"WARNING: no I-frames were written for {video_path}")
    else:
        print(f"{video_path.name}: extracted {len(extracted_frames)} I-frame(s).")

    return extracted_frames


def extract_iframes_from_dataset(video_root: Path, output_root: Path, overwrite: bool = False):
    video_root = Path(video_root)
    output_root = Path(output_root)

    if not video_root.exists():
        raise FileNotFoundError(
            f"PUMAVE-D video directory was not found:\n{video_root}\nUpdate PUMAVED_VIDEO_DIR."
        )

    all_video_files = sorted(
        path for path in video_root.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )

    if not all_video_files:
        raise FileNotFoundError(f"No supported videos were found under:\n{video_root}")

    selected_videos = []
    ignored_videos = []

    for video_path in all_video_files:
        relative_path = video_path.relative_to(video_root)
        matched_part, class_name = infer_class_from_relative_path(relative_path)

        if class_name is None:
            ignored_videos.append(relative_path)
            continue

        video_id = make_video_id(relative_path, matched_part)
        selected_videos.append((video_path, relative_path, class_name, video_id))

    if not selected_videos:
        raise RuntimeError(
            f"Videos were found, but none matched the retained classes: {CLASS_NAMES}. "
            "Check the PUMAVE-D folder layout."
        )

    print(f"Total video files found      : {len(all_video_files)}")
    print(f"Videos in retained 5 classes : {len(selected_videos)}")
    print(f"Skipped/unmatched videos      : {len(ignored_videos)}")

    total_iframes = 0

    for index, (video_path, relative_path, class_name, video_id) in enumerate(selected_videos, start=1):
        destination = output_root / class_name / video_id
        print(f"[{index}/{len(selected_videos)}] {relative_path} -> {class_name}/{video_id}")
        frames = extract_iframes_from_video(video_path=video_path, output_dir=destination, overwrite=overwrite)
        total_iframes += len(frames)

    print("\nI-frame extraction completed.")
    print(f"Videos processed       : {len(selected_videos)}")
    print(f"Total I-frames retained: {total_iframes}")
    print(f"Output directory       : {output_root}")


# =====================================================================
# 6. FACIAL PRE-PROCESSING WITH DLIB
# =====================================================================

_face_detector = dlib.get_frontal_face_detector()
_landmark_predictor = None


def get_landmark_predictor():
    global _landmark_predictor
    if _landmark_predictor is None:
        if not DLIB_LANDMARK_MODEL.exists():
            raise FileNotFoundError(
                f"DLib landmark model was not found:\n{DLIB_LANDMARK_MODEL}\n"
                "Download shape_predictor_68_face_landmarks.dat and update DLIB_LANDMARK_MODEL."
            )
        _landmark_predictor = dlib.shape_predictor(str(DLIB_LANDMARK_MODEL))
    return _landmark_predictor


def detect_face_and_shape(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = _face_detector(gray, 1)

    if len(faces) == 0:
        return None, None, None

    face_rectangle = max(
        faces, key=lambda rectangle: max(0, rectangle.width()) * max(0, rectangle.height())
    )

    predictor = get_landmark_predictor()
    shape = predictor(gray, face_rectangle)

    landmarks = np.array(
        [[shape.part(index).x, shape.part(index).y] for index in range(68)], dtype=np.float32
    )

    return face_rectangle, shape, landmarks


def align_crop_resize_face(image_bgr, shape, output_size=IMAGE_SIZE, padding=FACE_CHIP_PADDING):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    aligned_rgb = dlib.get_face_chip(image_rgb, shape, size=int(output_size), padding=float(padding))
    aligned_bgr = cv2.cvtColor(aligned_rgb, cv2.COLOR_RGB2BGR)

    if aligned_bgr.shape[:2] != (output_size, output_size):
        aligned_bgr = cv2.resize(aligned_bgr, (output_size, output_size), interpolation=cv2.INTER_LINEAR)

    return aligned_bgr


def preprocess_face(image_path: Path, output_size=IMAGE_SIZE):
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        return None

    _, shape, _ = detect_face_and_shape(image_bgr)
    if shape is None:
        return None

    return align_crop_resize_face(image_bgr=image_bgr, shape=shape, output_size=output_size)


def preprocess_dataset(iframe_root: Path, aligned_root: Path, output_size=IMAGE_SIZE, overwrite: bool = False):
    iframe_root = Path(iframe_root)
    aligned_root = Path(aligned_root)

    if not iframe_root.exists():
        raise FileNotFoundError(f"I-frame directory does not exist:\n{iframe_root}")

    frame_paths = sorted(iframe_root.rglob("iframe_*.png"))

    if not frame_paths:
        raise FileNotFoundError(f"No I-frames were found under:\n{iframe_root}")

    kept = 0
    skipped_no_face = 0
    failed_write = 0

    for index, frame_path in enumerate(frame_paths, start=1):
        relative_path = frame_path.relative_to(iframe_root)
        destination_path = aligned_root / relative_path
        destination_path.parent.mkdir(parents=True, exist_ok=True)

        if destination_path.exists() and not overwrite:
            kept += 1
            continue

        aligned_face = preprocess_face(frame_path, output_size=output_size)

        if aligned_face is None:
            skipped_no_face += 1
            continue

        success = cv2.imwrite(str(destination_path), aligned_face)

        if not success:
            failed_write += 1
            continue

        kept += 1

        if index % 200 == 0:
            print(
                f"Processed {index}/{len(frame_paths)} "
                f"(kept={kept}, no_face={skipped_no_face}, write_fail={failed_write})"
            )

    print("\nFacial preprocessing completed.")
    print(f"Input I-frames            : {len(frame_paths)}")
    print(f"Aligned faces retained    : {kept}")
    print(f"Skipped (no face detected): {skipped_no_face}")
    print(f"Failed writes             : {failed_write}")
    print(f"Output directory          : {aligned_root}")


# =====================================================================
# 7. RUN EXTRACTION / PREPROCESSING WHEN REQUESTED
# =====================================================================

if RUN_IFRAME_EXTRACTION:
    extract_iframes_from_dataset(
        video_root=PUMAVED_VIDEO_DIR, output_root=EXTRACTED_IFRAME_DIR, overwrite=OVERWRITE_EXISTING_IFRAMES
    )

if RUN_FACE_PREPROCESSING:
    preprocess_dataset(
        iframe_root=EXTRACTED_IFRAME_DIR, aligned_root=ALIGNED_FRAME_DIR,
        output_size=IMAGE_SIZE, overwrite=OVERWRITE_EXISTING_ALIGNED
    )


# =====================================================================
# 8. BUILD FRAME INDEX WITH ORIGINAL VIDEO IDs
# =====================================================================

def build_frame_index(aligned_root: Path):
    aligned_root = Path(aligned_root)

    if not aligned_root.exists():
        raise FileNotFoundError(
            f"Aligned facial-frame directory does not exist:\n{aligned_root}\n"
            "Run I-frame extraction/facial preprocessing first or update ALIGNED_FRAME_DIR."
        )

    records = []

    for class_name in CLASS_NAMES:
        class_directory = aligned_root / class_name

        if not class_directory.exists():
            print(f"WARNING: missing class folder: {class_directory}")
            continue

        class_index = CLASS_TO_INDEX[class_name]
        video_directories = sorted(path for path in class_directory.iterdir() if path.is_dir())

        for video_directory in video_directories:
            frame_files = sorted(video_directory.glob("iframe_*.png"))

            if not frame_files:
                continue

            video_id = f"{class_name}/{video_directory.name}"

            for frame_path in frame_files:
                records.append({"image_path": str(frame_path), "label": class_index, "video_id": video_id})

    if not records:
        raise FileNotFoundError(f"No aligned frames were found under:\n{aligned_root}")

    return records


records = build_frame_index(ALIGNED_FRAME_DIR)
print(f"\nTotal aligned I-frame records: {len(records)}")


# =====================================================================
# 9. VIDEO-LEVEL STRATIFIED 80:20 TRAIN/VALIDATION SPLIT
# =====================================================================

def split_records_by_video(records, validation_fraction=VAL_SPLIT, seed=SEED):
    video_to_label = {}

    for record in records:
        video_id = record["video_id"]
        label = int(record["label"])

        if video_id in video_to_label:
            if video_to_label[video_id] != label:
                raise ValueError(f"Video '{video_id}' has conflicting labels.")
        else:
            video_to_label[video_id] = label

    unique_video_ids = np.array(list(video_to_label.keys()))
    unique_video_labels = np.array([video_to_label[v] for v in unique_video_ids], dtype=np.int64)

    class_counts = Counter(unique_video_labels.tolist())

    missing_classes = [
        CLASS_NAMES[index] for index in range(NUM_CLASSES) if class_counts.get(index, 0) == 0
    ]

    if missing_classes:
        raise ValueError("No usable videos were found for class(es): " + ", ".join(missing_classes))

    too_small = {CLASS_NAMES[index]: count for index, count in class_counts.items() if count < 2}

    if too_small:
        raise ValueError(
            f"At least two usable videos per class are required for a stratified split. "
            f"Too-small classes: {too_small}"
        )

    train_video_ids, val_video_ids = train_test_split(
        unique_video_ids, test_size=validation_fraction, random_state=seed, shuffle=True,
        stratify=unique_video_labels,
    )

    train_video_set = set(train_video_ids)
    val_video_set = set(val_video_ids)

    if train_video_set & val_video_set:
        raise RuntimeError("Video leakage detected between training and validation.")

    train_records = [r for r in records if r["video_id"] in train_video_set]
    val_records = [r for r in records if r["video_id"] in val_video_set]

    return train_records, val_records, train_video_ids, val_video_ids, video_to_label


(train_records, val_records, train_video_ids, val_video_ids_unique, video_to_label) = split_records_by_video(records)

print("\nDataset split")
print("-------------")
print(f"Total videos      : {len(video_to_label)}")
print(f"Training videos   : {len(train_video_ids)}")
print(f"Validation videos : {len(val_video_ids_unique)}")
print(f"Training frames   : {len(train_records)}")
print(f"Validation frames : {len(val_records)}")


def print_video_distribution(split_name, video_ids):
    counts = Counter(video_to_label[v] for v in video_ids)
    distribution_text = ", ".join(f"{CLASS_NAMES[i]}={counts.get(i, 0)}" for i in range(NUM_CLASSES))
    print(f"{split_name}: {distribution_text}")


print_video_distribution("Train videos", train_video_ids)
print_video_distribution("Validation videos", val_video_ids_unique)


# =====================================================================
# 10. CREATE HUGGINGFACE DATASETS
# =====================================================================

train_ds = Dataset.from_dict({
    "image_path": [r["image_path"] for r in train_records],
    "label": [r["label"] for r in train_records],
    "video_id": [r["video_id"] for r in train_records],
})

val_ds = Dataset.from_dict({
    "image_path": [r["image_path"] for r in val_records],
    "label": [r["label"] for r in val_records],
    "video_id": [r["video_id"] for r in val_records],
})

val_video_ids = np.array(val_ds["video_id"])
val_labels_ordered = np.array(val_ds["label"], dtype=np.int64)

id2label = {index: class_name for index, class_name in enumerate(CLASS_NAMES)}
label2id = {class_name: index for index, class_name in id2label.items()}

print(f"\nTrain Dataset rows: {len(train_ds)}")
print(f"Validation Dataset rows: {len(val_ds)}")


# =====================================================================
# 11. IMAGE PROCESSOR AND TRAIN/VALIDATION TRANSFORMS
# =====================================================================

processor = ViTImageProcessor.from_pretrained(MODEL_NAME)

image_mean = processor.image_mean
image_std = processor.image_std

normalize = Normalize(mean=image_mean, std=image_std)

train_transforms = Compose([RandomResizedCrop(IMAGE_SIZE), RandomHorizontalFlip(), ToTensor(), normalize])
val_transforms = Compose([Resize(IMAGE_SIZE), CenterCrop(IMAGE_SIZE), ToTensor(), normalize])


def train_image_transforms(examples):
    images = [Image.open(p).convert("RGB") for p in examples["image_path"]]
    examples["pixel_values"] = [train_transforms(img) for img in images]
    return examples


def val_image_transforms(examples):
    images = [Image.open(p).convert("RGB") for p in examples["image_path"]]
    examples["pixel_values"] = [val_transforms(img) for img in images]
    return examples


train_ds.set_transform(train_image_transforms)
val_ds.set_transform(val_image_transforms)


# =====================================================================
# 12. COLLATE FUNCTION
# =====================================================================

def collate_fn(examples):
    pixel_values = torch.stack([e["pixel_values"] for e in examples])
    labels = torch.tensor([e["label"] for e in examples], dtype=torch.long)
    return {"pixel_values": pixel_values, "labels": labels}


train_dataloader = DataLoader(train_ds, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=True)
sample_batch = next(iter(train_dataloader))

print("\nSample batch shapes")
print("-------------------")
for key, value in sample_batch.items():
    print(key, value.shape)


# =====================================================================
# 13. LOAD VIT MODEL
# =====================================================================

model = ViTForImageClassification.from_pretrained(
    MODEL_NAME, num_labels=NUM_CLASSES, id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True
)

model.to(device)


# =====================================================================
# 14. METRICS
# =====================================================================

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    return {"accuracy": accuracy_score(labels, predictions)}


# =====================================================================
# 15. TRAINING ARGUMENTS
# =====================================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    save_strategy="epoch",
    eval_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    weight_decay=WEIGHT_DECAY,
    lr_scheduler_type="cosine",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    logging_dir=str(OUTPUT_DIR / "logs"),
    remove_unused_columns=False,
    save_total_limit=2,
    seed=SEED,
    data_seed=SEED,
    report_to="none",
)


# =====================================================================
# 16. TRAINER + EARLY STOPPING
# =====================================================================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    tokenizer=processor,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
)

print("\nStarting ViT training...")
trainer.train()


# =====================================================================
# 17. FRAME-LEVEL VALIDATION
# =====================================================================

outputs = trainer.predict(val_ds)

y_true_frame = np.asarray(outputs.label_ids, dtype=np.int64)
frame_logits = np.asarray(outputs.predictions)
y_pred_frame = np.argmax(frame_logits, axis=1).astype(np.int64)

if not (len(y_true_frame) == len(y_pred_frame) == len(val_video_ids) == len(val_labels_ordered)):
    raise RuntimeError("Validation predictions, labels, and video IDs are misaligned.")

if not np.array_equal(y_true_frame, val_labels_ordered):
    raise RuntimeError("Trainer prediction labels do not match validation dataset order.")

LABEL_IDS = np.arange(NUM_CLASSES)

frame_accuracy = accuracy_score(y_true_frame, y_pred_frame)

print("\nFrame-Level Evaluation")
print("----------------------")
print(f"Frame-level accuracy: {frame_accuracy * 100:.4f}%")

print(classification_report(y_true_frame, y_pred_frame, labels=LABEL_IDS, target_names=CLASS_NAMES, digits=4, zero_division=0))

frame_cm = confusion_matrix(y_true_frame, y_pred_frame, labels=LABEL_IDS)
frame_display = ConfusionMatrixDisplay(confusion_matrix=frame_cm, display_labels=CLASS_NAMES)

fig, ax = plt.subplots(figsize=(8, 6))
frame_display.plot(ax=ax, xticks_rotation=45, values_format="d")
plt.title("Frame-Level Confusion Matrix\nI-Frame Extraction + ViT")
plt.tight_layout()
plt.show()


# =====================================================================
# 18. MAJORITY VOTING FOR FINAL VIDEO-LEVEL PREDICTIONS
# =====================================================================

def softmax_numpy(logits):
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=1, keepdims=True)


def majority_vote_per_video(video_ids, frame_logits, frame_true_labels):
    video_ids = np.asarray(video_ids)
    frame_logits = np.asarray(frame_logits)
    frame_true_labels = np.asarray(frame_true_labels, dtype=np.int64)

    frame_probabilities = softmax_numpy(frame_logits)
    frame_predictions = np.argmax(frame_probabilities, axis=1).astype(np.int64)

    unique_videos = np.unique(video_ids)

    video_true = []
    video_pred = []

    for video_id in unique_videos:
        mask = video_ids == video_id
        predictions_for_video = frame_predictions[mask]
        probabilities_for_video = frame_probabilities[mask]
        true_labels_for_video = frame_true_labels[mask]

        true_classes = np.unique(true_labels_for_video)

        if len(true_classes) != 1:
            raise ValueError(f"Video '{video_id}' has inconsistent true labels: {true_classes.tolist()}")

        votes = np.bincount(predictions_for_video, minlength=NUM_CLASSES)
        maximum_votes = np.max(votes)
        tied_classes = np.flatnonzero(votes == maximum_votes)

        if len(tied_classes) == 1:
            final_prediction = int(tied_classes[0])
        else:
            mean_probabilities = np.mean(probabilities_for_video, axis=0)
            tied_probabilities = mean_probabilities[tied_classes]
            final_prediction = int(tied_classes[np.argmax(tied_probabilities)])

        video_true.append(int(true_classes[0]))
        video_pred.append(final_prediction)

    return np.array(video_true, dtype=np.int64), np.array(video_pred, dtype=np.int64), unique_videos


(y_true_video, y_pred_video, ordered_video_ids) = majority_vote_per_video(
    video_ids=val_video_ids, frame_logits=frame_logits, frame_true_labels=y_true_frame
)


# =====================================================================
# 19. VIDEO-LEVEL EVALUATION
# =====================================================================

video_accuracy = accuracy_score(y_true_video, y_pred_video)

print("\nVideo-Level Evaluation After Majority Voting")
print("--------------------------------------------")
print(f"Total videos evaluated: {len(ordered_video_ids)}")
print(f"Video-level accuracy: {video_accuracy * 100:.4f}%")

print(classification_report(y_true_video, y_pred_video, labels=LABEL_IDS, target_names=CLASS_NAMES, digits=4, zero_division=0))

video_cm = confusion_matrix(y_true_video, y_pred_video, labels=LABEL_IDS)
video_display = ConfusionMatrixDisplay(confusion_matrix=video_cm, display_labels=CLASS_NAMES)

fig, ax = plt.subplots(figsize=(8, 6))
video_display.plot(ax=ax, xticks_rotation=45, values_format="d")
plt.title("Video-Level Confusion Matrix After Majority Voting\nI-Frame Extraction + ViT")
plt.tight_layout()
plt.show()


# =====================================================================
# 20. PRINT FINAL PER-VIDEO PREDICTIONS
# =====================================================================

print("\nFinal Per-Video Predictions")
print("---------------------------")

for video_id, true_index, predicted_index in zip(ordered_video_ids, y_true_video, y_pred_video):
    print(f"{video_id:50s} | True: {CLASS_NAMES[true_index]:8s} | Predicted: {CLASS_NAMES[predicted_index]}")


# =====================================================================
# 21. FINAL TRAINER METRICS
# =====================================================================

print("\nTrainer prediction metrics")
print("--------------------------")
print(outputs.metrics)
