# ================================================================
# 1. IMPORTS
# ================================================================

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
import torch

from PIL import Image
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import ViTImageProcessor, ViTForImageClassification


# ================================================================
# 2. CONFIGURATION
# ================================================================

PUMAVED_VIDEO_DIR = Path(
    r"D:\IIT mandi\Facial datasets\PUMAVE-D"
)

RAW_KEYFRAME_DIR = Path(
    r"D:\IIT mandi\Facial datasets\PUMAVE-D_KEYFRAME\selected_keyframes"
)

ALIGNED_KEYFRAME_DIR = Path(
    r"D:\IIT mandi\Facial datasets\PUMAVE-D_KEYFRAME\aligned_keyframes"
)

DLIB_LANDMARK_MODEL = Path(
    r"C:\PATH\TO\shape_predictor_68_face_landmarks.dat"
)

OUTPUT_DIR = Path(
    r"D:\IIT mandi\Facial datasets\PUMAVE-D_KEYFRAME\attnvision_outputs"
)

MODEL_NAME = "dima806/face_emotions_image_detection"

RUN_KEYFRAME_PIPELINE = False
OVERWRITE_PREPARED_DATA = False
SAVE_RAW_KEYFRAMES = True

VIDEO_EXTENSIONS = {
    ".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg", ".m4v"
}

CLASS_NAMES = ["Anger", "Disgust", "Fear", "Happy", "Sad"]
NUM_CLASSES = len(CLASS_NAMES)
CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}

# PUMAVE-D class discovery is path-based here.
# Neutral is intentionally excluded to preserve the five-class protocol.
# Common noun/adjective variants are supported.
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

FACE_SIZE = 48
FACE_CHIP_PADDING = 0.25
DLIB_UPSAMPLE = 1

LANDMARK_DISPLACEMENT_THRESHOLD_PX = 5.0

VAL_SPLIT = 0.20
SEED = 42
BATCH_SIZE = 8
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
EARLY_STOPPING_PATIENCE = 5
NUM_WORKERS = 0
USE_AMP = True


# ================================================================
# 3. REPRODUCIBILITY / DEVICE
# ================================================================

os.environ.setdefault("PYTHONHASHSEED", str(SEED))
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
print("Using device:", device)


# ================================================================
# 4. PUMAVE-D DISCOVERY / LABEL PARSING
# ================================================================

def infer_pumaved_class(relative_path: Path):
    """
    Infer one retained PUMAVE-D emotion class from the video's parent folders.

    Supported examples:
        Anger/video01.mp4
        Subject01/Anger/video01.mp4

    Neutral/unmatched videos are ignored because the paper retains only
    Anger, Disgust, Fear, Happy, and Sad.
    """
    matches = []

    for part in relative_path.parts[:-1]:
        canonical = PUMAVED_CLASS_LOOKUP.get(part.lower())

        if canonical is not None:
            matches.append((part, canonical))

    if not matches:
        return None, None

    canonical_classes = {canonical for _, canonical in matches}

    if len(canonical_classes) > 1:
        raise ValueError(
            f"Ambiguous emotion folders in '{relative_path}': "
            f"{sorted(canonical_classes)}"
        )

    return matches[0]


def make_video_id(relative_path: Path, matched_class_part: str):
    """
    Create a stable PUMAVE-D source-video ID while removing the emotion folder.

    Example:
        Subject01/Anger/video01.mp4
        -> Subject01__video01
    """
    parts = list(relative_path.with_suffix("").parts)
    kept = []
    removed = False

    for part in parts:
        if not removed and part == matched_class_part:
            removed = True
            continue

        kept.append(part)

    return "__".join(kept) or relative_path.stem


def discover_pumaved_videos(video_root: Path):
    """Find retained five-class PUMAVE-D videos."""
    video_root = Path(video_root)

    if not video_root.exists():
        raise FileNotFoundError(
            f"PUMAVE-D video directory does not exist:\n{video_root}"
        )

    videos = sorted(
        path
        for path in video_root.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )

    if not videos:
        raise FileNotFoundError(
            f"No supported video files found under:\n{video_root}"
        )

    selected = []
    ignored = 0

    for video_path in videos:
        relative_path = video_path.relative_to(video_root)

        matched_class_part, class_name = infer_pumaved_class(
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
            "This implementation expects the emotion name to appear in a "
            "parent folder. If your downloaded PUMAVE-D copy uses emotion "
            "codes in filenames instead, update infer_pumaved_class()."
        )

    print("Total videos found       :", len(videos))
    print("Five-class videos        :", len(selected))
    print("Ignored/unmatched videos :", ignored)

    return selected


# ================================================================
# 5. FFMPEG VIDEO DECODER
# ================================================================

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
    check_ffmpeg_available()

    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=s=x:p=0",
        str(video_path),
    ]

    result = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
    )

    value = result.stdout.strip()

    if "x" not in value:
        raise RuntimeError(
            f"Could not obtain video dimensions: {video_path}"
        )

    width, height = map(int, value.split("x"))
    return width, height


def read_exact(stream, n_bytes):
    chunks = []
    remaining = n_bytes

    while remaining > 0:
        chunk = stream.read(remaining)

        if not chunk:
            break

        chunks.append(chunk)
        remaining -= len(chunk)

    data = b"".join(chunks)

    return data if len(data) == n_bytes else None


def decode_video_frames_ffmpeg(video_path: Path):
    """
    Stream all decoded frames from FFmpeg without writing them to disk first.
    """
    width, height = get_video_dimensions(video_path)
    frame_bytes = width * height * 3

    cmd = [
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
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=10**7,
    )

    frame_index = 0

    try:
        while True:
            raw = read_exact(process.stdout, frame_bytes)

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
            raise RuntimeError(
                f"FFmpeg decoding failed for {video_path}\n"
                + stderr.decode("utf-8", errors="replace")
            )


# ================================================================
# 6. DLIB FACE DETECTION + 68 LANDMARKS
# ================================================================

_face_detector = dlib.get_frontal_face_detector()
_landmark_predictor = None


def get_landmark_predictor():
    global _landmark_predictor

    if _landmark_predictor is None:
        if not DLIB_LANDMARK_MODEL.exists():
            raise FileNotFoundError(
                f"DLib landmark model not found:\n{DLIB_LANDMARK_MODEL}"
            )

        _landmark_predictor = dlib.shape_predictor(
            str(DLIB_LANDMARK_MODEL)
        )

    return _landmark_predictor


def detect_face_shape_landmarks(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = _face_detector(gray, DLIB_UPSAMPLE)

    if len(faces) == 0:
        return None, None, None

    # If several faces exist, use the largest facial region.
    face = max(
        faces,
        key=lambda r: max(0, r.width()) * max(0, r.height())
    )

    shape = get_landmark_predictor()(gray, face)

    landmarks = np.array(
        [
            [shape.part(i).x, shape.part(i).y]
            for i in range(68)
        ],
        dtype=np.float32,
    )

    return face, shape, landmarks


# ================================================================
# 7. LANDMARK-DISPLACEMENT KEY-FRAME SELECTION
# ================================================================

def mean_landmark_displacement(current_landmarks, reference_landmarks):
    """
    D = (1/68) * sum_i ||p_current,i - p_reference,i||_2
    """
    return float(
        np.linalg.norm(
            current_landmarks - reference_landmarks,
            axis=1,
        ).mean()
    )


def align_crop_resize_face(frame_bgr, shape):
    """
    Facial preprocessing:
    landmark-guided alignment -> crop -> resize to 48x48x3.
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    aligned_rgb = dlib.get_face_chip(
        frame_rgb,
        shape,
        size=FACE_SIZE,
        padding=FACE_CHIP_PADDING,
    )

    aligned_bgr = cv2.cvtColor(aligned_rgb, cv2.COLOR_RGB2BGR)

    if aligned_bgr.shape[:2] != (FACE_SIZE, FACE_SIZE):
        aligned_bgr = cv2.resize(
            aligned_bgr,
            (FACE_SIZE, FACE_SIZE),
            interpolation=cv2.INTER_LINEAR,
        )

    return aligned_bgr


def metadata_header():
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
    if not marker_path.exists():
        return False

    try:
        data = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False

    return (
        float(data.get("threshold_px", -1))
        == float(LANDMARK_DISPLACEMENT_THRESHOLD_PX)
        and int(data.get("face_size", -1)) == FACE_SIZE
    )


def prepare_video_keyframes(
    video_path: Path,
    raw_output_dir: Path,
    aligned_output_dir: Path,
    overwrite=False,
):
    """
    Select key frames and preprocess selected faces in one pass.

    The same DLib shape found during selection is reused for alignment,
    avoiding a second landmark-detection pass.
    """
    raw_output_dir.mkdir(parents=True, exist_ok=True)
    aligned_output_dir.mkdir(parents=True, exist_ok=True)

    marker_path = aligned_output_dir / "_complete.json"
    metadata_path = aligned_output_dir / "keyframe_landmarks.csv"

    if not overwrite and completion_marker_matches(marker_path):
        existing = sorted(aligned_output_dir.glob("keyframe_*.png"))

        if existing:
            print(
                f"Skipping {video_path.name}: "
                f"{len(existing)} prepared key frame(s) found."
            )
            return len(existing)

    if overwrite:
        for folder in (raw_output_dir, aligned_output_dir):
            for path in folder.glob("keyframe_*.png"):
                path.unlink()

        for path in (marker_path, metadata_path):
            if path.exists():
                path.unlink()

    selected_count = 0
    decoded_count = 0
    face_count = 0
    last_selected_landmarks = None

    with open(
        metadata_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(metadata_header())

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
                    displacement
                    >= LANDMARK_DISPLACEMENT_THRESHOLD_PX
                )

            if not is_keyframe:
                continue

            selected_count += 1

            filename = (
                f"keyframe_{selected_count:05d}"
                f"_frame_{frame_index:06d}.png"
            )

            # Reuse the detected landmark shape for preprocessing.
            aligned_face = align_crop_resize_face(frame_bgr, shape)

            aligned_path = aligned_output_dir / filename

            if not cv2.imwrite(str(aligned_path), aligned_face):
                raise RuntimeError(
                    f"Could not write aligned frame: {aligned_path}"
                )

            if SAVE_RAW_KEYFRAMES:
                raw_path = raw_output_dir / filename

                if not cv2.imwrite(str(raw_path), frame_bgr):
                    raise RuntimeError(
                        f"Could not write raw frame: {raw_path}"
                    )

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
        "face_size": FACE_SIZE,
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


def prepare_pumaved_keyframes(video_root: Path):
    videos = discover_pumaved_videos(video_root)
    total_selected = 0

    for index, (
        video_path,
        relative_path,
        class_name,
        video_id,
    ) in enumerate(videos, start=1):

        print(
            f"[{index}/{len(videos)}] "
            f"{relative_path} -> {class_name}/{video_id}"
        )

        raw_dir = RAW_KEYFRAME_DIR / class_name / video_id
        aligned_dir = ALIGNED_KEYFRAME_DIR / class_name / video_id

        total_selected += prepare_video_keyframes(
            video_path,
            raw_dir,
            aligned_dir,
            overwrite=OVERWRITE_PREPARED_DATA,
        )

    print("\nKey-frame extraction completed.")
    print("Videos processed       :", len(videos))
    print("Total key frames       :", total_selected)
    print(
        "Landmark threshold     :",
        LANDMARK_DISPLACEMENT_THRESHOLD_PX,
        "pixels",
    )


if RUN_KEYFRAME_PIPELINE:
    prepare_pumaved_keyframes(PUMAVED_VIDEO_DIR)


# ================================================================
# 8. BUILD KEY-FRAME INDEX
# ================================================================

def build_keyframe_index(aligned_root: Path):
    if not aligned_root.exists():
        raise FileNotFoundError(
            f"Aligned key-frame directory not found:\n{aligned_root}\n"
            "Set RUN_KEYFRAME_PIPELINE=True for the first run."
        )

    records = []

    for class_name in CLASS_NAMES:
        class_dir = aligned_root / class_name

        if not class_dir.exists():
            print("WARNING: missing class folder:", class_dir)
            continue

        label = CLASS_TO_INDEX[class_name]

        for video_dir in sorted(
            p for p in class_dir.iterdir() if p.is_dir()
        ):
            frames = sorted(video_dir.glob("keyframe_*.png"))

            if not frames:
                continue

            video_id = f"{class_name}/{video_dir.name}"

            for frame_path in frames:
                records.append(
                    {
                        "image_path": str(frame_path),
                        "label": label,
                        "video_id": video_id,
                    }
                )

    if not records:
        raise FileNotFoundError(
            f"No aligned key frames found under:\n{aligned_root}"
        )

    return records


records = build_keyframe_index(ALIGNED_KEYFRAME_DIR)

print("Total aligned facial key frames:", len(records))


# ================================================================
# 9. VIDEO-LEVEL STRATIFIED 80:20 SPLIT
# ================================================================

def split_records_by_video(records):
    video_to_label = {}

    for record in records:
        video_id = record["video_id"]
        label = int(record["label"])

        if (
            video_id in video_to_label
            and video_to_label[video_id] != label
        ):
            raise ValueError(
                f"Conflicting labels for video: {video_id}"
            )

        video_to_label[video_id] = label

    unique_video_ids = np.asarray(list(video_to_label.keys()))

    unique_video_labels = np.asarray(
        [video_to_label[v] for v in unique_video_ids],
        dtype=np.int64,
    )

    counts = Counter(unique_video_labels.tolist())

    missing = [
        CLASS_NAMES[i]
        for i in range(NUM_CLASSES)
        if counts.get(i, 0) == 0
    ]

    if missing:
        raise ValueError(
            "No usable videos for: " + ", ".join(missing)
        )

    too_small = {
        CLASS_NAMES[i]: count
        for i, count in counts.items()
        if count < 2
    }

    if too_small:
        raise ValueError(
            f"At least two videos/class are required: {too_small}"
        )

    train_videos, val_videos = train_test_split(
        unique_video_ids,
        test_size=VAL_SPLIT,
        random_state=SEED,
        shuffle=True,
        stratify=unique_video_labels,
    )

    train_set = set(train_videos)
    val_set = set(val_videos)

    if train_set & val_set:
        raise RuntimeError("Video-level data leakage detected.")

    train_records = [
        r for r in records if r["video_id"] in train_set
    ]

    val_records = [
        r for r in records if r["video_id"] in val_set
    ]

    return train_records, val_records, train_videos, val_videos


(
    train_records,
    val_records,
    train_video_ids,
    val_video_ids,
) = split_records_by_video(records)

print("\nDataset split")
print("-------------")
print("Training videos    :", len(train_video_ids))
print("Validation videos  :", len(val_video_ids))
print("Training key frames:", len(train_records))
print("Validation frames  :", len(val_records))


# ================================================================
# 10. PROCESSOR / PYTORCH DATASET
# ================================================================

processor = ViTImageProcessor.from_pretrained(MODEL_NAME)


class FacialKeyFrameDataset(Dataset):
    def __init__(self, records, processor):
        self.records = records
        self.processor = processor

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]

        with Image.open(record["image_path"]) as image:
            image = image.convert("RGB")

            processed = self.processor(
                images=image,
                return_tensors="pt",
            )

        return {
            "pixel_values": processed["pixel_values"].squeeze(0),
            "label": torch.tensor(
                record["label"],
                dtype=torch.long,
            ),
            "video_id": record["video_id"],
        }


def collate_fn(batch):
    return {
        "pixel_values": torch.stack(
            [item["pixel_values"] for item in batch]
        ),
        "labels": torch.stack(
            [item["label"] for item in batch]
        ),
        "video_ids": [
            item["video_id"] for item in batch
        ],
    }


train_dataset = FacialKeyFrameDataset(train_records, processor)
val_dataset = FacialKeyFrameDataset(val_records, processor)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
    collate_fn=collate_fn,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
    collate_fn=collate_fn,
)


# ================================================================
# 11. LOAD / ADAPT ATTN VISION (VIT) TO FIVE CLASSES
# ================================================================

id2label = {
    i: class_name
    for i, class_name in enumerate(CLASS_NAMES)
}

label2id = {
    class_name: i
    for i, class_name in id2label.items()
}

model = ViTForImageClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_CLASSES,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)

model.to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=NUM_EPOCHS,
)

amp_enabled = USE_AMP and device.type == "cuda"
scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)


# ================================================================
# 12. TRAINING / FRAME EVALUATION FUNCTIONS
# ================================================================

def train_one_epoch(model, loader):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        pixel_values = batch["pixel_values"].to(
            device,
            non_blocking=True,
        )

        labels = batch["labels"].to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=amp_enabled):
            outputs = model(
                pixel_values=pixel_values,
                labels=labels,
            )

            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        predictions = outputs.logits.argmax(dim=1)
        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        correct += predictions.eq(labels).sum().item()
        total += batch_size

    return (
        total_loss / max(total, 1),
        correct / max(total, 1),
    )


@torch.no_grad()
def evaluate_frames(model, loader):
    model.eval()

    total_loss = 0.0
    total = 0

    all_true = []
    all_pred = []
    all_probs = []
    all_video_ids = []

    for batch in loader:
        pixel_values = batch["pixel_values"].to(
            device,
            non_blocking=True,
        )

        labels = batch["labels"].to(
            device,
            non_blocking=True,
        )

        with torch.cuda.amp.autocast(enabled=amp_enabled):
            outputs = model(
                pixel_values=pixel_values,
                labels=labels,
            )

        probabilities = torch.softmax(outputs.logits, dim=1)
        predictions = probabilities.argmax(dim=1)

        batch_size = labels.size(0)

        total_loss += outputs.loss.item() * batch_size
        total += batch_size

        all_true.extend(labels.cpu().numpy().tolist())
        all_pred.extend(predictions.cpu().numpy().tolist())
        all_probs.append(probabilities.cpu().numpy())
        all_video_ids.extend(batch["video_ids"])

    return {
        "loss": total_loss / max(total, 1),
        "true": np.asarray(all_true, dtype=np.int64),
        "pred": np.asarray(all_pred, dtype=np.int64),
        "probabilities": np.concatenate(all_probs, axis=0),
        "video_ids": np.asarray(all_video_ids),
    }


# ================================================================
# 13. TRAIN WITH EARLY STOPPING
# ================================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

best_model_path = OUTPUT_DIR / "best_pumaved_keyframe_attnvision.pt"

history = {
    "train_loss": [],
    "train_accuracy": [],
    "val_loss": [],
    "val_accuracy": [],
}

best_val_accuracy = -np.inf
epochs_without_improvement = 0

for epoch in range(1, NUM_EPOCHS + 1):
    train_loss, train_accuracy = train_one_epoch(
        model,
        train_loader,
    )

    validation = evaluate_frames(model, val_loader)

    val_accuracy = accuracy_score(
        validation["true"],
        validation["pred"],
    )

    scheduler.step()

    history["train_loss"].append(train_loss)
    history["train_accuracy"].append(train_accuracy)
    history["val_loss"].append(validation["loss"])
    history["val_accuracy"].append(val_accuracy)

    print(
        f"Epoch {epoch:03d}/{NUM_EPOCHS} | "
        f"train_loss={train_loss:.4f} | "
        f"train_acc={train_accuracy:.4f} | "
        f"val_loss={validation['loss']:.4f} | "
        f"val_acc={val_accuracy:.4f}"
    )

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        epochs_without_improvement = 0

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "id2label": id2label,
                "label2id": label2id,
                "best_val_accuracy": best_val_accuracy,
            },
            best_model_path,
        )

    else:
        epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):
            print("Early stopping triggered.")
            break


# ================================================================
# 14. RELOAD BEST MODEL
# ================================================================

checkpoint = torch.load(
    best_model_path,
    map_location=device,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(device)


# ================================================================
# 15. SAVE TRAINING HISTORY / CURVES
# ================================================================

with open(
    OUTPUT_DIR / "training_history.csv",
    "w",
    newline="",
    encoding="utf-8",
) as file:
    writer = csv.writer(file)

    writer.writerow(
        [
            "epoch",
            "train_loss",
            "train_accuracy",
            "val_loss",
            "val_accuracy",
        ]
    )

    for i in range(len(history["train_loss"])):
        writer.writerow(
            [
                i + 1,
                history["train_loss"][i],
                history["train_accuracy"][i],
                history["val_loss"][i],
                history["val_accuracy"][i],
            ]
        )

plt.figure(figsize=(8, 6))
plt.plot(history["train_accuracy"], label="Train Accuracy")
plt.plot(history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("AttnVision/ViT Accuracy - PUMAVE-D Key Frames")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "training_accuracy.png")
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(history["train_loss"], label="Train Loss")
plt.plot(history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("AttnVision/ViT Loss - PUMAVE-D Key Frames")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "training_loss.png")
plt.show()


# ================================================================
# 16. FINAL FRAME-LEVEL EVALUATION
# ================================================================

frame_results = evaluate_frames(model, val_loader)

frame_true = frame_results["true"]
frame_pred = frame_results["pred"]
frame_probabilities = frame_results["probabilities"]
frame_video_ids = frame_results["video_ids"]

LABEL_IDS = np.arange(NUM_CLASSES)

frame_accuracy = accuracy_score(frame_true, frame_pred)

print("\nFrame-Level Evaluation")
print("----------------------")
print(f"Frame-level accuracy: {frame_accuracy * 100:.4f}%")

frame_report = classification_report(
    frame_true,
    frame_pred,
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
    frame_pred,
    labels=LABEL_IDS,
)

fig, ax = plt.subplots(figsize=(8, 6))

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
    "DLib Key-Frame Selection + AttnVision/ViT"
)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "frame_level_confusion_matrix.png")
plt.show()


# ================================================================
# 17. MAJORITY VOTING
# ================================================================

def majority_vote_per_video(
    video_ids,
    frame_probabilities,
    frame_true_labels,
):
    """
    Highest number of frame votes wins.
    If tied, use highest mean softmax probability among tied classes.
    """
    video_ids = np.asarray(video_ids)
    probabilities = np.asarray(frame_probabilities)
    true_labels = np.asarray(frame_true_labels, dtype=np.int64)

    frame_predictions = np.argmax(
        probabilities,
        axis=1,
    )

    ordered_video_ids = np.unique(video_ids)

    video_true = []
    video_pred = []

    for video_id in ordered_video_ids:
        mask = video_ids == video_id

        predictions = frame_predictions[mask]
        probs = probabilities[mask]
        truth = np.unique(true_labels[mask])

        if len(truth) != 1:
            raise ValueError(
                f"Inconsistent labels for video '{video_id}': "
                f"{truth.tolist()}"
            )

        votes = np.bincount(
            predictions,
            minlength=NUM_CLASSES,
        )

        tied_classes = np.flatnonzero(
            votes == votes.max()
        )

        if len(tied_classes) == 1:
            final_prediction = int(tied_classes[0])

        else:
            mean_probabilities = probs.mean(axis=0)

            final_prediction = int(
                tied_classes[
                    np.argmax(
                        mean_probabilities[tied_classes]
                    )
                ]
            )

        video_true.append(int(truth[0]))
        video_pred.append(final_prediction)

    return (
        np.asarray(video_true, dtype=np.int64),
        np.asarray(video_pred, dtype=np.int64),
        ordered_video_ids,
    )


video_true, video_pred, ordered_video_ids = majority_vote_per_video(
    frame_video_ids,
    frame_probabilities,
    frame_true,
)


# ================================================================
# 18. FINAL VIDEO-LEVEL EVALUATION
# ================================================================

video_accuracy = accuracy_score(
    video_true,
    video_pred,
)

print("\nVideo-Level Evaluation After Majority Voting")
print("--------------------------------------------")
print("Videos evaluated    :", len(ordered_video_ids))
print(f"Video-level accuracy: {video_accuracy * 100:.4f}%")

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
    confusion_matrix=video_cm,
    display_labels=CLASS_NAMES,
).plot(
    ax=ax,
    xticks_rotation=45,
    values_format="d",
)

plt.title(
    "Video-Level Confusion Matrix After Majority Voting\n"
    "DLib Key-Frame Selection + AttnVision/ViT"
)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "video_level_confusion_matrix.png")
plt.show()


# ================================================================
# 19. SAVE FINAL VIDEO PREDICTIONS
# ================================================================

with open(
    OUTPUT_DIR / "video_level_predictions.csv",
    "w",
    newline="",
    encoding="utf-8",
) as file:
    writer = csv.writer(file)

    writer.writerow(
        [
            "video_id",
            "true_label",
            "predicted_label",
        ]
    )

    print("\nFinal Per-Video Predictions")
    print("---------------------------")

    for video_id, true_index, pred_index in zip(
        ordered_video_ids,
        video_true,
        video_pred,
    ):
        true_label = CLASS_NAMES[true_index]
        predicted_label = CLASS_NAMES[pred_index]

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
