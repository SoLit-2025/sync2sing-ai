# train_note_level_multilabel.py
# ------------------------------------------------------------
# Note 길이만큼 오디오를 자르고 멜스펙으로 만들어
# bending / vibrato / breath 멀티라벨 분류기 학습
# ------------------------------------------------------------
import datetime
import glob
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from backports.zoneinfo import ZoneInfo
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm


def _save_dir_default():
    ts = datetime.datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d_%H-%M")
    try:
        project_root = Path(__file__).resolve().parent   # 스크립트 파일 위치 기준
    except NameError:
        project_root = Path.cwd()                        # __file__ 없는 환경(Jupyter) 대응
    return str(project_root / "weights" / ts)


# -------------------------------
# Config
# -------------------------------
@dataclass
class Config:
    # 데이터 루트(예: "C:/USERS/DS/DOWNLOADS/177.다음색 가이드보컬 데이터/01.데이터")
    DATA_ROOT: str = r"C:\USERS\DS\DOWNLOADS\177.다음색 가이드보컬 데이터\01.데이터"

    # 라벨 폴더(학습/검증). 폴더 구조는 질문에서 준 경로 그대로 사용
    LABEL_TRAIN_SUBDIR: str = os.path.join("1.Training", "라벨링데이터", "TL1")
    LABEL_VALID_SUBDIR: str = os.path.join("2.Validation", "라벨링데이터", "VL1")

    # 음원 폴더(학습/검증). TS1/VS1에 원천 wav 존재
    AUDIO_TRAIN_SUBDIR: str = os.path.join("1.Training", "원천데이터", "TS1")
    AUDIO_VALID_SUBDIR: str = os.path.join("2.Validation", "원천데이터", "VS1")

    # 오디오/멜스펙 파라미터
    TARGET_SR: int = 16000
    N_MELS: int = 128
    N_FFT: int = 1024              # ~64ms @16k
    HOP_LENGTH: int = 256          # 16ms hop
    FMIN: int = 50
    FMAX: int = 8000

    # 멜 시간 프레임 고정 길이(패딩/트렁케이션)
    # 노트가 짧아도 CNN은 고정 길이를 기대하므로, 짧으면 반복/패딩, 길면 컷
    MAX_FRAMES: int = 64           # 64*16ms ≈ 1.02s (대부분 노트 커버)

    # 학습 파라미터
    BATCH_SIZE: int = 32
    EPOCHS: int = 20
    LR: float = 1e-3
    WEIGHT_DECAY: float = 1e-5

    # 데이터 밸런스가 한쪽으로 치우칠 수 있어 class weight 사용 권장
    USE_CLASS_WEIGHTS: bool = True

    # 저장
    SAVE_DIR: str = field(default_factory=_save_dir_default)
    MODEL_NAME: str = "model_aihub.pth"


CFG = Config()
os.makedirs(CFG.SAVE_DIR, exist_ok=True)


# -------------------------------
# 유틸: 경로 매핑 (라벨 json → 원천 wav)
# -------------------------------
def map_json_to_wav(json_path: str, data_root: str, is_train: bool) -> str:
    """
    라벨 경로(TL1/VL1)에 있는 json 경로를 원천 wav 경로(TS1/VS1)로 변환.
    디렉터리 구조/파일명은 동일하다고 가정하고, 확장자만 .wav로 치환.
    """
    rel = os.path.relpath(json_path, start=os.path.join(data_root,
                                                        CFG.LABEL_TRAIN_SUBDIR if is_train else CFG.LABEL_VALID_SUBDIR))
    audio_root = os.path.join(data_root, CFG.AUDIO_TRAIN_SUBDIR if is_train else CFG.AUDIO_VALID_SUBDIR)
    # 확장자 교체(.json → .wav 또는 .WAV)
    base, _ = os.path.splitext(rel)
    # 실제 파일은 .wav 또는 .WAV일 수 있음 → 둘 다 탐색
    candidates = [os.path.join(audio_root, base + ".wav"),
                  os.path.join(audio_root, base + ".WAV")]
    for c in candidates:
        if os.path.exists(c):
            return c
    # 간혹 파일명이 조금 다를 수 있어, 같은 폴더에서 base 접두로 검색
    folder = os.path.dirname(os.path.join(audio_root, rel))
    stem = os.path.basename(base)
    hits = glob.glob(os.path.join(folder, stem + ".*"))
    for h in hits:
        if h.lower().endswith((".wav", ".flac")):
            return h
    raise FileNotFoundError(f"WAV not found for JSON: {candidates[0]}")


# -------------------------------
# 오디오/멜스펙 기능
# -------------------------------
def load_audio(path: str, target_sr: int) -> Tuple[np.ndarray, int]:
    y, sr = librosa.load(path, sr=target_sr, mono=True)
    return y, target_sr


def slice_by_time(y: np.ndarray, sr: int, start_s: float, end_s: float) -> np.ndarray:
    a = int(max(0, math.floor(start_s * sr)))
    b = int(min(len(y), math.ceil(end_s * sr)))
    if b <= a:
        # 최소 40ms 패딩
        b = min(len(y), a + int(0.04 * sr))
    return y[a:b]


def rms_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - x.mean()
    s = np.sqrt((x ** 2).mean() + eps)
    return x / s


def mel_spectrogram(x: np.ndarray, sr: int) -> np.ndarray:
    S = librosa.feature.melspectrogram(
        y=x, sr=sr,
        n_fft=CFG.N_FFT,
        hop_length=CFG.HOP_LENGTH,
        n_mels=CFG.N_MELS,
        fmin=CFG.FMIN,
        fmax=CFG.FMAX,
        power=2.0
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db  # [n_mels, T]


def pad_or_crop_time(A: np.ndarray, max_frames: int) -> np.ndarray:
    """
    A: [n_mels, T]
    - 길면 앞에서 자르고, 짧으면 반-반 패딩(제로)으로 고정 길이
    """
    n_mels, T = A.shape
    if T == max_frames:
        return A
    if T > max_frames:
        return A[:, :max_frames]
    # pad
    pad_total = max_frames - T
    left = pad_total // 2
    right = pad_total - left
    return np.pad(A, ((0, 0), (left, right)), mode="constant")


# -------------------------------
# 데이터셋
# -------------------------------
class NoteJsonDataset(Dataset):
    def __init__(self, json_paths: List[str], data_root: str, is_train: bool):
        self.items = []                 # (wav_path, st, et, label[3])
        self._mel_cache = {}            # wav_path -> full mel [n_mels, T]
        self._build(json_paths, data_root, is_train)

    # 전체 파일 멜스펙 캐시
    def _get_full_mel_cached(self, wav_path: str) -> np.ndarray:
        mel = self._mel_cache.get(wav_path)
        if mel is not None:
            return mel
        # 빠른 로드를 위해 soundfile 사용(가능하면)
        y, sr = librosa.load(wav_path, sr=CFG.TARGET_SR, mono=True)
        y = rms_normalize(y)
        mel = mel_spectrogram(y, sr)         # [n_mels, T_all]
        self._mel_cache[wav_path] = mel
        return mel
    
    # 시간(s) → 멜 프레임 슬라이스
    def _slice_mel_by_time(self, mel: np.ndarray, st_s: float, et_s: float, sr: int) -> np.ndarray:
        hop_t = CFG.HOP_LENGTH / sr
        a = max(0, int(math.floor(st_s / hop_t)))
        b = min(mel.shape[1], int(math.ceil(et_s / hop_t)))
        if b <= a:  # 최소 몇 프레임은 확보
            b = min(mel.shape[1], a + 3)
        return mel[:, a:b]

    def _build(self, json_paths: List[str], data_root: str, is_train: bool):
        for jp in tqdm(json_paths, desc="Indexing JSON"):
            try:
                with open(jp, "r", encoding="utf-8") as f:
                    obj = json.load(f)
            except Exception as e:
                print(f"[WARN] JSON read fail: {jp} ({e})")
                continue

            try:
                wav_path = map_json_to_wav(jp, data_root, is_train)
            except FileNotFoundError as e:
                print(f"[WARN] {e}")
                continue

            notes = obj.get("notes", [])
            for n in notes:
                # 필드 이름은 예시 그대로 사용
                st = float(n.get("start_time", 0.0))
                et = float(n.get("end_time", 0.0))
                if et <= st:
                    continue  # 잘못된 구간 스킵

                # 라벨(멀티라벨: bending, vibrato, breath)
                lb_bend = 1 if n.get("is_bending", False) else 0
                lb_vibr = 1 if n.get("is_vibrt", False) else 0
                lb_bre  = 1 if n.get("is_breath", False) else 0
                label = np.array([lb_bend, lb_vibr, lb_bre], dtype=np.float32)

                # 너무 짧은 구간(예: <80ms)은 학습 안정 위해 스킵
                if (et - st) < 0.08:
                    continue

                self.items.append((wav_path, st, et, label))

        self.items.sort(key=lambda it: it[0])  # wav_path 기준 정렬

        if len(self.items) == 0:
            raise RuntimeError("No training items found. Check paths and JSON format.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        wav_path, st, et, label = self.items[idx]
        mel_full = self._get_full_mel_cached(wav_path)
        mel_seg  = self._slice_mel_by_time(mel_full, st, et, CFG.TARGET_SR)  # ← 핵심
        mel_seg  = pad_or_crop_time(mel_seg, CFG.MAX_FRAMES)
        x = torch.tensor(mel_seg, dtype=torch.float32).unsqueeze(0)
        y_lab = torch.tensor(label, dtype=torch.float32)
        return x, y_lab


# -------------------------------
# 모델 (가벼운 2D CNN + GAP)
# -------------------------------
class SmallCNN(nn.Module):
    def __init__(self, n_mels: int = 128, n_classes: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),  # [n_mels/2, T/2]

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),  # [n_mels/4, T/4]

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.Dropout(0.2),
        )
        # Global Average Pooling over [freq, time]
        self.classifier = nn.Linear(128, n_classes)

    def forward(self, x):
        # x: [B, 1, n_mels, T]
        h = self.features(x)           # [B, 128, n_mels/4, T/4]
        h = h.mean(dim=[2, 3])         # GAP → [B, 128]
        logits = self.classifier(h)    # [B, 3]
        return logits


# -------------------------------
# 학습/평가
# -------------------------------
def compute_class_weights(dataset: NoteJsonDataset) -> torch.Tensor:
    """
    멀티라벨 클래스별 불균형 보정용 가중치 (pos_weight in BCEWithLogitsLoss)
    pos_weight = (N - P) / P
    """
    labels = [it[3] for it in dataset.items]  # numpy arrays
    labels = np.stack(labels, axis=0)         # [N, 3]
    weights = []
    N = labels.shape[0]
    for c in range(labels.shape[1]):
        P = labels[:, c].sum()
        if P == 0:
            w = 1.0
        else:
            w = (N - P) / max(P, 1.0)
        weights.append(w)
    return torch.tensor(weights, dtype=torch.float32)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for xb, yb in tqdm(loader, leave=False):
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = np.zeros(3, dtype=np.int64)
    total_count = np.zeros(3, dtype=np.int64)

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        total_loss += loss.item() * xb.size(0)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).long().cpu().numpy()
        ytrue = yb.long().cpu().numpy()
        total_correct += (preds == ytrue).sum(axis=0)
        total_count += preds.shape[0]

    acc_per_class = (total_correct / np.maximum(total_count, 1)).tolist()
    return total_loss / len(loader.dataset), acc_per_class


def collect_json_files(root: str) -> List[str]:
    # 라벨 루트 아래 모든 json 파일 수집
    return [p for p in glob.glob(os.path.join(root, "**", "*.json"), recursive=True)]


def main():
    print("=====실행=====")
    os.makedirs(CFG.SAVE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # 라벨 파일 수집
    train_label_root = os.path.join(CFG.DATA_ROOT, CFG.LABEL_TRAIN_SUBDIR)
    valid_label_root = os.path.join(CFG.DATA_ROOT, CFG.LABEL_VALID_SUBDIR)
    train_jsons = collect_json_files(train_label_root)
    valid_jsons = collect_json_files(valid_label_root)

    # Dataset / DataLoader
    train_ds = NoteJsonDataset(train_jsons, CFG.DATA_ROOT, is_train=True)
    valid_ds = NoteJsonDataset(valid_jsons, CFG.DATA_ROOT, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=CFG.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # Model / Loss / Optim
    model = SmallCNN(n_mels=CFG.N_MELS, n_classes=3).to(device)

    if CFG.USE_CLASS_WEIGHTS:
        pos_weight = compute_class_weights(train_ds).to(device)
        print("pos_weight:", pos_weight)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY)

    # Train
    best_val = 1e9
    for epoch in range(1, CFG.EPOCHS + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, acc_pc = evaluate(model, valid_loader, criterion, device)
        print(f"[{epoch:02d}] train_loss={tr_loss:.4f}  val_loss={va_loss:.4f}  "
              f"acc(bend/vibr/breath)={acc_pc}")

        # Save best
        if va_loss < best_val:
            best_val = va_loss
            save_path = os.path.join(CFG.SAVE_DIR, CFG.MODEL_NAME)
            torch.save({
                "model": model.state_dict(),
                "cfg": CFG.__dict__,
                "val_loss": best_val
            }, save_path)
            print(f"Saved: {save_path}")

    print("Done.")


if __name__ == "__main__":
    main()
