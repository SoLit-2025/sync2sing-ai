import os
import glob
import pickle
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset

class AnnotatedVocalSetDataset(Dataset):
    def __init__(self, spectrogram_dir, annotation_dir, task_type="technique"):
        self.spectrogram_dir = spectrogram_dir
        self.annotation_dir = annotation_dir
        self.task_type = task_type
        self.file_list = []
        self.label_list = []
        self.label_to_idx = {}
        self.annotations = {}

        if task_type == "singer":
            self.data_folder = os.path.join(spectrogram_dir, "data_by_singer")
        elif task_type == "technique":
            self.data_folder = os.path.join(spectrogram_dir, "data_by_technique")
        elif task_type == "vowel":
            self.data_folder = os.path.join(spectrogram_dir, "data_by_vowel")
        else:
            raise ValueError("task_type은 'singer', 'technique', 'vowel' 중 하나여야 합니다.")
        
        self._load_data()
        self._load_annotations()

    def _load_data(self):
        pkl_files = glob.glob(os.path.join(self.data_folder, "**/*.pkl"), recursive=True)
        labels = set()

        for file_path in pkl_files:
            label = self._extract_label_from_path(file_path)
            labels.add(label)
            self.file_list.append(file_path)

        self.label_to_idx = {label: idx for idx, label in enumerate(sorted(labels))}
        self.label_list = [self.label_to_idx[self._extract_label_from_path(f)] for f in self.file_list]
        
        print(f"총 {len(self.file_list)}개 파일 로드됨")
        print(f"클래스 수: {len(self.label_to_idx)}")
        print(f"클래스들: {list(self.label_to_idx.keys())}")

    def _load_annotations(self):
        for file_path in self.file_list:
            # 청크 파일명에서 원본 오디오 파일명 추출
            base_name = self._extract_original_filename(file_path)
            annotation_path = os.path.join(self.annotation_dir, f"{base_name}.csv")
            
            if os.path.exists(annotation_path):
                try:
                    self.annotations[file_path] = pd.read_csv(annotation_path)
                except Exception as e:
                    print(f"주석 파일 로드 실패: {annotation_path}, 오류: {e}")

    def _extract_original_filename(self, chunk_file_path):
        """청크 파일명에서 원본 파일명 추출"""
        # 예: f1_arpeggios_belt_c_a_chunk_000.pkl → f1_arpeggios_belt_c_a
        filename = os.path.basename(chunk_file_path)
        filename_without_ext = os.path.splitext(filename)[0]  # .pkl 제거
        
        # _chunk_XXX 부분 제거
        if "_chunk_" in filename_without_ext:
            base_name = filename_without_ext.split("_chunk_")[0]
        else:
            base_name = filename_without_ext
            
        return base_name

    def _extract_chunk_number(self, chunk_file_path):
        """청크 파일명에서 청크 번호 추출"""
        # 예: f1_arpeggios_belt_c_a_chunk_000.pkl → 0
        filename = os.path.basename(chunk_file_path)
        if "_chunk_" in filename:
            chunk_part = filename.split("_chunk_")[1]
            chunk_num = int(chunk_part.split(".")[0])  
            return chunk_num
        return 0

    def __getitem__(self, idx):
        # Mel Spectrogram 로드
        file_path = self.file_list[idx]
        with open(file_path, 'rb') as f:
            mel = pickle.load(f)  # (n_mels, time)
        
        # tag
        label = self.label_list[idx]

        # 주석 필터링 (3초 청크에 해당하는 부분만)
        chunk_num = self._extract_chunk_number(file_path)
        chunk_start = chunk_num * 3.0
        chunk_end = chunk_start + 3.0
        annotation = self._filter_annotations(file_path, chunk_start, chunk_end)

        return (
            torch.tensor(mel, dtype=torch.float32).unsqueeze(0),  # (1, n_mels, time)
            torch.tensor(label, dtype=torch.long),
            annotation
        )

    def _filter_annotations(self, file_path, chunk_start, chunk_end):
        """청크 시간에 맞는 주석 추출"""
        if file_path not in self.annotations:
            return {}
        
        df = self.annotations[file_path]
        
        # 주석 파일에 'onset', 'offset' 컬럼이 있는지 확인
        if 'onset' in df.columns and 'offset' in df.columns:
            mask = (df['onset'] >= chunk_start) & (df['offset'] <= chunk_end)
            return df[mask]
        else:
            # 다른 형태의 주석 파일이면 전체 반환
            return df

    def _extract_label_from_path(self, file_path):
        parts = file_path.split(os.sep)
        
        if self.task_type == "singer":
            return parts[parts.index("data_by_singer") + 1]
        elif self.task_type == "technique":
            return parts[parts.index("data_by_technique") + 1]
        elif self.task_type == "vowel":
            return parts[parts.index("data_by_vowel") + 1]

    def __len__(self):
        return len(self.file_list)

    def get_num_classes(self):
        return len(self.label_to_idx)
