import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np

# 프로젝트 루트 경로 설정
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.cnn_model import AnnotatedVocalSetCNN
from audio_processor import AudioPreprocessor

class TagPredictor:
    def __init__(self, model_path, label_mapping):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.idx_to_label = {v:k for k,v in label_mapping.items()}
        self.model = self._load_model(model_path)
        self.preprocessor = AudioPreprocessor()

    def _load_model(self, model_path):
        """모델 로드 구현"""
        model = AnnotatedVocalSetCNN(num_classes=len(self.idx_to_label))
        model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        model.to(self.device).eval()
        return model

    def predict(self, audio_path, top_k=3):
        """예측 수행 메인 함수"""
        try:
            # 1. 전처리
            processed = self.preprocessor.process_user_audio(audio_path)
            
            if len(processed) == 0:
                raise ValueError("처리된 오디오 청크가 없습니다.")
            
            # 2. 모델 추론
            predictions = []
            for mel in processed:
                tensor = torch.tensor(mel, device=self.device).unsqueeze(0).unsqueeze(0)

                expected_shape = (1, 1, 128, 258)
                if tensor.shape != expected_shape:
                    raise ValueError(f"잘못된 입력 형태: {tensor.shape} (예상: {expected_shape})")
                
                with torch.no_grad():
                    output = self.model(tensor.float())
                    prob = torch.softmax(output, dim=1).cpu().numpy()
                    predictions.append(prob)
            
            # 3. 결과 집계
            avg_prob = np.mean(predictions, axis=0).flatten()
            top_indices = np.argsort(avg_prob)[-top_k:][::-1]

            
            #  결과 정렬 
            return sorted(
                [(self.idx_to_label[idx], float(avg_prob[idx])) for idx in top_indices],
                key=lambda x: x[1], 
                reverse=True
            )
            
        except Exception as e:
            raise RuntimeError(f"예측 실패: {e}")

if __name__ == "__main__":
    LABEL_MAPPING = {
        'belt': 0,
        'breathy': 1,
        'fast_forte': 2,
        'fast_piano': 3,
        'forte': 4,
        'inhaled': 5,
        'lip_trill': 6,
        'messa': 7,
        'pp': 8,
        'slow_forte': 9,
        'slow_piano': 10,
        'spoken': 11,
        'straight': 12,
        'trill': 13,
        'vibrato': 14,
        'vocal_fry': 15
    }

    model_path = "weights/2025-06-03_00-03/best_model.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
    
    predictor = TagPredictor(
        model_path=model_path,
        label_mapping=LABEL_MAPPING
    )
    
    result = predictor.predict("data/raw/VocalSet1-2/data_by_singer/male5/arpeggios/vocal_fry/m5_arpeggios_vocal_fry_a.wav")

    for i, (tag, prob) in enumerate(result, 1):
        print(f"{i}. {tag}: {prob*100:.2f}%")
