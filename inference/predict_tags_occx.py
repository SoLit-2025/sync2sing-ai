import onnxruntime as ort
import numpy as np
from inference.audio_processor import AudioPreprocessor

class TagPredictorONNX:
    def __init__(self, model_path, label_mapping):
        self.device = "cpu"
        self.idx_to_label = {v: k for k, v in label_mapping.items()}
        self.session = ort.InferenceSession(model_path)
        self.preprocessor = AudioPreprocessor()

    def _softmax(self, x):
        x = np.asarray(x, dtype=np.float32)
        x = x.reshape(-1)
        x = x - np.max(x)
        ex = np.exp(x)
        return ex / (np.sum(ex) + 1e-12)

    def predict(self, audio_path, top_k=3):
        """
        - audio_path: .wav 파일 경로
        - top_k: 상위 k개 태그 반환
        """
        try:
            # 1. 오디오 전처리
            processed = self.preprocessor.process_user_audio(audio_path)
            if len(processed) == 0:
                raise ValueError("처리된 오디오 청크가 없습니다.")

            predictions = []
            for mel in processed:
                # ONNX 모델 입력 shape에 맞게: (1, 1, 128, 258)
                input_array = np.expand_dims(np.expand_dims(mel, axis=0), axis=0).astype(np.float32)
                ort_inputs = {self.session.get_inputs()[0].name: input_array}
                prob = self.session.run(None, ort_inputs)[0]  # shape: (1, 클래스 수)
                prob = self._softmax(prob)
                predictions.append(prob)

            # 3. 결과 집계
            avg_prob = np.mean(predictions, axis=0).flatten()
            top_indices = np.argsort(avg_prob)[-top_k:][::-1]

            # 결과 정렬해서 반환
            return sorted(
                [(self.idx_to_label[idx], float(avg_prob[idx])) for idx in top_indices],
                key=lambda x: x[1], 
                reverse=True
            )

        except Exception as e:
            raise RuntimeError(f"ONNX 예측 실패: {e}")

# 사용 예시
if __name__ == "__main__":
    LABEL_MAPPING = {
        'belt': 0, 'breathy': 1, 'fast_forte': 2, 'fast_piano': 3,
        'forte': 4, 'inhaled': 5, 'lip_trill': 6, 'messa': 7,
        'pp': 8, 'slow_forte': 9, 'slow_piano': 10, 'spoken': 11,
        'straight': 12, 'trill': 13, 'vibrato': 14, 'vocal_fry': 15
    }
    model_path = "weights/model.onnx"
    predictor = TagPredictorONNX(model_path, LABEL_MAPPING)
    result = predictor.predict("샘플_음성파일.wav")
    for i, (tag, prob) in enumerate(result, 1):
        print(f"{i}. {tag}: {prob*100:.2f}%")
