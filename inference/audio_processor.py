# inference/audio_processor.py
import librosa
import noisereduce as nr
import numpy as np
from pydub import AudioSegment
from pydub.silence import split_on_silence

class AudioPreprocessor:
    def __init__(self, target_sr=44100):
        self.target_sr = target_sr

    def process_user_audio(self, audio_path):
        """전체 전처리 파이프라인 실행"""
        try:
            # 1. 오디오 로드 및 샘플링 레이트 통일
            audio, sr = librosa.load(audio_path, sr=self.target_sr)
            
            # 2. 노이즈 감소
            reduced_noise  = nr.reduce_noise(
                y=audio, 
                sr=sr,
                stationary=True,
                prop_decrease=0.75
            )
            
            # 3. 침묵 구간 제거
            cleaned_audio = self._remove_silence(reduced_noise, sr)
            
            # 4. 3초 청크 분할
            chunks = self._split_to_chunks(cleaned_audio, sr)
            
            # 5. 정규화 및 Mel 변환
            processed = [self._extract_mel(self._normalize(chunk)) for chunk in chunks]
            
            return processed
            
        except Exception as e:
            raise RuntimeError(f"전처리 실패: {e}")

    def _remove_silence(self, audio, sr):
        """침묵 구간 제거 구현"""
        audio_segment = AudioSegment(
            audio.tobytes(),
            frame_rate=sr,
            sample_width=audio.dtype.itemsize,
            channels=1
        )
        
        chunks = split_on_silence(
            audio_segment,
            min_silence_len=500,
            silence_thresh=-40,
            keep_silence=100
        )
        
        if not chunks:
            return audio
            
        combined = sum(chunks, AudioSegment.empty())
        return np.array(combined.get_array_of_samples()) / 32768.0

    def _split_to_chunks(self, audio, sr):
        """3초 청크 분할 로직"""
        chunk_size = int(3 * sr)
        
        if len(audio) != chunk_size:
            audio = audio[:chunk_size]  # 초과분 자르기
            if len(audio) < chunk_size:  # 부족분 패딩
                audio = np.pad(audio, (0, chunk_size - len(audio)))
    
        return [audio]  # 1개 청크만 반환

    def _normalize(self, audio):
        """정규화 처리"""
        return (audio - np.mean(audio)) / (np.std(audio) + 1e-8)

    def _extract_mel(self, audio):
        """Mel 스펙트로그램 추출"""
        mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=self.target_sr,
        n_mels=128,
        n_fft=2048,
        hop_length=512
    )

        # 시간 축 보정 (258프레임으로 고정)
        max_frames = 258  # 모델 입력 크기에 맞춤
        if mel_spec.shape[1] > max_frames:
            mel_spec = mel_spec[:, :max_frames]  # 초과분 트리밍
        elif mel_spec.shape[1] < max_frames:
            # 부족분 제로 패딩
            pad_width = max_frames - mel_spec.shape[1]
            mel_spec = np.pad(mel_spec, ((0,0), (0,pad_width)), mode='constant')
        return librosa.power_to_db(mel_spec, ref=np.max)
