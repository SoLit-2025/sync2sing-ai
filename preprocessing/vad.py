import librosa
import numpy as np
from pydub import AudioSegment
from pydub.silence import split_on_silence
import os

def remove_silence(audio_path, output_path, silence_thresh=-40, min_silence_len=500):
    """
    오디오에서 침묵 구간 제거
    """
    # 오디오 로드
    audio = AudioSegment.from_wav(audio_path)
    
    # 침묵 기준으로 분할 후 다시 합치기
    chunks = split_on_silence(
        audio,
        min_silence_len=min_silence_len,  # 최소 침묵 길이 (ms)
        silence_thresh=silence_thresh,    # 침묵 임계값 (dBFS)
        keep_silence=100  # 침묵 일부 유지 (ms)
    )
    
    # 분할된 청크들을 다시 합치기
    cleaned_audio = AudioSegment.empty()
    for chunk in chunks:
        cleaned_audio += chunk
    
    # 저장
    cleaned_audio.export(output_path, format="wav")
    return len(cleaned_audio) / 1000.0  # 길이 반환 (초)

def process_vocalset_silence_removal():
    """
    VocalSet 전체 데이터에서 침묵 제거
    """
    raw_base = "data/raw/VocalSet1-2"
    output_base = "data/processed"
    
    for data_type in ["data_by_singer", "data_by_technique", "data_by_vowel"]:
        input_dir = os.path.join(raw_base, data_type)
        output_dir = os.path.join(output_base, f"cleaned_{data_type}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if file.endswith('.wav'):
                    input_path = os.path.join(root, file)
                    rel_path = os.path.relpath(root, input_dir)
                    output_subdir = os.path.join(output_dir, rel_path)
                    os.makedirs(output_subdir, exist_ok=True)
                    output_path = os.path.join(output_subdir, file)
                    
                    try:
                        duration = remove_silence(input_path, output_path)
                        print(f"처리 완료: {file}, 길이: {duration:.2f}초")
                    except Exception as e:
                        print(f"오류 발생: {file}, {e}")


if __name__ == "__main__":
    process_vocalset_silence_removal()

