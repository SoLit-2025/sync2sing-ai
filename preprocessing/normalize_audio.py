import librosa
import soundfile as sf
import numpy as np
import os

def normalize_audio(audio_path, output_path):
    """
    오디오를 평균 0, 분산 1로 정규화해서 저장
    """
    # 오디오 로드
    y, sr = librosa.load(audio_path, sr=None)
    
    # 정규화 (평균 0, 표준편차 1)
    mean = np.mean(y)
    std = np.std(y)
    if std == 0:
        y_normalized = y - mean  # 무음 파일 등 표준편차 0일 때 예외 처리
    else:
        y_normalized = (y - mean) / std
    
    # 저장
    sf.write(output_path, y_normalized, sr)
    return y_normalized

def process_vocalset_normalization():
    """
    VocalSet 세그먼트 정규화
    """
    segments_base = "data/segments"
    normalized_base = "data/normalized"
    
    for data_type in ["data_by_singer", "data_by_technique", "data_by_vowel"]:
        input_dir = os.path.join(segments_base, data_type)
        output_dir = os.path.join(normalized_base, data_type)
        
        if not os.path.exists(input_dir):
            continue
            
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if file.endswith('.wav'):
                    input_path = os.path.join(root, file)
                    rel_path = os.path.relpath(root, input_dir)
                    output_subdir = os.path.join(output_dir, rel_path)
                    os.makedirs(output_subdir, exist_ok=True)
                    output_path = os.path.join(output_subdir, file)
                    
                    try:
                        normalize_audio(input_path, output_path)
                        print(f"정규화 완료: {file}")
                    except Exception as e:
                        print(f"오류 발생: {file}, {e}")

if __name__ == "__main__":
    process_vocalset_normalization()
