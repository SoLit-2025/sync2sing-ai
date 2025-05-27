import librosa
import soundfile as sf
import os
import numpy as np

def split_audio_to_chunks(audio_path, output_dir, chunk_duration=3.0, sr=44100):
    """
    오디오를 3초 청크로 분할
    """
    # 오디오 로드
    y, current_sr = librosa.load(audio_path, sr=sr)
    
    # 청크 크기 계산
    chunk_samples = int(chunk_duration * sr)
    
    # 3초 미만이면 건너뛰기
    if len(y) < chunk_samples:
        print(f"건너뛰기 (길이 부족): {audio_path}")
        return []
    
    # 청크로 분할
    chunks = []
    for i in range(0, len(y) - chunk_samples + 1, chunk_samples):
        chunk = y[i:i + chunk_samples]
        chunks.append(chunk)
    
    # 저장
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    saved_files = []
    
    for i, chunk in enumerate(chunks):
        output_filename = f"{base_name}_chunk_{i:03d}.wav"
        output_path = os.path.join(output_dir, output_filename)
        sf.write(output_path, chunk, sr)
        saved_files.append(output_path)
    
    return saved_files

def process_vocalset_segmentation():
    """
    VocalSet 전체 데이터 분할
    """
    processed_base = "data/processed"
    segments_base = "data/segments"
    
    for data_type in ["cleaned_data_by_singer", "cleaned_data_by_technique", "cleaned_data_by_vowel"]:
        input_dir = os.path.join(processed_base, data_type)
        output_dir = os.path.join(segments_base, data_type.replace("cleaned_", ""))
        
        if not os.path.exists(input_dir):
            continue
            
        os.makedirs(output_dir, exist_ok=True)
        
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if file.endswith('.wav'):
                    input_path = os.path.join(root, file)
                    rel_path = os.path.relpath(root, input_dir)
                    output_subdir = os.path.join(output_dir, rel_path)
                    os.makedirs(output_subdir, exist_ok=True)
                    
                    try:
                        chunks = split_audio_to_chunks(input_path, output_subdir)
                        print(f"분할 완료: {file}, {len(chunks)}개 청크")
                    except Exception as e:
                        print(f"오류 발생: {file}, {e}")

if __name__ == "__main__":
    process_vocalset_segmentation()
