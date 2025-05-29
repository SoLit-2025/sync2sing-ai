import librosa
import numpy as np
import os
import pickle

def extract_mel_spectrogram(audio_path, sr=44100, n_mels=128, n_fft=2048, hop_length=512):
    """
    멜스펙트로그램 추출 
    """
    # 오디오 로드
    y, sr = librosa.load(audio_path, sr=sr)
    
    # 멜스펙트로그램 추출
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
    )
    
    # 로그 스케일 변환 (dB)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    return mel_spec_db

def process_vocalset_mel_extraction():
    """
    VocalSet 멜스펙트로그램 추출
    """
    normalized_base = "data/normalized"
    spectrograms_base = "data/spectrograms"
    
    for data_type in ["data_by_singer", "data_by_technique", "data_by_vowel"]:
        input_dir = os.path.join(normalized_base, data_type)
        output_dir = os.path.join(spectrograms_base, data_type)
        
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
                    
                    # .wav -> .pkl 변환
                    output_filename = file.replace('.wav', '.pkl')
                    output_path = os.path.join(output_subdir, output_filename)
                    
                    try:
                        mel_spec = extract_mel_spectrogram(input_path)
                        
                        # 저장
                        with open(output_path, 'wb') as f:
                            pickle.dump(mel_spec, f)
                            
                        print(f"추출 완료: {file}")
                    except Exception as e:
                        print(f"오류 발생: {file}, {e}")

if __name__ == "__main__":
    process_vocalset_mel_extraction()
