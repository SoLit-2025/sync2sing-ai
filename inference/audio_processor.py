import librosa
import noisereduce as nr
import soundfile as sf
import io

def process_user_audio(audio_path):
    try:
        audio, sr = librosa.load(audio_path, sr=44100)

        reduced_noise = nr.reduce_noise(y=audio, sr=sr, stationary=True, prop_decrease=0.75)

        buf = io.BytesIO()
        sf.write(buf, reduced_noise, sr, format='WAV', subtype='PCM_16')
        buf.seek(0)

        return buf
    
    except Exception as e:
        raise RuntimeError(f"오디오 처리 중 오류 발생: {e}")
