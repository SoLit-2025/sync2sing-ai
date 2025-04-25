# sync2sing-ai

AI 기반 발성 피드백 시스템

## 주요 기능

- 🎧 오디오 전처리 (보컬 분리, 정규화 등)
- 🧠 CNN 기반 보컬 태그 예측
- 🗣 태그 기반 피드백 문장 생성
- 🧪 사용자 녹음 테스트 및 결과 리포트
- 🌐 REST API (FastAPI) 기반 모델 서버 제공

## 디렉토리 구조

- `data/` : 오디오, spectrogram, 레이블 등 데이터 저장
- `preprocessing/` : 데이터 전처리 스크립트
- `model/` : CNN 모델 정의 및 학습
- `inference/` : 사용자 테스트 및 태그 예측
- `feedback/` : 피드백 문장 생성 시스템
- `notebooks/` : Colab / Jupyter 실험 노트북
- `utils/` : 공통 유틸 함수
- `configs/` : 하이퍼파라미터 설정
- `tests/` : 테스트 코드
- `api/` : FastAPI 기반 API 서버

## 설치 방법

```bash
# 1. 가상환경 생성 (선택)
python -m venv venv
venv\\Scripts\\activate

# 2. 필수 라이브러리 설치
pip install -r requirements.txt

# 3. FastAPI 서버 실행 (예시)
uvicorn api.main:app --reload

## 개발 환경

- Python 3.8 이상
- VSCode + Google Colab
- FastAPI (for REST API server)
```
>>>>>>> a2bf9e3 (🎉 Tada: AI 프로젝트 초기 설정)
