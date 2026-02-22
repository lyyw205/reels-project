# Reels Analyzer

레퍼런스 영상에서 구조(템플릿)를 추출하고, 새 숙소 자산으로 자동 재조합하는 파이프라인.

## 설치

```bash
# CPU 전용 torch 설치 (권장)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 패키지 설치
pip install -e ".[dev]"

# (선택) PaddleOCR 설치 (정확도 우선)
pip install -e ".[paddleocr]"
```

## 사용법

```bash
# 전체 파이프라인
reels analyze video.mp4 -o ./output

# 개별 단계
reels ingest video.mp4
reels segment normalized.mp4
reels analyze-shots normalized.mp4 --only place camera

# 템플릿 DB
reels db list
reels db search --place bedroom --bpm-min 100
reels db export reels_001 ./template.json
```

## 파이프라인 구조

```
Ingest → Shot Segmentation → Per-shot Analysis → Template Synthesis
  ↓           ↓                    ↓                    ↓
다운로드    PySceneDetect     CLIP/OCR/Whisper     Template JSON
정규화     키프레임 추출     optical flow/librosa    SQLite DB
```

## 설정

`config/default.yaml`에서 기본값 설정. 환경변수로 오버라이드 가능:

```bash
REELS_ANALYSIS_SPEECH_MODEL=large-v3 reels analyze video.mp4
```

## 테스트

```bash
pytest tests/unit/ -v              # 빠른 유닛 테스트
pytest tests/ -v --run-slow        # 전체 (slow 포함)
pytest tests/ --cov=reels          # 커버리지
```
