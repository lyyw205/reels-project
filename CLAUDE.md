# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

숙소 마케팅 쇼츠 영상 자동화 파이프라인. Two core pipelines:
1. **Analysis Pipeline**: Reference video → structure extraction → Template DB
2. **Production Pipeline**: Accommodation images → feature extraction → storyboard → render spec

## Commands

```bash
source .venv/bin/activate

# Analysis (reference video analysis)
reels analyze <video_path> --save-db     # full pipeline
reels ingest <video_path>                # ingest only
reels segment <video_path>               # shot segmentation only
reels analyze-shots <video_path>         # per-shot analysis only

# Production (shorts storyboard generation)
reels produce <img1> <img2> ... --name "숙소명" --target couple --output output/
reels produce <img1> <img2> --no-web     # skip web verification
reels produce <img1> <img2> --i2v       # with image-to-video conversion

# Template DB
reels db list
reels db search --place bedroom
reels db export <template_id> out.json

# Tests
.venv/bin/python -m pytest tests/ -q              # all tests (575 passed)
.venv/bin/python -m pytest tests/unit/production/ -q  # production only
.venv/bin/python -m pytest tests/unit/test_cli_produce.py -q  # single file
.venv/bin/python -m pytest tests/ --run-slow       # include slow tests
.venv/bin/python -m pytest tests/ --cov=reels      # with coverage

# Linting
ruff check reels/
mypy reels/
```

## Architecture

Two pipelines share `reels/models/` and `reels/db/`:

### Analysis Pipeline

```
reels/pipeline.py (orchestrator, with PipelineState for resume)
  → reels/ingest/         download (yt-dlp) + normalize (FFmpeg)
  → reels/segmentation/   shot detection (PySceneDetect) + keyframe extraction
  → reels/analysis/       5 analyzers via Analyzer Protocol + AnalysisRunner
  → reels/synthesis/      combine into Template JSON
  → reels/storage.py      TemplateArchiver (file archive)
  → reels/db/repository.py  SQLite persistence
```

### Production Pipeline

```
reels/production/agent.py (ProductionAgent orchestrator)
  → feature_extractor.py   images → Feature list (VLM, async parallel with semaphore)
  → claim_gate.py          confidence → ClaimLevel (CONFIRMED/PROBABLE/SUGGESTIVE)
  → web_verifier.py        optional web verification for low-confidence features
  → template_matcher.py    select best template from DB (composite scoring)
  → copy_writer.py         marketing copy generation with factual claims checking
  → storyboard_builder.py  shot assembly (role assignment, timing, asset mapping)
  → render_spec.py         Remotion render spec + SRT subtitles
  → i2v/                   (optional Image-to-Video, --i2v flag, requires pip install ".[i2v]")
```

### Shared

- `reels/models/` — Shot, Template, VideoMetadata, IngestResult (Pydantic v2)
- `reels/production/models.py` — 20 production-specific models (AccommodationInput, Feature, VerifiedFeature, Storyboard, RenderSpec, etc.)
- `reels/db/repository.py` — SQLite with `json_each` queries, `search_composite()` for multi-filter
- `reels/config.py` — pydantic-settings with YAML defaults + `REELS_*` env var override

## Key Patterns

### Protocol pattern
`Analyzer` (analysis) and `VLMBackend` (production) are `typing.Protocol` classes for swappable implementations. Analysis has 5 concrete analyzers (place/camera/subtitle/speech/rhythm); production uses `CLIPBackend` by default or `ClaudeVisionBackend` when configured.

### Config propagation
`config/default.yaml` → loaded as dict → each module reads its section:
```python
cfg = (config or {}).get("production", {}).get("feature_extraction", {})
```

### ClaimGate 3-tier system
Controls what copy tone is allowed based on feature confidence:
- `>= 0.75` → CONFIRMED (assertive copy: "노천탕 있는 프라이빗 힐링")
- `0.50-0.75` → PROBABLE (hedged: "노천탕 느낌의 야외 욕실")
- `< 0.50` → SUGGESTIVE (atmospheric only)

Factual words ("무료", "제공", "무제한", "포함", "운영") are only allowed at CONFIRMED level.

### Async + Semaphore
`FeatureExtractor` runs VLM API calls in parallel via `asyncio.gather` with `Semaphore(max_concurrent)` for rate limiting and exponential backoff retry (3 attempts).

### Content-hash cache
`ResponseCache` uses SHA-256 of image bytes to cache VLM responses, avoiding duplicate API calls.

### Async test pattern
pytest-asyncio is **not installed**. All async tests use `asyncio.run()` wrapper:
```python
def test_something():
    async def _run():
        result = await some_async_func()
        assert result == expected
    asyncio.run(_run())
```

### Backend selection
`config/default.yaml` → `production.feature_extraction.backend`:
- `"clip"` (default): Local CLIP model, no API key needed
- `"claude"`: Claude Vision API, requires ANTHROPIC_API_KEY

### Template matching composite scoring
Weights: place_overlap 0.30, duration_fit 0.20, camera_variety 0.15, rhythm_match 0.15, shot_count_fit 0.20.

## Stack

Python 3.12, Pydantic v2, Click, Rich, OpenCV, CLIP (default production backend, local), faster-whisper, EasyOCR, librosa, Anthropic SDK (Claude Vision, optional production backend), SQLite, FFmpeg, yt-dlp.
