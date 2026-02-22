# Reels 파이프라인 문서

## 1. 프로젝트 개요

Reels는 **숙소 마케팅 쇼츠 영상 자동화 파이프라인**입니다. 두 개의 독립적인 핵심 파이프라인으로 구성됩니다:

1. **분석 파이프라인 (Analysis Pipeline)**: 레퍼런스 영상 분석 → 구조 추출 → 템플릿 DB 저장
2. **프로덕션 파이프라인 (Production Pipeline)**: 숙소 이미지 → 특징 추출 → 스토리보드 → 렌더 스펙 생성

이 두 파이프라인은 `reels/models/`, `reels/db/`, `reels/config.py`의 공유 컴포넌트를 통해 협력합니다.

---

## 2. 분석 파이프라인 (Analysis Pipeline)

### 2.1 전체 흐름

```
reels analyze <video_path> [--save-db] [--resume]
  │
  ├─→ reels/ingest/        (다운로드 + 정규화)
  │    ├─ VideoDownloader (yt-dlp)
  │    ├─ VideoNormalizer (FFmpeg)
  │    └─ VideoProber (ffprobe)
  │
  ├─→ reels/segmentation/  (샷 감지 + 키프레임 추출)
  │    ├─ ShotDetector (PySceneDetect)
  │    └─ Postprocessing (병합, 키프레임)
  │
  ├─→ reels/analysis/      (5개 분석기)
  │    ├─ PlaceAnalyzer (CLIP embeddings → 장소 분류)
  │    ├─ CameraAnalyzer (optical flow → 카메라 움직임)
  │    ├─ SubtitleAnalyzer (EasyOCR → 자막 감지)
  │    ├─ SpeechAnalyzer (faster-whisper → 음성 인식)
  │    └─ RhythmAnalyzer (librosa → 비트 감지)
  │
  ├─→ reels/synthesis/     (5개 분석 결과 조합)
  │    └─ TemplateAssembler → Template JSON
  │
  ├─→ reels/storage.py     (파일 아카이브)
  │    └─ TemplateArchiver
  │
  └─→ reels/db/            (SQLite 저장)
       └─ TemplateRepository
```

### 2.2 Ingest (수집 및 정규화)

**경로**: `reels/ingest/`

#### VideoDownloader
- **역할**: yt-dlp를 사용하여 YouTube 또는 로컬 파일 다운로드
- **출력**: 원본 비디오 파일

#### VideoNormalizer
- **역할**: FFmpeg를 사용하여 비디오 정규화
- **설정**:
  - 전략: `preserve_aspect` (종횡비 유지)
  - 최대 짧은 변: 1080px
  - 목표 FPS: 30
  - 비디오 코덱: libx264
  - 오디오 코덱: aac
  - 샘플레이트: 16000 Hz

#### VideoProber
- **역할**: ffprobe를 사용하여 메타데이터 추출
- **출력**: VideoMetadata (해상도, FPS, 지속시간, 코덱 등)

### 2.3 Segmentation (샷 분할)

**경로**: `reels/segmentation/`

#### ShotDetector
- **감지기**: PySceneDetect (상장 변화 감지)
- **설정**:
  - 임계값: 27.0
  - 최소 샷 지속시간: 0.25초
  - 병합 유사성 임계값: 0.92

#### Postprocessing
- 유사한 샷 병합
- 각 샷에서 키프레임 추출 (3개 키프레임/샷)
- Shot 객체 생성 (Pydantic v2)

### 2.4 Analysis (5개 분석기)

**경로**: `reels/analysis/`

모든 분석기는 `Analyzer` Protocol을 구현합니다 (타입 안정적인 프로토콜 패턴):

```python
class Analyzer(Protocol[T]):
    @property
    def name(self) -> str: ...

    def analyze_shot(self, shot: Shot, context: AnalysisContext) -> T: ...
    def analyze_batch(self, shots: list[Shot], context: AnalysisContext) -> list[T]: ...
    def cleanup(self) -> None: ...
```

#### PlaceAnalyzer
- **모델**: OpenAI CLIP ViT-Base-32
- **역할**: 키프레임 이미지를 CLIP 임베딩으로 변환 → 택시노미와 매칭
- **택시노미**: `data/taxonomy.json` (침실, 욕실, 로비, 외관 등)
- **설정**:
  - 배치 크기: 16
  - 자동 발견 임계값: 0.15
  - 레퍼런스 매칭 임계값: 0.75
- **출력**: PlaceAnalysis (place_tags, scores)

#### CameraAnalyzer
- **방법**: Farneback optical flow
- **역할**: 샷 내 카메라 움직임 분류
- **카테고리**: static, pan_left, pan_right, tilt_up, tilt_down, push_in, pull_out, handheld, gimbal_smooth
- **샘플 간격**: 2 프레임
- **모션 임계값**: 2.0
- **출력**: CameraAnalysis (camera_type, confidence)

#### SubtitleAnalyzer
- **모델**: EasyOCR (다국어)
- **역할**: 비디오에서 텍스트 감지 및 인식
- **출력**: SubtitleAnalysis (detected_text, bboxes, timing)

#### SpeechAnalyzer
- **모델**: Faster-Whisper (Whisper 최적화)
- **역할**: 오디오에서 음성 인식
- **출력**: SpeechAnalysis (transcription, confidence, language)

#### RhythmAnalyzer
- **라이브러리**: librosa
- **역할**: 비트, 템포, 음악 에너지 분석
- **출력**: RhythmAnalysis (tempo, beat_times, energy_profile)

### 2.5 Synthesis (조합)

**경로**: `reels/synthesis/`

#### TemplateAssembler
- **역할**: 5개 분석기의 결과를 하나의 Template JSON으로 조합
- **스키마 검증**: SchemaValidator로 Pydantic 모델 준수 확인
- **출력**: Template 객체
  ```python
  class Template(BaseModel):
      template_id: str
      source_video: str
      duration_sec: float
      shots: list[Shot]
      metadata: dict
      created_at: datetime
  ```

### 2.6 Storage (저장)

**경로**: `reels/storage.py`

#### TemplateArchiver
- **역할**: Template을 파일 시스템에 저장 (JSON + 관련 자산)
- **구조**: `{work_dir}/templates/{template_id}/`

### 2.7 Database (SQLite)

**경로**: `reels/db/repository.py`

#### TemplateRepository
- **DB**: SQLite
- **테이블**: templates, shots, analyses
- **조회 메서드**:
  - `search_composite(filters)`: 여러 필터로 템플릿 검색
  - `json_each` 쿼리로 CLIP 점수 범위 필터링
  - 장소, 카메라, 템포 기반 검색

### 2.8 Pipeline State (재개 기능)

**경로**: `reels/pipeline.py`

#### PipelineState
- **역할**: 파이프라인 실행 상태를 JSON으로 추적
- **단계**: ingest → segmentation → analysis → synthesis
- **재개 기능**:
  - `--resume` 플래그로 마지막 완료 단계 이후부터 재개
  - `should_skip(stage)`: 이미 완료한 단계 건너뛰기
  - `get_stage_data(stage)`: 단계별 캐시된 데이터 복구

### 2.9 CLI 명령어

```bash
# 전체 파이프라인 (영상 분석 + DB 저장)
reels analyze <video_path> --save-db

# 수집만 (다운로드 + 정규화)
reels ingest <video_path>

# 분할만 (샷 감지 + 키프레임)
reels segment <video_path>

# 분석만 (5개 분석기 실행)
reels analyze-shots <video_path>

# 템플릿 DB 조회
reels db list
reels db search --place bedroom
reels db export <template_id> out.json
```

---

## 3. 프로덕션 파이프라인 (Production Pipeline)

### 3.1 전체 흐름

```
reels produce <images> --name "숙소명" [--target couple] [--no-web] [--team] [--i2v]
  │
  ├─→ reels/production/feature_extractor.py
  │    └─ VLM 백엔드 (CLIP 또는 Claude Vision)
  │       특징 추출 + 캐시 + 병렬화 (Semaphore)
  │
  ├─→ reels/production/claim_gate.py
  │    └─ 신뢰도 → ClaimLevel (CONFIRMED/PROBABLE/SUGGESTIVE)
  │
  ├─→ reels/production/web_verifier.py (선택적)
  │    └─ 웹 검증 (신뢰도 낮은 특징)
  │
  ├─→ reels/production/template_matcher.py
  │    └─ DB에서 최적 템플릿 선택 (composite scoring)
  │
  ├─→ reels/production/copy_writer.py
  │    └─ 마케팅 카피 생성 (Hook Line + Caption)
  │
  ├─→ reels/production/storyboard_builder.py
  │    └─ 숏 어셈블리 (Role 할당, 타이밍, 자산 매핑)
  │
  ├─→ reels/production/render_spec.py
  │    └─ Remotion 렌더 스펙 + SRT 자막 생성
  │
  └─→ reels/production/i2v/ (선택적, --i2v)
       └─ 이미지→영상 변환 (Replicate API)
```

### 3.2 Feature Extraction (특징 추출)

**경로**: `reels/production/feature_extractor.py`

#### FeatureExtractor
- **VLM 백엔드**:
  - `clip` (기본값): CLIPVisionBackend (로컬, 빠름)
  - `claude`: ClaudeVisionBackend (API, 고정확)
- **병렬화**:
  - `asyncio.gather`로 모든 이미지 동시 분석
  - `Semaphore(max_concurrent=5)` 속도 제한
  - 지수 백오프 재시도 (최대 3회)
- **캐시**:
  - ResponseCache (SHA-256 기반)
  - 이미지 바이트의 해시 → VLM 응답 재사용
  - 캐시 디렉토리: `.cache/vlm`
- **특징 병합**:
  - 유사 특징 통합 (임계값: 0.8)
  - 카테고리별 중요도 가중치:
    - 어메니티 (AMENITY): 1.0 (노천탕, 사우나 등)
    - 뷰 (VIEW): 0.9 (오션뷰, 산뷰)
    - 다이닝 (DINING): 0.8 (조식, 바베큐)
    - 액티비티 (ACTIVITY): 0.7 (키즈, 반려견)
    - 장면 (SCENE): 0.5 (침실, 욕실)
- **출력**: Feature 객체 리스트
  ```python
  class Feature(BaseModel):
      tag: str                              # 한국어 (예: "노천탕")
      tag_en: str                           # 영문 (예: "outdoor_bath")
      confidence: float                     # 0.0 ~ 1.0
      evidence_images: list[str]            # 파일명
      description: str                      # VLM 추론 설명
      category: FeatureCategory             # SCENE/AMENITY/VIEW/DINING/ACTIVITY
  ```

### 3.3 Claim Gate (신뢰도 평가)

**경로**: `reels/production/claim_gate.py`

#### ClaimGate
- **역할**: 특징 신뢰도 → 카피 톤 제어
- **3-tier 시스템**:

  | ClaimLevel | 조건 | 허용 카피 톤 | 예시 |
  |---|---|---|---|
  | **CONFIRMED** | ≥ 0.75 | 단정적 (단정) | "노천탕 있는 프라이빗 힐링" |
  | **PROBABLE** | 0.50~0.75 | 완곡 표현 (암시) | "노천탕 느낌의 야외 욕실" |
  | **SUGGESTIVE** | < 0.50 | 분위기만 (분위기) | "자연스러운 욕실 공간" |

- **팩트 단어 제한**:
  - CONFIRMED 레벨에서만 허용: "무료", "제공", "무제한", "포함", "운영"
  - 예: "무료 WiFi" → CONFIRMED만 허용

- **신뢰도 보정**:
  - 증거 이미지 2장 이상 → +0.10 보너스
  - 상한: 1.0

- **출력**: VerifiedFeature 객체 리스트
  ```python
  class VerifiedFeature(Feature):
      claim_level: ClaimLevel              # CONFIRMED/PROBABLE/SUGGESTIVE
      web_evidence: list[WebEvidence]      # 웹 검증 결과
      copy_tone: str                       # "단정" / "암시" / "분위기"
  ```

### 3.4 Web Verification (웹 검증)

**경로**: `reels/production/web_verifier.py`

#### WebVerifier
- **역할**: 낮은 신뢰도 특징을 웹 검색으로 검증
- **신뢰 도메인**: 공식 숙박 사이트 (booking.com, airbnb.com 등)
- **작동 조건**:
  - 특징 신뢰도 < 0.75 AND 숙소 이름 제공됨
  - 웹 검증 활성화 플래그 설정됨
- **출력**: WebEvidence (URL, 스니펫, 검증 신뢰도)
- **ClaimGate 재평가**: 웹 검증 후 신뢰도 업데이트

### 3.5 Template Matching (템플릿 매칭)

**경로**: `reels/production/template_matcher.py`

#### TemplateMatcher
- **역할**: 추출된 특징과 일치하는 최적의 템플릿 DB에서 선택
- **Composite Scoring** (가중 점수):

  | 요소 | 가중치 | 설명 |
  |---|---|---|
  | Place Overlap | 0.30 | 특징 장소 매칭 |
  | Duration Fit | 0.20 | 숏 개수 맞춤 |
  | Camera Variety | 0.15 | 다양한 카메라 움직임 |
  | Rhythm Match | 0.15 | 템포 유사성 |
  | Shot Count Fit | 0.20 | 특징 개수와 숏 개수 일치 |

- **출력**: MatchResult
  ```python
  class MatchResult(BaseModel):
      template_id: str
      score: float                         # 0.0 ~ 1.0
      shot_count: int                      # 템플릿의 샷 개수
  ```

### 3.6 Copy Writer (카피 작성)

**경로**: `reels/production/copy_writer.py`

#### CopyWriter
- **역할**: 검증된 특징으로 마케팅 카피 생성
- **카피 구조**:
  - **Hook Line** (7~12자): 주의 끌기
    - 예: "노천탕의 설렘"
  - **Caption Lines** (최대 14자 × 2줄): 상세 설명
    - 예:
      ```
      준비된 프라이빗 야외욕실
      하늘을 보며 피부 진정 테라피
      ```

- **ClaimLevel별 처리**:
  - CONFIRMED: 모든 표현 허용 (단정적 표현 우선)
  - PROBABLE: 팩트 단어 제외, 완곡 표현
  - SUGGESTIVE: 분위기 중심 표현만

- **출력**: CopyResult
  ```python
  class CopyLine(BaseModel):
      text: str
      feature_tag_en: str | None          # 연결된 특징
      copy_tone: str
  ```

### 3.7 Storyboard Builder (스토리보드 구성)

**경로**: `reels/production/storyboard_builder.py`

#### StoryboardBuilder
- **역할**: 숏을 Role별로 할당하고 타이밍 설정
- **Shot Role** (4가지):

  | Role | 용도 | 타이밍 | 특징 |
  |---|---|---|---|
  | **HOOK** | 오프닝 (주의 끌기) | 0~3초 | 가장 임팩트 강한 특징 |
  | **FEATURE** | 핵심 특징 노출 | 중간 | 주요 특징들 집중 조명 |
  | **SUPPORT** | 분위기 강화 | 후반 | 배경/분위기 강조 |
  | **CTA** | 클로징 (행동 유도) | 끝 | "예약하기" 등 CTA 카피 |

- **할당 로직**:
  1. CONFIRMED 특징들 → FEATURE 우선
  2. 1순위 특징 → HOOK
  3. 후속 특징들 → SUPPORT
  4. 마지막 → CTA

- **출력**: Storyboard
  ```python
  class ShotAssignment(BaseModel):
      shot_index: int
      role: ShotRole
      feature_tag_en: str | None
      copy_line: CopyLine | None
      timing: tuple[float, float]          # (start_sec, end_sec)

  class Storyboard(BaseModel):
      shots: list[ShotAssignment]
      total_duration_sec: float
  ```

### 3.8 Render Spec Generation (렌더 스펙)

**경로**: `reels/production/render_spec.py`

#### RenderSpecGenerator
- **역할**: Remotion 렌더 스펙 + SRT 자막 생성
- **출력 형식**:
  - `render_spec.json`: Remotion 구성 (레이아웃, 타이밍, 자산 경로)
  - `subtitles.srt`: 시간 동기화된 자막 (SRT 형식)

- **자막 내용**:
  ```
  00:00:00,000 --> 00:00:03,000
  노천탕의 설렘

  00:00:03,000 --> 00:00:08,000
  준비된 프라이빗 야외욕실
  하늘을 보며 피부 진정 테라피
  ```

- **Remotion 스펙**:
  ```json
  {
    "composition": "AccommodationShorts",
    "fps": 30,
    "duration_sec": 15,
    "shots": [
      {
        "index": 0,
        "role": "hook",
        "video_path": "...",
        "duration_sec": 3,
        "caption": "노천탕의 설렘",
        "caption_tone": "단정"
      }
    ]
  }
  ```

### 3.9 I2V (Image-to-Video)

**경로**: `reels/production/i2v/`

#### Motion Selector & Replicate Integration
- **역할**: 정적 이미지를 동영상으로 변환 (선택적)
- **활성화**: `--i2v` 플래그
- **API**: Replicate (Runway, Stability AI 등)
- **용도**: 숙소 이미지를 패닝/줌 효과와 함께 영상화
- **설정**:
  - 모션 프리셋: subtle, medium, dramatic
  - 지속시간: 설정 가능

### 3.10 Creative Team Mode (크리에이티브 팀)

**경로**: `reels/production/creative_team/`

#### 선택적 AI 크리에이티브 팀 (--team 플래그)

- **CreativePlanner**
  - 입력: VerifiedFeature 리스트 + 타겟 오디언스
  - 출력: CreativeBrief (hero_features, narrative_structure, concept_keywords)

- **ProducerDirector**
  - 입력: CreativeBrief
  - 출력: ProductionGuidelines (shot_sequence, visual_style, music_mood)

- **CreativeWriter**
  - 입력: ProductionGuidelines + Features
  - 출력: EnhancedCopy (refined hook line, captions)

- **QAReviewer**
  - 입력: 모든 이전 출력
  - 검증: 일관성, 정확성, 마케팅 효과
  - 최대 2회 수정 루프

**CLI**: `reels produce *.jpg --team`

### 3.11 CLI 명령어

```bash
# 기본 프로덕션 (CLIP 백엔드, 웹 검증 활성화)
reels produce img1.jpg img2.jpg --name "스테이 패스포트" --target couple --output output/

# 웹 검증 스킵
reels produce *.jpg --no-web

# Claude Vision 백엔드 사용
reels produce *.jpg --backend claude

# 크리에이티브 팀 모드 (AI 크리에이티브 평가)
reels produce *.jpg --team

# 이미지→영상 변환 포함
reels produce *.jpg --i2v

# 모든 옵션 조합
reels produce *.jpg --name "스테이 패스포트" --target family --backend claude --team --i2v --output output/
```

---

## 4. 공유 컴포넌트

### 4.1 Models (데이터 모델)

**경로**: `reels/models/` + `reels/production/models.py`

#### 분석 파이프라인 모델 (`reels/models/`)
```python
# Shot.py
class Shot(BaseModel):
    shot_id: str
    start_frame: int
    end_frame: int
    start_sec: float
    end_sec: float
    keyframes: list[Path]                  # 키프레임 이미지 경로

# Template.py
class Template(BaseModel):
    template_id: str
    source_video: str
    duration_sec: float
    shots: list[Shot]
    metadata: dict[str, Any]
    created_at: datetime

# VideoMetadata.py
class VideoMetadata(BaseModel):
    width: int
    height: int
    fps: float
    duration_sec: float
    video_codec: str
    audio_codec: str

# IngestResult.py
class IngestResult(BaseModel):
    video_path: Path
    audio_path: Path | None
    metadata: VideoMetadata
```

#### 프로덕션 파이프라인 모델 (`reels/production/models.py`)
```python
# 입력
class AccommodationInput(BaseModel):
    name: str | None                       # 숙소명
    region: str | None                     # 지역
    target_audience: TargetAudience        # couple/family/solo/friends
    price_range: PriceRange | None         # budget/mid/luxury
    images: list[Path]                     # 필수
    video_clips: list[Path] = []
    custom_instructions: str | None

# 특징
class Feature(BaseModel):
    tag: str                               # 한국어 태그
    tag_en: str                            # 영문 태그
    confidence: float                      # 0.0 ~ 1.0
    evidence_images: list[str]
    description: str
    category: FeatureCategory              # SCENE/AMENITY/VIEW/DINING/ACTIVITY

class VerifiedFeature(Feature):
    claim_level: ClaimLevel                # CONFIRMED/PROBABLE/SUGGESTIVE
    web_evidence: list[WebEvidence] = []
    copy_tone: str                         # "단정" / "암시" / "분위기"

# 결과
class ProductionResult(BaseModel):
    project_id: str
    status: str                            # "success" / "partial" / "failed"
    features: list[VerifiedFeature]
    storyboard: Storyboard | None
    render_spec: dict[str, Any] | None
    errors: list[str] = []
    warnings: list[str] = []
```

### 4.2 Database (SQLite)

**경로**: `reels/db/repository.py`

#### TemplateRepository
- **DB 경로**: `{work_dir}/templates.db`
- **테이블**:
  - `templates`: template_id, source_video, duration_sec, created_at
  - `shots`: shot_id, template_id, start_sec, end_sec
  - `analyses`: shot_id, analyzer_name, result_json

- **조회 메서드**:
  ```python
  def search_composite(
      self,
      places: list[str] | None = None,
      cameras: list[str] | None = None,
      tempo_range: tuple[float, float] | None = None,
      duration_range: tuple[float, float] | None = None
  ) -> list[Template]:
      """여러 필터로 템플릿 검색 (json_each 사용)"""

  def list_all() -> list[Template]:
      """모든 템플릿 조회"""

  def get_by_id(template_id: str) -> Template:
      """ID로 템플릿 조회"""
  ```

### 4.3 Configuration (설정)

**경로**: `reels/config.py` + `config/default.yaml`

#### 설정 우선순위
1. `config/default.yaml` (기본값)
2. `REELS_*` 환경변수 오버라이드
3. 런타임 config 인자

#### 주요 섹션

```yaml
# 파이프라인 전체
pipeline:
  work_dir: "./work"
  cache_enabled: true
  log_level: "INFO"
  max_video_duration_sec: 300
  cleanup_keyframes_after_synthesis: false

# Ingest (수집)
ingest:
  normalize:
    strategy: "preserve_aspect"
    max_short_side: 1080
    target_fps: 30
    audio_sample_rate: 16000
    video_codec: "libx264"
    audio_codec: "aac"

# Segmentation (분할)
segmentation:
  detector: "pyscenedetect"
  threshold: 27.0
  min_shot_duration_sec: 0.25
  merge_similar_threshold: 0.92

# Analysis (분석)
analysis:
  parallel: false

  place:
    model: "openai/clip-vit-base-patch32"
    keyframes_per_shot: 3
    taxonomy_path: "data/taxonomy.json"
    auto_discover: true
    discovery_threshold: 0.15
    ref_match_threshold: 0.75
    batch_size: 16

  camera:
    optical_flow_method: "farneback"
    sample_interval_frames: 2
    motion_threshold: 2.0

  speech:
    model: "base"                          # tiny/base/small/medium/large
    language: "ko"

  rhythm:
    hop_length: 512
    n_fft: 2048

# Production (프로덕션)
production:
  feature_extraction:
    backend: "clip"                        # clip or claude
    max_concurrent: 5
    cache_dir: ".cache/vlm"
    merge_threshold: 0.8

  claim_gate:
    confirmed_threshold: 0.75
    probable_threshold: 0.50
    multi_evidence_bonus: 0.10

  template_matching:
    place_overlap_weight: 0.30
    duration_fit_weight: 0.20
    camera_variety_weight: 0.15
    rhythm_match_weight: 0.15
    shot_count_fit_weight: 0.20

  copy_writer:
    hook_min_chars: 7
    hook_max_chars: 12
    caption_max_chars: 14
    caption_max_lines: 2

  i2v:
    model: "runway"                        # runway or stability
    enabled: false
    motion_preset: "medium"

  creative_team:
    enabled: false
    llm_model: "claude-3-5-sonnet"
    max_revisions: 2
    planner:
      temperature: 0.8
      max_tokens: 1024
    producer_director:
      temperature: 0.8
      max_tokens: 1024
    writer:
      temperature: 0.7
      max_tokens: 512
    qa_reviewer:
      temperature: 0.6
      max_tokens: 1024

# Backend (API 키 등)
backends:
  claude:
    api_key: null                          # ANTHROPIC_API_KEY 환경변수에서 읽음
  openai:
    api_key: null                          # OPENAI_API_KEY 환경변수에서 읽음
  replicate:
    api_key: null                          # REPLICATE_API_KEY 환경변수에서 읽음
```

#### 설정 로드 패턴
```python
from reels.config import load_config

config = load_config()  # default.yaml + 환경변수 로드

# 섹션 접근
production_cfg = (config or {}).get("production", {})
feature_cfg = production_cfg.get("feature_extraction", {})
backend_name = feature_cfg.get("backend", "clip")
```

### 4.4 Backends (VLM/LLM)

**경로**: `reels/production/backends/`

#### VLMBackend Protocol
```python
class VLMBackend(Protocol):
    """Vision-Language Model 인터페이스 (swappable)"""

    async def analyze_image(
        self, image: Path, prompt: str
    ) -> list[dict[str, Any]]: ...

    async def analyze_batch(
        self, images: list[Path], prompt: str
    ) -> list[list[dict[str, Any]]]: ...

    def cleanup(self) -> None: ...
```

#### CLIPVisionBackend
- **모델**: OpenAI CLIP ViT-Base-32
- **장점**: 로컬 실행 (API 없음), 빠름, 저비용
- **단점**: 영문 태그만 지원

#### ClaudeVisionBackend
- **모델**: Claude 3.5 Sonnet Vision
- **장점**: 한국어 이해, 고정확, 복잡한 추론
- **단점**: API 호출 (비용, 지연)
- **주의**: ANTHROPIC_API_KEY 필요

### 4.5 Cache (응답 캐시)

**경로**: `reels/production/cache.py`

#### ResponseCache
- **캐시 키**: SHA-256(이미지 바이트)
- **저장 위치**: `.cache/vlm/`
- **형식**: JSON 파일
- **용도**: 동일 이미지 재분석 스킵

```python
cache = ResponseCache(".cache/vlm")
cache_key = cache.get_key(image_path)     # SHA-256 해시

# 캐시 조회
if cache.has(cache_key):
    features = cache.get(cache_key)       # 저장된 특징
else:
    features = await vlm.analyze_image(image_path)
    cache.put(cache_key, features)        # 저장
```

---

## 5. 설계 패턴

### 5.1 Protocol 패턴 (교체 가능한 구현)

**분석**: Analyzer Protocol
```python
class Analyzer(Protocol[T]):
    @property
    def name(self) -> str: ...
    def analyze_shot(self, shot: Shot, context: AnalysisContext) -> T: ...
    def analyze_batch(self, shots: list[Shot], context: AnalysisContext) -> list[T]: ...
    def cleanup(self) -> None: ...
```

**프로덕션**: VLMBackend Protocol
```python
class VLMBackend(Protocol):
    async def analyze_image(self, image: Path, prompt: str) -> list[dict]: ...
    async def analyze_batch(self, images: list[Path], prompt: str) -> list[list[dict]]: ...
    def cleanup(self) -> None: ...
```

이 패턴으로 CLIP ↔ Claude, PySceneDetect ↔ 다른 감지기 간 교체가 간단합니다.

### 5.2 Config Propagation (설정 전파)

각 모듈이 YAML 설정의 특정 섹션을 읽습니다:
```python
# ClaimGate
cfg = (config or {}).get("production", {}).get("claim_gate", {})
self.confirmed_threshold = cfg.get("confirmed_threshold", 0.75)

# FeatureExtractor
cfg = (config or {}).get("production", {}).get("feature_extraction", {})
self.backend_name = cfg.get("backend", "clip")
```

### 5.3 Async + Semaphore (병렬화 제어)

FeatureExtractor는 여러 이미지를 동시에 분석하되, 속도를 제어합니다:
```python
async def extract(self, images: list[Path]) -> list[Feature]:
    tasks = [self._analyze_with_limit(img) for img in images]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    # Semaphore가 최대 5개 동시 API 호출 유지
```

### 5.4 ClaimGate 3-tier System (신뢰도 기반 카피 제어)

```
confidence >= 0.75
    ↓
CONFIRMED (단정적 카피 허용)
"무료 WiFi 제공" ✓

0.50 <= confidence < 0.75
    ↓
PROBABLE (완곡 표현)
"WiFi 이용 가능" ✓

confidence < 0.50
    ↓
SUGGESTIVE (분위기만)
"디지털 편의시설" ✓ (구체적 팩트 불가)
```

### 5.5 Content-Hash Cache (응답 캐시)

VLM 응답을 이미지 해시로 캐시하여 중복 분석을 방지합니다:
```python
key = hashlib.sha256(image_bytes).hexdigest()
if cache.has(key):
    features = cache.get(key)  # 파일에서 로드
else:
    features = await vlm.analyze_image(image)
    cache.put(key, features)   # 파일에 저장
```

### 5.6 Composite Scoring (가중 점수)

TemplateMatcher는 5개 요소를 가중 조합하여 최적 템플릿을 선택합니다:
```
score = (
    0.30 * place_overlap_score +
    0.20 * duration_fit_score +
    0.15 * camera_variety_score +
    0.15 * rhythm_match_score +
    0.20 * shot_count_fit_score
)
```

---

## 6. 기술 스택

| 영역 | 라이브러리 |
|---|---|
| **언어** | Python 3.12 |
| **데이터 검증** | Pydantic v2 |
| **CLI** | Click |
| **로깅** | logging (Python 표준) |
| **UI** | Rich |
| **비디오** | OpenCV, FFmpeg (외부), yt-dlp |
| **분석** | CLIP, faster-whisper, EasyOCR, librosa |
| **샷 감지** | PySceneDetect |
| **LLM/Vision** | Anthropic SDK (Claude Vision) |
| **데이터베이스** | SQLite |
| **웹 검증** | requests, BeautifulSoup |

---

## 7. 개발 가이드

### 7.1 환경 설정

```bash
# 가상환경 생성 및 활성화
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows

# 의존성 설치
pip install -e ".[dev]"    # 개발 + 테스트 의존성
pip install -e ".[i2v]"    # I2V 옵션 의존성 포함 (선택사항)
```

### 7.2 테스트

```bash
# 전체 테스트 (426개 pass 목표)
python -m pytest tests/ -q

# 프로덕션 파이프라인만
python -m pytest tests/unit/production/ -q

# 특정 테스트 파일
python -m pytest tests/unit/test_cli_produce.py -q

# Slow 테스트 포함
python -m pytest tests/ --run-slow

# 커버리지 확인
python -m pytest tests/ --cov=reels --cov-report=html
```

**주의**: pytest-asyncio는 미설치. 모든 async 테스트는 `asyncio.run()` 래퍼 사용:
```python
def test_async_feature_extraction():
    async def _run():
        features = await extractor.extract([img1, img2])
        assert len(features) > 0
    asyncio.run(_run())
```

### 7.3 린트 및 타입 검사

```bash
# Linting (Ruff)
ruff check reels/

# 타입 검사 (mypy)
mypy reels/ --ignore-missing-imports
```

### 7.4 새 분석기 추가

1. `reels/analysis/` 에 새 모듈 생성
2. `Analyzer` Protocol 구현:
   ```python
   class MyAnalyzer:
       @property
       def name(self) -> str:
           return "my_analyzer"

       def analyze_shot(self, shot: Shot, context: AnalysisContext) -> MyAnalysisResult:
           # 구현
           return MyAnalysisResult(...)
   ```
3. `reels/analysis/__init__.py`에 export
4. `reels/pipeline.py`에 등록
5. `reels/synthesis/assembler.py`에서 결과 통합

### 7.5 새 VLM 백엔드 추가

1. `reels/production/backends/`에 새 모듈 생성
2. `VLMBackend` Protocol 구현:
   ```python
   class MyVLMBackend:
       async def analyze_image(self, image: Path, prompt: str) -> list[dict]:
           # 구현
           return features
   ```
3. `reels/production/feature_extractor.py`의 `_create_backend()` 업데이트
4. `config/default.yaml`에 backend 옵션 추가

---

## 8. 트러블슈팅

| 문제 | 원인 | 해결 |
|---|---|---|
| `ModuleNotFoundError: No module named 'reels'` | 가상환경 비활성화 또는 설치 실패 | `source .venv/bin/activate` 후 `pip install -e .` |
| `FFmpeg not found` | FFmpeg 미설치 | `apt-get install ffmpeg` (Linux) 또는 `brew install ffmpeg` (Mac) |
| `CLIP model download timeout` | 네트워크 느림 | 수동 다운로드: `python -m torch.hub` |
| `API rate limit exceeded` | Claude API 호출 초과 | `--no-web` 플래그로 웹 검증 비활성화, 또는 CLIP 백엔드 사용 |
| `Database locked` | 동시 접근 | SQLite WAL 모드 확인 (자동 설정됨) |
| `Out of memory` | 대용량 비디오/이미지 | `max_video_duration_sec` 제한, 또는 배치 크기 감소 |

---

## 9. 디렉토리 구조

```
reels-project/
├── reels/
│   ├── __init__.py
│   ├── __main__.py              # CLI 진입점
│   ├── config.py                # pydantic-settings 설정
│   ├── exceptions.py
│   ├── pipeline.py              # 분석 파이프라인 + PipelineState
│   ├── storage.py               # TemplateArchiver
│   │
│   ├── models/                  # Pydantic v2 공유 모델
│   │   ├── shot.py
│   │   ├── template.py
│   │   ├── metadata.py
│   │   ├── analysis.py
│   │   └── __init__.py
│   │
│   ├── ingest/                  # 수집 및 정규화
│   │   ├── downloader.py
│   │   ├── normalizer.py
│   │   ├── probe.py
│   │   └── __init__.py
│   │
│   ├── segmentation/            # 샷 분할
│   │   ├── detector.py          # PySceneDetect
│   │   ├── postprocess.py
│   │   └── __init__.py
│   │
│   ├── analysis/                # 5개 분석기
│   │   ├── base.py              # Analyzer Protocol + AnalysisContext
│   │   ├── runner.py            # AnalysisRunner
│   │   ├── place.py             # PlaceAnalyzer (CLIP)
│   │   ├── camera.py            # CameraAnalyzer (optical flow)
│   │   ├── subtitle.py          # SubtitleAnalyzer (EasyOCR)
│   │   ├── speech.py            # SpeechAnalyzer (faster-whisper)
│   │   ├── rhythm.py            # RhythmAnalyzer (librosa)
│   │   ├── taxonomy.py
│   │   └── __init__.py
│   │
│   ├── synthesis/               # 템플릿 조합
│   │   ├── assembler.py         # TemplateAssembler
│   │   ├── schema_validator.py
│   │   └── __init__.py
│   │
│   ├── db/                      # SQLite
│   │   ├── repository.py        # TemplateRepository
│   │   └── __init__.py
│   │
│   ├── production/              # 프로덕션 파이프라인
│   │   ├── agent.py             # ProductionAgent (오케스트레이터)
│   │   ├── models.py            # 20+ production 전용 모델
│   │   ├── feature_extractor.py # FeatureExtractor + 병렬화
│   │   ├── claim_gate.py        # ClaimGate (3-tier 신뢰도)
│   │   ├── web_verifier.py      # WebVerifier
│   │   ├── template_matcher.py  # TemplateMatcher (composite scoring)
│   │   ├── copy_writer.py       # CopyWriter
│   │   ├── storyboard_builder.py # StoryboardBuilder
│   │   ├── render_spec.py       # RenderSpecGenerator
│   │   ├── cache.py             # ResponseCache (SHA-256 기반)
│   │   │
│   │   ├── backends/            # VLM 백엔드 (Protocol)
│   │   │   ├── base.py          # VLMBackend, LLMBackend Protocol
│   │   │   ├── clip_vision.py   # CLIPVisionBackend (로컬)
│   │   │   └── claude_vision.py # ClaudeVisionBackend (API)
│   │   │
│   │   ├── i2v/                 # 이미지→영상 변환 (선택사항)
│   │   │   ├── motion_selector.py
│   │   │   └── __init__.py
│   │   │
│   │   ├── creative_team/       # AI 크리에이티브 팀 (선택사항)
│   │   │   ├── planner.py       # CreativePlanner
│   │   │   ├── producer.py      # ProducerDirector
│   │   │   ├── writer.py        # CreativeWriter
│   │   │   ├── qa_reviewer.py   # QAReviewer
│   │   │   ├── models.py
│   │   │   ├── llm.py
│   │   │   └── __init__.py
│   │   │
│   │   ├── omc_helpers/         # Oh-My-Claude 통합 (선택사항)
│   │   │   ├── phase0_runner.py
│   │   │   ├── assemble.py
│   │   │   ├── render_spec_runner.py
│   │   │   ├── validate_output.py
│   │   │   ├── code_checks.py
│   │   │   └── __init__.py
│   │   │
│   │   └── __init__.py
│   │
│   ├── utils/                   # 유틸리티
│   │   ├── video.py
│   │   ├── audio.py
│   │   ├── image.py
│   │   ├── paths.py
│   │   └── __init__.py
│   │
│   └── __pycache__/
│
├── tests/
│   ├── unit/
│   │   ├── test_cli_produce.py
│   │   ├── production/
│   │   ├── analysis/
│   │   └── ...
│   ├── integration/
│   └── conftest.py
│
├── config/
│   └── default.yaml             # 기본 설정 (YAML)
│
├── data/
│   └── taxonomy.json            # 장소 택시노미
│
├── CLAUDE.md                    # 개발자 가이드 (this file)
├── PIPELINE.md                  # 파이프라인 문서 (Korean)
├── README.md
├── pyproject.toml               # Python 프로젝트 설정
├── setup.py
├── ruff.toml                    # Ruff 린트 설정
├── pyright.json                 # Pyright 타입 검사 설정
└── .gitignore
```

---

## 10. 다음 단계

- 새 분석기 추가: `reels/analysis/` 구조 학습
- 커스텀 VLM 통합: `reels/production/backends/` 패턴 참조
- 웹 검증 확장: `reels/production/web_verifier.py` 신뢰 도메인 추가
- 크리에이티브 팀 커스터마이징: `reels/production/creative_team/` 프롬프트 조정
- I2V 파이프라인: `reels/production/i2v/` 모션 프리셋 확장

---

**문서 버전**: 1.0
**업데이트**: 2026-02-23
**유지보수**: 한국어 기준 (관련자: 개발팀)
