<!-- Copilot instructions for Research-ai-whitebalance -->
# 코파일럿/에이전트 사용 지침 (간단 요약)

이 레포는 이미지 기반 검출기(detectors)와 화면 출력(handler)을 결합한 연구용 코드베이스입니다. 에이전트는 아래 핵심 패턴과 파일을 먼저 확인하면 빠르게 생산적으로 작업할 수 있습니다.

- **아키텍처(큰 그림)**
  - `libs/detectors/` : 검출기 래퍼(프로젝트 인터페이스). 예: `libs/detectors/dino.py`, `libs/detectors/yoloworld.py`.
  - `libs/detectors/GroundingDINO/` : GroundingDINO 오픈소스 코드(벤더 포함). 독립적 설치/빌드 필요.
  - `handler/` : 화면 출력 혹은 후처리 핸들러. 예: `handler/screen_handler.py`.
  - `config/` : 실행 설정 YAML (`dino.yaml`, `yoloworld.yaml`) — 이 파일들로 어떤 detector/handler를 쓸지 결정.
  - `assets/weights/` : 모델 가중치 보관.

- **핵심 실행·개발 워크플로우**
  - GroundingDINO는 `libs/detectors/GroundingDINO/README.md`의 설치 절차(및 `requirements.txt`)를 따릅니다. (예: `pip install -e .` inside that folder)
  - 데모(로컬) 실행 예:
    ```bash
    CUDA_VISIBLE_DEVICES=0 python libs/detectors/GroundingDINO/demo/inference_on_a_image.py \
      -c groundingdino/config/GroundingDINO_SwinT_OGC.py \
      -p assets/weights/groundingdino_swint_ogc.pth \
      -i path/to/image.jpg -o outdir -t "chair" [--cpu-only]
    ```
  - CUDA가 없거나 빌드 문제(`NameError: name '_C' is not defined`) 발생 시 GroundingDINO를 다시 설치하거나 `--cpu-only`를 사용하고 `CUDA_HOME` 환경변수를 확인하세요.

- **프로젝트 규약·중요 관찰사항(꼭 읽어야 할 것들)**
  - Detector 인터페이스: 각 detector 클래스는 `detect(img)` 메서드를 제공하고, 일반적으로 `[[x1,y1,x2,y2, score, label], ...]` 형식의 리스트를 반환하도록 설계되어 있습니다. **하지만 좌표 단위가 일관되지 않습니다!**
    - `libs/detectors/dino.py`(GroundingDINO 래퍼): 내부에서 정규화 좌표를 원래 이미지 픽셀 좌표(absolute pixel)로 복원하고 반환합니다.
    - `libs/detectors/yoloworld.py`: 좌표를 이미지 크기로 나눈 **정규화된 [0,1] 범위**로 반환합니다.
    - 결과를 소비하는 `handler`(예: `handler/screen_handler.py`)는 이 불일치를 고려해 입력을 변환해야 합니다.
  - 이미지 포맷: 입력 이미지는 OpenCV 관행(BGR, `numpy.ndarray`)을 따릅니다. GroundingDINO 래퍼는 내부에서 BGR→RGB 변환과 ImageNet 정규화를 수행합니다. YOLOWorld는 `ultralytics` 예측을 그대로 사용합니다.
  - 설정 파일(`config/*.yaml`)은 detector/handler 구성의 출발점입니다. 다만 `config/dino.yaml`에 오탈자(`text_threshol`, `self.with_nms`)가 있으니 실제 파라미터는 `libs/detectors/*.py`의 디폴트/키 이름을 확인해서 사용하세요.

- **통합 포인트·외부 의존성**
  - GroundingDINO (벤더 포함): 자체 `requirements.txt`, `setup.py`가 있으며, 컴파일(특히 CUDA extension) 필요 가능성 있음.
  - Ultralytics/YOLOWorld: `ultralytics` 패키지 필요, 모델 파일은 `assets/weights/`에 둠.

- **디버깅 팁(프로젝트 특화)**
  - 빌드/임포트 에러 `_C` 관련: GroundingDINO C-extensions가 제대로 설치되지 않은 경우 발생 — `pip install -e .` 재실행 및 `CUDA_HOME` 확인.
  - GPU 문제: `nvidia-smi`로 GPU 사용 가능 여부 확인, 필요시 `CUDA_VISIBLE_DEVICES`로 장치 지정.
  - 좌표 불일치 문제 발생 시: 호출 코드(핸들러)에서 detector 종류에 따라 normalize/denormalize 유틸(간단 변환 함수)을 추가해 일관성 유지.

- **주요 파일(빠르게 살펴볼 곳)**
  - `libs/detectors/dino.py` — GroundingDINO 래퍼(입력 전처리, detect 반환 형식 확인)
  - `libs/detectors/yoloworld.py` — YOLOWorld 래퍼(정규화 좌표 반환)
  - `libs/detectors/GroundingDINO/README.md` — 벤더 설치/데모/주의사항
  - `handler/screen_handler.py` — 화면 렌더링, 좌표 사용 방식을 확인해야 함
  - `config/*.yaml` — 실행 시 선택되는 구성

피드백: 이 초안에서 더 자세히 문서화할 부분(예: 핸들러가 기대하는 좌표 단위 표준화 방식, 자주 쓰는 CLI/스크립트 목록)이 있나요? 원하시면 좌표 변환 유틸 제안 코드도 추가해 드리겠습니다.
