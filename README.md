# DCC

2025 데이터 크리에이터 캠프(DCC) 대회에서 진행한 GPR(Ground Penetrating Radar) 기반 객체 탐지 프로젝트입니다.

GPR 데이터의 단면 이미지와 라벨 데이터를 YOLO 형식으로 정리하고, YOLOv11n 모델을 사용해 지하 구조물 및 이상 객체를 탐지하는 Vision 파이프라인을 실험했습니다.

## Project Overview

- 주제: GPR 이미지 기반 객체 탐지
- 모델: YOLOv11n
- 데이터 단면: xz 평면
- 평가 흐름: train/validation/test 분할 후 mAP@IoU50 기준 평가
- 정리 노션: [2025 DCC](https://www.notion.so/2025-DCC-253dc14b476d8088b0fbd96f6edfb630?source=copy_link)

## Why xz Plane

대회 정리 노션 기준으로, 학습 데이터는 xy, xz, yz 단면 중 xz 평면을 선택했습니다.

xz 단면은 수평 방향(x)과 깊이 방향(z)을 함께 보여주기 때문에 지하 구조물의 깊이 정보와 연속적인 형태 변화를 학습하기에 적합하다고 판단했습니다. 특히 GPR 데이터에서 z축은 깊이를 의미하므로, 파이프나 공동처럼 지하에 존재하는 객체를 탐지할 때 중요한 단서가 됩니다.

## Classes

`project/h100/dataset.yaml` 기준 탐지 클래스는 다음 5개입니다.

- bounding box
- bounding patch
- bounding pipe
- bounding manhole
- bounding cavity

## Repository Structure

```text
.
├── project/
│   ├── h100/
│   │   ├── dataset.yaml
│   │   └── main.ipynb
│   └── scripts/
│       ├── labels_count.py
│       ├── object_detection.py
│       └── train_test_split.py
├── [교안] 크리에이터캠프_고등부/
├── [실습파일] 크리에이터캠프_고등부/
├── bounding_*.png
└── README.md
```

## Scripts

- `labels_count.py`: JSON 라벨 파일을 탐색해 클래스 분포를 확인합니다.
- `object_detection.py`: GPR 이미지와 JSON 라벨을 매칭하고 바운딩 박스를 시각화합니다.
- `train_test_split.py`: 이미지와 라벨을 train/validation/test로 분할합니다.

## Notes

대용량 데이터셋, 압축 파일, 학습 결과물, 모델 가중치 파일은 GitHub 업로드 용량 문제를 피하기 위해 저장소에서 제외했습니다.

제외된 주요 항목은 다음과 같습니다.

- `project/GPR_dataset/`
- `project/labels_yolo/`
- `project/*.zip`
- `project/h100/runs/`
- `*.pt`
