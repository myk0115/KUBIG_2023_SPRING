
<img width="689" alt="image" src="https://user-images.githubusercontent.com/11497518/221871124-4689a296-c605-4ab2-ae02-6ed601ae10d5.png">

# 월간 데이콘 Computer Vision 이상치 탐지 알고리즘 경진대회

## DL - CV_이상치 탐지 1팀

16기 유우혁

16기 이은찬

17기 임종우

## Task 소개

사물의 종류 및 정상 샘플과 비정상(이상치) 샘플을 분류

평가 산식: Macro-F1 score(클래스별/레이블별 F1-score의 평균)

## Data Augmentation

* Crop
* Rotation
* Flip

## Model

EfficientNet-B7

#### Optimizer

* AdamP Optimizer(lr : 5e-4, momentum : 0.9, weight_decay : 0.01)

* OneCycleLR (Learning Rate Scheduler) 사용

#### Loss

* Focal loss

---

### Transfer Learning

1. Pretrained EfficientNet-B7
2. Pretrained 원본 데이터 모델
3. Final Model(Augmented Data trained)

완성본에서는 2번을 사용합니다.

---

## Final Result

<img width="480" alt="image" src="https://user-images.githubusercontent.com/11497518/221873360-93f0d233-ae3b-4206-a279-05e76d29a830.png">
