
# Computer Vision 이상치 탐지 알고리즘 경연대회
![image](https://user-images.githubusercontent.com/109740391/221884356-487e4af6-86ab-4a8f-a173-b9af55a6f5d2.png)

### 팀원 – 16기 이수찬, 17기 김지윤, 17기 이서연 

### 1. 대회 소개
사물의 종류를 분류하고 정상 샘플과 비정상(이상치) 샘플을 분류하는 컴퓨터 비전 알고리즘 개발

### 2. 데이터
-	train [Folder] : 4277개 이미지, 88개 label
-	test[Folder] : 2154개 이미지 
-	train_df(csv) : train folder에 대한 정보(인덱스, 파일명, 클래스 등)
-	test_df(csv) : test folder에 대한 정보(인덱스, 파일명)

### 3. 전처리 
-	Augmentation, Normalization

### 4. 모델링
-	1차 시도 : transforms.RandomAffine((-45,-45)) + efficientnet_b3 + epoch수(30)
-	2차 시도 : transforms.RandomAffine((-180,180)) +efficientnet_b3 +epoch수(40) 
-	3차 시도 : 2차 시도 + epoch수(50)로 진행
-	4차 시도 : 3차 시도 + epoch수(40)+post-processing 진행

### 5. 후처리
-	1) class 불균형으로 인한 good label의 과한 예측 방지 
-	2) i.의 결과를 이용해도 헷갈려하는 class에 대해 추가 학습

### 6. 제언
-	1) 전처리에서 이상치 데이터 불균형 문제를 해결하기 위한 방법 추가
-	2) 모델 학습 과정에서 K-fold 교차 검증 사용

