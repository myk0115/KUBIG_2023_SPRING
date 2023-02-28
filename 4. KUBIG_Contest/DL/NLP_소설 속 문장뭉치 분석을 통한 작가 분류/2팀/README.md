# Deep Learning NLP Project
## 데이콘 소설 작가 분류 AI 경진대회

https://dacon.io/competitions/open/235670/overview/description

16기: 김상옥 / 17기: 우윤규, 김연규

## 1. Data
- text : 문장 뭉치
- author : 작가 인덱스 값

## 2. Data Preparation
- 딥러닝: 문장부호 제거, 띄어쓰기 기반 토큰화 / 불용어 처리 X
- 머신러닝: 텍스트 자체의 특징 기반 특성공학 실시

## 3. Modeling
- 딥러닝: LSTM, RNN
- 머신러닝: XGBoost, Random Forest
- Softvoting: LSTM + XGBoost + Random Forest

## 4. Result
- LogLoss: XGBoost > Softvoting > Random Forest > LSTM
- Accuracy: Softvoting > XGBoost > Random Forest > LSTM
- 결과 분석:
  * 딥러닝: 동일 인물에 대한 소설 -> 문맥을 파악하는 과정에서 과적합 발생  
  * 머신러닝: 특성공학을 통해 유의미한 피쳐 생성 -> 과적합 해결
