#  🔥 개인 투자자를 위한 기업 최신 뉴스 정보 요약 서비스 🔥

<img width="1214" alt="문서요약 서비스화면" src="https://user-images.githubusercontent.com/87636737/221851933-fef54a74-8cd7-4d9d-b553-a9064f9223d7.png">

기업의 비재무적 정보(최신 기사)를 손쉽게 얻을 수 있는 서비스를 개발


Team : 14기 김태영, 15기 박민규, 16기 임청수

<br/><br/>


## Project Descriptions

**[주제]**  
개인 투자자를 위한 기업 최신 뉴스 정보 요약 서비스 


**[설명]**  
- KoBART 기반 문서 생성요약 모델 구축
- Sentnece Transformer를 이용해 요약 품질 검수
- Fast API 기반 백엔드 / HTML, CSS, JS 기반 프론트엔드 서비스 개발


**[평가 산식]**  
ROUGE(Recall Oriented Understudy for Gisting Evaluation)
-	ROUGE-L (Precision = 0.415 / Recall = 0.440)

<br/><br/>


## Environment
Colab Pro+  

<br/><br/>

## Data  
- [AI Hub 문서 요약 데이터셋](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=97)
- 네이버 증권 종목 검색 후 뉴스 공시탭의 뉴스원문

<br/><br/>


## Code Descriptions
1. main.py
- fastapi Backend 구동
- fine-tuned model load

2. main_code_v2.py
- Crawling
- 문서 생성 요약

3. front1, front2, koogle
- Frontend 디자인

