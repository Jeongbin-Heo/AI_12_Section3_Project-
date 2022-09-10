# 와인 리뷰 데이터 감정 분석
* **개요** : 와인 리뷰 데이터를 이용하여 해당 리뷰 내용의 긍정/부정 여부를 판별하는 모델 설계
* **진행 기간** : 2022. 05. 19 ~ 2022. 05. 25
* **사용 스킬** : `etc`


### &nbsp;

## *I. 프로젝트 개요*
### 프로젝트 배경
* ㅇㅇㅇㅇ

### 사용 데이터
* **데이터 출처** : [Wine Reviews Dataset](https://www.kaggle.com/datasets/zynicide/wine-reviews)
* **데이터 수** : 129971
<img width="1076" alt="스크린샷 2022-09-10 오후 8 50 47" src="https://user-images.githubusercontent.com/97662174/189482133-28017fd4-bf82-460e-acce-cf019bd594a8.png">

|Feature|Description|
|:---:|---|
|country|와인이 생산된 국가|
|description|와인 평가 내용|
|desigination|와인 이름|
|points|와인 평가 점수(1~100)|
|price|와인의 가격|
|province|와인이 생산된 지역|
|region_1|구체적인 지역 1|
|region_2|구체적인 지역 2|
|taster_name|와인 평가를 남긴 사람|
|taster_twitter_handle|와인 평가를 남긴 사람의 트위터 계정|
|title|와인 평가 제목|
|variety|와인의 종류|
|winery|와인이 생산된 와이너리|


### &nbsp;

## *II. 데이터 전처리*
### Feature Engineering
* `description` `points` 특성만 사용
* 새로운 특성 `sentiment`  : 평가 점수의 평균치를 계산, 평균보다 높으면 긍정(1), 평균보다 낮으면 부정(0)

<img width="447" alt="스크린샷 2022-09-10 오후 9 07 41" src="https://user-images.githubusercontent.com/97662174/189482542-e3cec702-99a0-4046-901c-e0146a958d72.png">

|Feature|Description|
|:---:|---|
|description|와인 평가 내용|
|points|와인 평가 점수(1~100)|
|sentiment|해당 와인 평가의 긍정(1)/부정(0) 여부|

### 정규표현식, 내장함수, 불용어 제거, 표제어 추출

### 학습 세트 / 테스트 세트 분리

### Tokenizer 이용 데이터 변환



### &nbsp;

## *III. 모델 설계*


### &nbsp;

## *IV. 모델 활용*
