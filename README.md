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

### 감정 분석에 필요한 형태로 변환
<img width="595" alt="스크린샷 2022-09-10 오후 9 24 39" src="https://user-images.githubusercontent.com/97662174/189483218-5674a2ca-40e9-4507-a098-a215a15f10ce.png">

1. 정규표현식(regex)를 사용하여 숫자, 문장 부호 등을 제거하고 **알파벳만 추출**
2. 내장함수 lower()를 사용하여 대문자를 **소문자**로 통일
3. NLTK 불용어 사전에 등록되어 있는 단어들(ex. I, of, both, ...)을 감정 분석에 필요없는 **불용어**로 판단하여 제거
4. NLTK WordNet Lemmatizer를 사용하여 과거형(tried), 복수형(bottles) 등으로 나타나있는 단어를 표제어 형태로 변환

### 데이터셋 분리
* 모델에 적용시키기 위해 사용할 Tokenizer를 학습시키고 적용시키기 위해 학습 셋 / 테스트 셋으로 분리
* 학습 셋 80%, 테스트 셋 20% 로 분리

### 모델에 적용시킬 수 있는 형태로 변환
<img width="565" alt="스크린샷 2022-09-10 오후 9 36 13" src="https://user-images.githubusercontent.com/97662174/189483617-a0c25adb-e5f3-47e7-989a-b24a6db55392.png">

1. 문장을 단어로 토큰화
2. 학습 셋에 대해 학습시킨 Tokenizer를 사용하여 단어에 해당되는 숫자의 시퀀스 형태로 변환
3. 모델에 적용시키기 위해 다른 길이의 시퀀스들이 일정한 길이로 만들어지도록 패딩 처리



### &nbsp;

## *III. 모델 설계 및 학습*
### Word2vec의 임베딩 가중치 행렬 생성
<img width="858" alt="스크린샷 2022-09-10 오후 11 47 55" src="https://user-images.githubusercontent.com/97662174/189488567-1bbcec2c-6327-4538-997c-97d65cb48eca.png">

학습시킨 Tokenizer를 통해 모든 리뷰 데이터에 나타나있는 단어들의 수는 23401개라는 것을 알 수 있었다. 
이를 일반적인 원-핫 인코딩 방식으로 학습을 하게 된다면 각 단어들이 너무 큰 23401차원의 데이터로 모델에 학습되는 것이기에 차원의 저주에 빠질 우려가 있다.
따라서 여기에 임베딩 가중치 행렬을 곱해주어 각 단어들을 300차원의 벡터로 나타내어줄 것인데, 이 때 구글 뉴스를 통해 학습된 Word2vec 단어 벡터들을 이용하여 임베딩 가중치 행렬을 만들어주었다.

### 딥러닝 모델 설계

```
model = Sequential()
model.add(Embedding(vocab_size, 300, input_length = maxlen, weights = [embedding_matrix], trainable = False))
model.add(LSTM(128, return_sequences = True))
model.add(LSTM(128))
model.add(Dropout(0.1))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))
```

<img width="624" alt="스크린샷 2022-09-10 오후 11 59 52" src="https://user-images.githubusercontent.com/97662174/189489052-0ae613ea-7a58-47f7-af7f-96ce1e5df88d.png">


### 모델 학습


### &nbsp;

## *IV. 모델 활용*
