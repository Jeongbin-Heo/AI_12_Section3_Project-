# :wine_glass: 와인 리뷰 데이터 감정 분석
* **진행 기간** : 2022. 05. 19 ~ 2022. 05. 25

### &nbsp;

## :books: 사용 Skill
* **Google Colab**
* **Python 3.7.13**
  * 데이터 전처리 : pandas, numpy, nltk, re, sklearn, tensorflow
  * 모델링 : tensorflow, keras, json
* **DL Models**
  * Embedding
  * Word2vec
  * LSTM
  * Dropout

### &nbsp;

## :pushpin: Contents
* :one: **프로젝트 개요**
* :two: **데이터 전처리**
* :three: **모델링**
* :four: **모델 활용**
* :five: ****

### &nbsp;

## :one: 프로젝트 개요
### 프로젝트 진행 배경
* 와인을 구매할 때 고려해야 할 요소가 많기에 다른 주류에 비해 진입장벽이 높은 편에 속함
* 따라서 와인을 잘 모르는 입문자들의 경우에는 평가 내용을 보고 구매하는 것이 일반적
* 하지만 소믈리에와 같이 와인을 평가하는 사람들은 은유적인 표현을 써서 직관적으로 해당 평가 내용만 보고 와인이 맛이 있는지 없는지 판단하기가 쉽지 않음
### 프로젝트 목적
* 와인 평가 내용을 적용시켰을 때 긍정/부정 여부를 판별하는 모델을 개발
* 와인 입문자들에게 구매하는데 도움이 될 뿐 아니라, 다른 분야에도 적용 기대


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

## :two: 데이터 전처리
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

> 1. 정규표현식(regex)를 사용하여 숫자, 문장 부호 등을 제거하고 **알파벳만 추출**
> 2. 내장함수 lower()를 사용하여 대문자를 **소문자**로 통일
> 3. NLTK 불용어 사전에 등록되어 있는 단어들(ex. I, of, both, ...)을 감정 분석에 필요없는 **불용어**로 판단하여 제거
> 4. NLTK WordNet Lemmatizer를 사용하여 과거형(tried), 복수형(bottles) 등으로 나타나있는 단어를 표제어 형태로 변환

### 데이터셋 분리
* 모델에 적용시키기 위해 사용할 Tokenizer를 학습시키고 적용시키기 위해 학습 셋 / 테스트 셋으로 분리
* 학습 셋 80%, 테스트 셋 20% 로 분리

### 모델에 적용시킬 수 있는 형태로 변환
<img width="565" alt="스크린샷 2022-09-10 오후 9 36 13" src="https://user-images.githubusercontent.com/97662174/189483617-a0c25adb-e5f3-47e7-989a-b24a6db55392.png">

> 1. 문장을 단어로 토큰화
> 2. 학습 셋에 대해 학습시킨 Tokenizer를 사용하여 단어에 해당되는 숫자의 시퀀스 형태로 변환
> 3. 모델에 적용시키기 위해 다른 길이의 시퀀스들이 일정한 길이로 만들어지도록 패딩 처리



### &nbsp;

## :three: 모델

### 딥러닝 설계

<img width="560" alt="스크린샷 2022-09-11 오전 4 48 52" src="https://user-images.githubusercontent.com/97662174/189499516-0ee6dec1-7529-4912-9123-23403104f380.png">

> **Embedding layer 사용 이유**
>> <img width="858" alt="스크린샷 2022-09-10 오후 11 47 55" src="https://user-images.githubusercontent.com/97662174/189488567-1bbcec2c-6327-4538-997c-97d65cb48eca.png">
>>
>> 학습시킨 Tokenizer를 통해 모든 리뷰 데이터에 나타나있는 단어들의 수는 23401개라는 것을 알 수 있는데,
>> 이를 일반적인 원-핫 인코딩 방식으로 학습을 하게 된다면 각 단어들이 너무 큰 23401차원의 데이터로 모델에 학습되는 것이기에 차원의 저주에 빠질 우려가 있다.  
>> 따라서 임베딩 가중치 행렬을 곱해주어 각 단어들을 300차원의 벡터로 나타내어줄 것인데, 이 때 구글 뉴스를 통해 학습된 Word2vec 모델의 단어 벡터들을 이용하여 임베딩 가중치 행렬을 만들어주었다.
>
> **LSTM layer 사용 이유**
>> 성능을 향상시키기 위해 추가한 은닉층이다.  
>> 일반적인 RNN의 경우에는 역전파로 가중치를 업데이트하는 과정에서 기울기 소실/폭발이 발생할 수 있기에 LSTM layer를 추가해주었다.


### 모델 컴파일
* **옵티마이저** : adam
* **손실 함수** : binary_crossentrophy
* **평가 지표** : accuracy

### 모델 학습
* **배치 사이즈** : 100
* **에포크** : 20

|Data set|loss|accuracy|
|:---:|:---:|:---:|
|Train set|0.3370|**0.8476**|
|Test set|0.3710|**0.8315**|

### &nbsp;

## :four: 모델 활용
* 리뷰 내용 텍스트를 전처리하는 과정을 거쳐 모델에 적용시켰을 때의 결과 = 0과 1 사이의 값
* 0.5를 기준으로 기준값보다 높으면 긍정, 낮으면 부정으로 판단
> * 예시문장 : Big, rich and off-dry, this is powered by intense spiciness and rounded texture. Lychees dominate the fruit profile, giving an opulent feel to the aftertaste. Drink now.
> * 결과값 : 0.8757
> * 판별 : 긍정 -> 해당 리뷰가 남겨진 와인 구매를 고려할 수 있다!

### &nbsp;

## :five: 한계점
* **캐글 데이터셋 사용** : 크롤링을 통해 직접 후기 텍스트를 수집하려고 하였으나 모델의 학습을 통한 성능 향상을 위해 캐글에 올라와있는 정제된 데이터셋을 사용한 점이 아쉬움으로 남음. 추후 직접 데이터를 수집하는 과정까지 시도해볼 필요가 있음  
* **모델 기반 추가적인 서비스 구상** : 이번 프로젝트는 리뷰 텍스트의 긍정/부정 여부를 판별하는 모델을 만드는데 그쳤지만 이 모델을 활용하여 와인을 추천해주는 서비스를  프로젝트를 더 확장시킬 방안을 구상할 수 있음
