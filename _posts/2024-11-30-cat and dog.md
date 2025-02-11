---
title: "cat and dog ML 돌려보기"
date: 2024-11-30 16:00:00 +/- TTTT
categories: [졸업작품, cat and dog]
tags: [yolov5, cat and dog, ml]		# TAG는 반드시 소문자로 이루어져야함!
---
# https://www.youtube.com/watch?v=FLf5qmSOkwU&ab_channel=BalajiSrinivasan


```python
import numpy as np
import cv2
import os
import random
import matplotlib.pyplot as plt
import pickle
```


```python
DIRECTORY = r'C:\Users\sunny\Desktop\졸업작품\cats and dogs\dogscats\dogscats\train'
CATEGORIES = ['cats','dogs'] #영상과 경로가 달라 수정함
```

# imread에 붙이는 헤더는 plt로 해야함.
## 영상에선 cv2.imread로 나오지만, plt로 해야 오류 없음

### IMG_SIZE = 50으로 설정하면 너무 이미지가 흐려짐


```python
IMG_SIZE = 100  # 이미지의 크기를 100x100 픽셀로 설정
data = []  # 이미지와 레이블 데이터를 저장할 빈 리스트 초기화

for category in CATEGORIES:  # 각 카테고리에 대해 반복
    folder = os.path.join(DIRECTORY, category)  # 카테고리 폴더 경로 생성
    label = CATEGORIES.index(category)  # 카테고리의 인덱스를 레이블로 지정
    print(folder)  # 현재 작업 중인 폴더 경로 출력

    for img in os.listdir(folder):  # 폴더 내의 각 이미지 파일에 대해 반복
        img_path = os.path.join(folder, img)  # 이미지 파일의 전체 경로 생성
        img_arr = plt.imread(img_path)  # 이미지를 배열로 읽기
        img_arr = cv2.resize(img_arr, (IMG_SIZE, IMG_SIZE))  # 이미지 크기를 100x100으로 조정
        plt.imshow(img_arr)  # 이미지를 화면에 출력 (디버깅 용도)
        data.append([img_arr, label])  # 이미지 배열과 레이블을 data 리스트에 추가
```

    C:\Users\sunny\Desktop\졸업작품\cats and dogs\dogscats\dogscats\train\cats
    C:\Users\sunny\Desktop\졸업작품\cats and dogs\dogscats\dogscats\train\dogs
    


    
![png](output_4_1.png)
    


# data의 크기 출력


```python
len(data)
```




    23000



# data 안의 정보를 랜덤하게 섞음


```python
random.shuffle(data)
```


```python
data[0]
```




    [array([[[125, 123, 131],
             [129, 129, 129],
             [115, 125, 142],
             ...,
             [ 11,  28,  61],
             [ 13,  26,  60],
             [ 22,  35,  70]],
     
            [[101,  95, 105],
             [112, 111, 116],
             [ 70,  81, 101],
             ...,
             [  9,  28,  61],
             [ 11,  27,  63],
             [ 10,  28,  64]],
     
            [[ 59,  49,  58],
             [ 76,  75,  81],
             [ 15,  29,  56],
             ...,
             [  9,  23,  51],
             [ 10,  27,  57],
             [ 10,  26,  59]],
     
            ...,
     
            [[ 33,  37,  36],
             [ 30,  34,  33],
             [ 24,  28,  28],
             ...,
             [  3,  33,  73],
             [  0,  25,  68],
             [  9,  40, 100]],
     
            [[ 20,  24,  24],
             [ 14,  18,  17],
             [ 15,  19,  18],
             ...,
             [ 12,  46, 109],
             [ 13,  44,  98],
             [  2,  27,  64]],
     
            [[ 11,  15,  14],
             [ 11,  15,  14],
             [  8,  12,  11],
             ...,
             [  6,  36,  96],
             [ 19,  57, 113],
             [ 20,  56,  99]]], dtype=uint8),
     0]




```python
X=[]
y=[]

for features, labels in data:
    X.append(features)
    y.append(labels)
```


```python
X=np.array(X)
y=np.array(y)
```


```python
len(X)
```




    23000



# pickle 라이브러리 사용해서 파일 쓰기


```python
pickle.dump(X,open('X.pkl','wb'))
pickle.dump(y,open('y.pkl','wb'))
```


```python

```
