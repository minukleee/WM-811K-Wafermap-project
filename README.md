# WM-811K-Wafermap-project
웨이퍼맵 분류 머신러닝 프로젝트: 3중 피쳐 + 분류 알고리즘

# Step1: 파일 불러오기 및 데이터 확인: WM-811K Wafermap

Input: WM-811K dataset provided by (http://mirlab.org/dataSet/public/).

```python
# loading libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline 

import os
print(os.listdir("."))
import warnings
warnings.filterwarnings("ignore")
```

`
['LSWMD.pkl', 'Untitled.ipynb', '.ipynb_checkpoints']
`

```python
df=pd.read_pickle("LSWMD.pkl")
df.head()
df.tail()
```


# Step2: 데이터 시각화 및 전처리

```python
import matplotlib.pyplot as plt
%matplotlib inline


uni_Index=np.unique(df.waferIndex, return_counts=True)
plt.bar(uni_Index[0],uni_Index[1], color='gold', align='center', alpha=0.5)
plt.title(" wafer Index distribution")
plt.xlabel("index #")
plt.ylabel("frequency")
plt.xlim(0,26)
plt.ylim(30000,34000)
plt.show()
```

<Figure size 640x480 with 1 Axes><img width="589" height="455" alt="image" src="https://github.com/user-attachments/assets/fdcca787-db66-423e-8cb3-fe215e1d2fb4" />
  
Wafer 인덱스 열 확인, 센서 불량 확인 가능(1lot = 25wafers), 해당 열 불필요



## waferIndex 열 삭제 & dimension(웨이퍼 사이즈) 정의 & 최대최소값 확인

```python
df = df.drop(['waferIndex'], axis = 1)
```
`
waferIndex 열 삭제
`

```python
def find_dim(x):
    dim0=np.size(x,axis=0)
    dim1=np.size(x,axis=1)
    return dim0,dim1
df['waferMapDim']=df.waferMap.apply(find_dim)
df.sample(5)

```python
max(df.waferMapDim), min(df.waferMapDim)
```

`
((300, 202), (6, 21))
`
웨이퍼 사이즈의 최대 및 최소값 확인


```python
uni_waferDim=np.unique(df.waferMapDim, return_counts=True)
uni_waferDim[0].shape[0]
```
`
632
`
총 632가지 사이즈가 존재한다.


## 2차원 리스트 정제 및 범주형 데이터 인코딩


```python
df['failureType'] = df['failureType'].apply(lambda x: x[0][0] if len(x) > 0 and len(x[0]) > 0 else None)
df['trianTestLabel'] = df['trianTestLabel'].apply(lambda x: x[0][0] if len(x) > 0 and len(x[0]) > 0 else None)
```
`
리스트 정제
`


```python
df['failureNum'] = df.failureType
df['trainTestNum'] = df.trianTestLabel

mapping_type = {'Center':0,'Donut':1,'Edge-Loc':2,'Edge-Ring':3,'Loc':4,'Random':5,'Scratch':6,'Near-full':7,'none':8}
mapping_traintest = {'Training':0,'Test':1}

df = df.replace({'failureNum': mapping_type, 'trainTestNum': mapping_traintest})
```
`웨이퍼맵 이름 데이터 인코딩 `

```python
print(df[['failureType', 'failureNum', 'trianTestLabel', 'trainTestNum']].head())
```
`test`


## 결측 데이터 확인 + 시각화

```python
tol_wafers = df.shape[0]
tol_wafers
```
`811457`

```python
df_withlabel = df[(df['failureNum']>=0) & (df['failureNum']<=8)]
df_withlabel =df_withlabel.reset_index()
df_withpattern = df[(df['failureNum']>=0) & (df['failureNum']<=7)]
df_withpattern = df_withpattern.reset_index()
df_nonpattern = df[(df['failureNum']==8)]
df_withlabel.shape[0], df_withpattern.shape[0], df_nonpattern.shape[0]
```

`(172950, 25519, 147431)`

```python
import matplotlib.pyplot as plt
%matplotlib inline

from matplotlib import gridspec
fig = plt.figure(figsize=(20, 4.5)) 
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 2.5]) 
ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1])

no_wafers=[tol_wafers-df_withlabel.shape[0], df_withpattern.shape[0], df_nonpattern.shape[0]]

colors = ['silver', 'orange', 'gold']
explode = (0.1, 0, 0)  # explode 1st slice
labels = ['no-label','label&pattern','label&non-pattern']
ax1.pie(no_wafers, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)

uni_pattern=np.unique(df_withpattern.failureNum, return_counts=True)
labels2 = ['','Center','Donut','Edge-Loc','Edge-Ring','Loc','Random','Scratch','Near-full']
ax2.bar(uni_pattern[0],uni_pattern[1]/df_withpattern.shape[0], color='gold', align='center', alpha=0.9)
ax2.set_title("failure type frequency")
ax2.set_ylabel("% of pattern wafers")
ax2.set_xticklabels(labels2)

plt.show()
```

<Figure size 2000x450 with 2 Axes><img width="1541" height="412" alt="image" src="https://github.com/user-attachments/assets/23757fce-ab8b-4838-94bf-0652f8e5bd69" />
only 라벨 결측이 압도적으로 높은걸 확인 + 패턴 별 결측 비율 확인


## 웨이퍼 맵 시각화

```python
fig, ax = plt.subplots(nrows = 10, ncols = 10, figsize=(20, 20))
ax = ax.ravel(order='C')
for i in range(100):
    img = df_withpattern.waferMap[i]
    ax[i].imshow(img)
    ax[i].set_title(df_withpattern.failureType[i][0][0], fontsize=10)
    ax[i].set_xlabel(df_withpattern.index[i], fontsize=8)
    ax[i].set_xticks([])
    ax[i].set_yticks([])
plt.tight_layout()
plt.show()
```

<Figure size 2000x2000 with 100 Axes><img width="1969" height="1993" alt="image" src="https://github.com/user-attachments/assets/d75741f7-37c0-4ed6-9b44-410919574737" />
10 * 10 으로 일부만 확인한 상태, 다음은 타입 별 구분

```python
x = [0,1,2,3,4,5,6,7]
labels2 = ['Center','Donut','Edge-Loc','Edge-Ring','Loc','Random','Scratch','Near-full']

for k in x:
    fig, ax = plt.subplots(nrows = 1, ncols = 10, figsize=(18, 12))
    ax = ax.ravel(order='C')
    for j in [k]:
        img = df_withpattern.waferMap[df_withpattern.failureType==labels2[j]]
        for i in range(10):
            ax[i].imshow(img[img.index[i]])
            ax[i].set_title(df_withpattern.failureType[img.index[i]][0][0], fontsize=10)
            ax[i].set_xlabel(df_withpattern.index[img.index[i]], fontsize=10)
            ax[i].set_xticks([])
            ax[i].set_yticks([])
    plt.tight_layout()
    plt.show()
```
<Figure size 1800x1200 with 10 Axes><img width="1789" height="222" alt="image" src="https://github.com/user-attachments/assets/1981adfe-8796-4bc8-8232-a8f687d8a8eb" />
타입 별 같은 행으로 정렬하였다. 가장 우수한 특성을 보이는 웨이퍼를 선정해서 분류해서 다시 시각화 해보자.


```python
x = [9,340, 3, 16, 0, 25, 84, 37]
labels2 = ['Center','Donut','Edge-Loc','Edge-Ring','Loc','Random','Scratch','Near-full']

#ind_def = {'Center': 9, 'Donut': 340, 'Edge-Loc': 3, 'Edge-Ring': 16, 'Loc': 0, 'Random': 25,  'Scratch': 84, 'Near-full': 37}
fig, ax = plt.subplots(nrows = 2, ncols = 4, figsize=(20, 10))
ax = ax.ravel(order='C')
for i in range(8):
    img = df_withpattern.waferMap[x[i]]
    ax[i].imshow(img)
    ax[i].set_title(df_withpattern.failureType[x[i]][0][0],fontsize=24)
    ax[i].set_xticks([])
    ax[i].set_yticks([])
plt.tight_layout()
plt.show() 
```
<Figure size 2000x1000 with 8 Axes><img width="1975" height="996" alt="image" src="https://github.com/user-attachments/assets/200ceed6-7b40-46c7-b65f-421356d59593" />
이제 해당 웨이퍼맵들이 기준점이 될 예정.
  
다음은 feature 선정의 단계로 넘어간다.


# 1번째 feature 선정 - Density-based Features (13개: 시각화 참고)

```python
import skimage
from skimage import measure
from skimage.transform import radon
from skimage.transform import probabilistic_hough_line
from skimage import measure
from scipy import interpolate
from scipy import stats
```
`진행하기 전 미리 라이브러리 임포트`

```python
an=np.linspace(0, 2*np.pi, 100)                
plt.plot(2.5*np.cos(an), 2.5*np.sin(an))            
plt.axis('equal')
plt.axis([-4,4,-4,4])                

plt.plot([-2.5, 2.5], [1.5, 1.5])
plt.plot([-2.5, 2.5], [0.5, 0.5 ])
plt.plot([-2.5, 2.5], [-0.5, -0.5 ])
plt.plot([-2.5, 2.5], [-1.5,-1.5 ])

plt.plot([0.5, 0.5], [-2.5, 2.5])
plt.plot([1.5, 1.5], [-2.5, 2.5])
plt.plot([-0.5, -0.5], [-2.5, 2.5])
plt.plot([-1.5, -1.5], [-2.5, 2.5])


for i in range(-1, 2):  # x축 범위: -1, 0, 1
    for j in range(-1, 2):  # y축 범위: -1, 0, 1
        x = [i - 0.5, i + 0.5, i + 0.5, i - 0.5]
        y = [j - 0.5, j - 0.5, j + 0.5, j + 0.5]
        plt.fill(x, y, 'blue', alpha=0.2)

x = [-0.5, 0.5, 0.5, -0.5]
y = [1.5, 1.5, 2.5, 2.5]
plt.fill(x, y, 'blue', alpha=0.2) 
x = [-0.5, 0.5, 0.5, -0.5]
y = [-1.5, -1.5, -2.5, -2.5]
plt.fill(x, y, 'blue', alpha=0.2) 
y = [-0.5, 0.5, 0.5, -0.5]
x = [-1.5, -1.5, -2.5, -2.5]
plt.fill(x, y, 'blue', alpha=0.2) 
y = [-0.5, 0.5, 0.5, -0.5]
x = [1.5, 1.5, 2.5, 2.5]
plt.fill(x, y, 'blue', alpha=0.2) 


plt.title("Devide wafer map to 13 regions")
plt.xticks([]) # 축 없애기
plt.yticks([])
plt.show()
```




















