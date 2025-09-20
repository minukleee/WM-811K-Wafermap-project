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
<Figure size 1800x1200 with 10 Axes><img width="1789" height="222" alt="image" src="https://github.com/user-attachments/assets/cbf6bbb5-8dbc-4bf5-b399-430c4f9f5890" />
<Figure size 1800x1200 with 10 Axes><img width="1789" height="222" alt="image" src="https://github.com/user-attachments/assets/47bf9b76-42b2-4713-b5aa-6196e3429e14" />
<Figure size 1800x1200 with 10 Axes><img width="1789" height="222" alt="image" src="https://github.com/user-attachments/assets/d8c1c9f9-856d-422d-8b28-65a9a4bd2528" />
<Figure size 1800x1200 with 10 Axes><img width="1789" height="222" alt="image" src="https://github.com/user-attachments/assets/6db46837-31d4-4fdb-9b86-d5088b03749c" />
<Figure size 1800x1200 with 10 Axes><img width="1789" height="222" alt="image" src="https://github.com/user-attachments/assets/98eb4486-586f-45d0-be27-04226a51d39f" />
<Figure size 1800x1200 with 10 Axes><img width="1789" height="222" alt="image" src="https://github.com/user-attachments/assets/0aabc167-cab4-4bd3-bb5a-5feaf774cbcf" />
<Figure size 1800x1200 with 10 Axes><img width="1789" height="222" alt="image" src="https://github.com/user-attachments/assets/b4d6d518-7748-4a12-b385-a03a3ae7be12" />
<Figure size 1800x1200 with 10 Axes><img width="1789" height="249" alt="image" src="https://github.com/user-attachments/assets/a333cafc-f2fd-402b-a868-1c2d6a453a6d" />


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


## 1번째 feature 선정 - Density-based Features (13개: 시각화 참고)

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
plt.xticks([]) 
plt.yticks([])
plt.show()
```
<Figure size 640x480 with 1 Axes><img width="515" height="411" alt="image" src="https://github.com/user-attachments/assets/f4de2a6f-9c7f-4e6d-b1ad-d7d7e6354eb1" />
  
중간 9부위와 상하좌우 4부위 사용해서 웨이퍼맵의 총 13부분의 밀도를 feature로 사용한다

이제 해당 region을 df_withpattern['fea_reg']에 입력한다.

```python
df_withpattern['fea_reg']=df_withpattern.waferMap.apply(find_regions)
```
다음은 시각화로 재확인

```python
x = [9,340, 3, 16, 0, 25, 84, 37]
labels2 = ['Center','Donut','Edge-Loc','Edge-Ring','Loc','Random','Scratch','Near-full']

fig, ax = plt.subplots(nrows = 2, ncols = 4,figsize=(20, 10))
ax = ax.ravel(order='C')
for i in range(8):
    ax[i].bar(np.linspace(1,13,13),df_withpattern.fea_reg[x[i]])
    ax[i].set_title(df_withpattern.failureType[x[i]][0][0],fontsize=15)
    ax[i].set_xticks([])
    ax[i].set_yticks([])

plt.tight_layout()
plt.show() 
```

<Figure size 2000x1000 with 8 Axes><img width="1990" height="989" alt="image" src="https://github.com/user-attachments/assets/39ee2876-a348-4dd7-893a-302a22075ca8" />

타입 별 나름 뚜렷한 차이를 보이므로 유의미한 feature로 채택 가능함.


## 2번째 feature 선정 - Radon-based Features (40개: 20개 평균, 20개 표준편차 피처)

라돈 변환은 위치별 픽셀의 밀도 변화를 감지하는 데 탁월하다. -> 웨이퍼맵 분류에 적합함.

```python
def change_val(img):
    img[img==1] =0  
    return img

df_withpattern_copy = df_withpattern.copy()
df_withpattern_copy['new_waferMap'] =df_withpattern_copy.waferMap.apply(change_val)
```


```python
x = [9,340, 3, 16, 0, 25, 84, 37]
labels2 = ['Center','Donut','Edge-Loc','Edge-Ring','Loc','Random','Scratch','Near-full']

fig, ax = plt.subplots(nrows = 2, ncols = 4, figsize=(20, 10))
ax = ax.ravel(order='C')
for i in range(8):
    img = df_withpattern_copy.waferMap[x[i]]
    theta = np.linspace(0., 180., max(img.shape), endpoint=False)
    sinogram = radon(img, theta=theta)    
      
    ax[i].imshow(sinogram, cmap=plt.cm.Greys_r, extent=(0, 180, 0, sinogram.shape[0]), aspect='auto')
    ax[i].set_title(df_withpattern_copy.failureType[x[i]][0][0],fontsize=15)
    ax[i].set_xticks([])
plt.tight_layout()

plt.show()
```
<Figure size 2000x1000 with 8 Axes><img width="1989" height="990" alt="image" src="https://github.com/user-attachments/assets/90fd5c45-1519-427c-8560-50721c7ed536" />

라돈 변환 결과 시각화. 해당 결과로 그대로 사용해서는 안된다.
웨이퍼의 크기가 모두 다 다르기 때문, 20개의 평균 & 표준편차 값으로 변환


```python
def cubic_inter_mean(img):
    theta = np.linspace(0., 180., max(img.shape), endpoint=False)
    sinogram = radon(img, theta=theta)
    xMean_Row = np.mean(sinogram, axis = 1)
    x = np.linspace(1, xMean_Row.size, xMean_Row.size)
    y = xMean_Row
    f = interpolate.interp1d(x, y, kind = 'cubic')
    xnew = np.linspace(1, xMean_Row.size, 20)
    ynew = f(xnew)/100   # use interpolation function returned by `interp1d`
    return ynew

def cubic_inter_std(img):
    theta = np.linspace(0., 180., max(img.shape), endpoint=False)
    sinogram = radon(img, theta=theta)
    xStd_Row = np.std(sinogram, axis=1)
    x = np.linspace(1, xStd_Row.size, xStd_Row.size)
    y = xStd_Row
    f = interpolate.interp1d(x, y, kind = 'cubic')
    xnew = np.linspace(1, xStd_Row.size, 20)
    ynew = f(xnew)/100   # use interpolation function returned by `interp1d`
    return ynew
```

```python
df_withpattern_copy['fea_cub_mean'] =df_withpattern_copy.waferMap.apply(cubic_inter_mean)
df_withpattern_copy['fea_cub_std'] =df_withpattern_copy.waferMap.apply(cubic_inter_std)
```

```python
x = [9, 340, 3, 16, 0, 25, 84, 37]
labels2 = ['Center','Donut','Edge-Loc','Edge-Ring','Loc','Random','Scratch','Near-full']

fig, ax = plt.subplots(nrows = 2, ncols = 4,figsize=(20, 10))
ax = ax.ravel(order='C')
for i in range(8):
    ax[i].bar(np.linspace(1,20,20),df_withpattern_copy.fea_cub_mean[x[i]])
    ax[i].set_title(df_withpattern_copy.failureType[x[i]][0][0],fontsize=10)
    ax[i].set_xticks([])
    # ax[i].set_xlim([0,21])   
    # ax[i].set_ylim([0,1])
plt.tight_layout()
plt.show() 
```
<Figure size 2000x1000 with 8 Axes><img width="1990" height="989" alt="image" src="https://github.com/user-attachments/assets/92b1ff1c-9a4b-47ef-92a1-a02ced395bfe" />


```python
fig, ax = plt.subplots(nrows = 2, ncols = 4,figsize=(20, 10))
ax = ax.ravel(order='C')
for i in range(8):
    ax[i].bar(np.linspace(1,20,20),df_withpattern_copy.fea_cub_std[x[i]])
    ax[i].set_title(df_withpattern_copy.failureType[x[i]][0][0],fontsize=10)
    ax[i].set_xticks([])
    # ax[i].set_xlim([0,21])   
    # ax[i].set_ylim([0,0.3])
plt.tight_layout()
plt.show()
```

<Figure size 2000x1000 with 8 Axes><img width="1989" height="989" alt="image" src="https://github.com/user-attachments/assets/1f84722a-d882-4c2d-9743-c07af449f20b" />


평균과 표준편차 각각 20 features 그래프로 plot하였다. 끝.

## 3번째 feature 선정 - Geometry-based Features (6개: 면적, 둘레, 장축길이, 단축길이, 이심률(비율), 밀도)

```python
x = [9,340, 3, 16, 0, 25, 84, 37]
labels2 = ['Center','Donut','Edge-Loc','Edge-Ring','Loc','Random','Scratch','Near-full']

fig, ax = plt.subplots(nrows = 2, ncols = 4,figsize=(20, 10))
ax = ax.ravel(order='C')
for i in range(8):
    img = df_withpattern_copy.waferMap[x[i]]
    zero_img = np.zeros(img.shape)
    img_labels = measure.label(img, connectivity=1, background=0) # neighbors=4 부분만 삭제
    img_labels = img_labels-1
    if img_labels.max()==0:
        no_region = 0
    else:
        info_region = stats.mode(img_labels[img_labels>-1], axis = None)
        no_region = info_region[0]
    
    zero_img[np.where(img_labels==no_region)] = 2 
    ax[i].imshow(zero_img)
    ax[i].set_title(df_withpattern_copy.failureType[x[i]][0][0],fontsize=10)
    ax[i].set_xticks([])
plt.tight_layout()
plt.show() 
```

<Figure size 2000x1000 with 8 Axes><img width="1989" height="983" alt="image" src="https://github.com/user-attachments/assets/535ed192-837b-43b3-9c22-7e080f1dbf55" />

패턴 별 가장 특징이 두드러지는 부분만 시각화하였다.


아래는 각 객체(면적, 둘레, 장축길이, 단축길이, 이심률(비율), 밀도)를 정의하고 계산하는 코드.

```python
def cal_dist(img,x,y):
    dim0=np.size(img,axis=0)    
    dim1=np.size(img,axis=1)
    dist = np.sqrt((x-dim0/2)**2+(y-dim1/2)**2)
    return dist  
```

```python
from scipy import stats
from skimage import measure


def fea_geom(img):
    norm_area=img.shape[0]*img.shape[1]
    norm_perimeter=np.sqrt((img.shape[0])**2+(img.shape[1])**2)
    
    img_labels = measure.label(img, connectivity=1, background=0)

    if img_labels.max()==0:
        img_labels[img_labels==0]=1
        no_region = 0
    else:
        info_region = stats.mode(img_labels[img_labels>0], axis=None)
        no_region = info_region.mode - 1
    
    prop = measure.regionprops(img_labels.astype(int))
    
    if hasattr(no_region, "__len__"):
        no_region = no_region[0]
        
    prop_area = prop[no_region].area/norm_area
    prop_perimeter = prop[no_region].perimeter/norm_perimeter 
    
    prop_cent = prop[no_region].local_centroid 
    prop_cent = cal_dist(img,prop_cent[0],prop_cent[1])
    
    prop_majaxis = prop[no_region].major_axis_length/norm_perimeter 
    prop_minaxis = prop[no_region].minor_axis_length/norm_perimeter  
    prop_ecc = prop[no_region].eccentricity  
    prop_solidity = prop[no_region].solidity  
    
    return prop_area, prop_perimeter, prop_majaxis, prop_minaxis, prop_ecc, prop_solidity
```

```python
df_withpattern_copy['fea_geom'] = df_withpattern_copy.waferMap.apply(fea_geom)
```
`df_withpattern_copy['fea_geom']에 해당 return값 입력.`

```python
df_withpattern_copy.fea_geom[340]
```
`(np.float64(0.30881585811163276),
 np.float64(3.4633305623147477),
 np.float64(0.7464951525564261),
 np.float64(0.5214489845402435),
 0.7155811292862498,
 np.float64(0.6103092783505155))`
 도넛타입[340]을 예로 6가지 객체값이 잘 읽히는 것 확인
 

# Step3: feature 종합 단계 (총정리)

Density-based Features: 13개


Radon-based Features: 40개


Geometry-based Features: 6개


총합: 59개가 나오는지 확인.

```python
df_all=df_withpattern_copy.copy()
a=[df_all.fea_reg[i] for i in range(df_all.shape[0])]
b=[df_all.fea_cub_mean[i] for i in range(df_all.shape[0])] 
c=[df_all.fea_cub_std[i] for i in range(df_all.shape[0])] 
d=[df_all.fea_geom[i] for i in range(df_all.shape[0])] 
fea_all = np.concatenate((np.array(a),np.array(b),np.array(c),np.array(d)),axis=1) 
```

```python
fea_all.shape
```

`(25519, 59)`

```python
label=[df_all.failureNum[i] for i in range(df_all.shape[0])]
label=np.array(label)

label
```

`array([4., 2., 2., ..., 3., 2., 3.], shape=(25519,))`

```python
len(label)
```

`25519`


# Step4: train_test_split
훈련 데이터와 테스트 데이터를 분류한다.

```python
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from collections import  Counter

# feature 데이터와 정답 데이터 분류하고 훈련데이터와 테스트데이터 분류 
X = fea_all
y = label

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)                      
print('Training target statistics: {}'.format(Counter(y_train)))
print('Testing target statistics: {}'.format(Counter(y_test)))

RANDOM_STATE =42
```


# Step5: 머신러닝 알고리즘 선정: One-VS-One multi-class SVM 

```python
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier

clf2 = OneVsOneClassifier(LinearSVC(random_state = RANDOM_STATE)).fit(X_train, y_train)

y_train_pred = clf2.predict(X_train)
y_test_pred = clf2.predict(X_test)

train_acc2 = np.sum(y_train == y_train_pred, axis=0, dtype='float') / X_train.shape[0]
test_acc2 = np.sum(y_test == y_test_pred, axis=0, dtype='float') / X_test.shape[0]

print('One-Vs-One Training acc: {}'.format(train_acc2*100))
print('One-Vs-One Testing acc: {}'.format(test_acc2*100))
```

`One-Vs-One Training acc: 82.73159517216155
One-Vs-One Testing acc: 82.3667711598746`

```python
#평가
print(classification_report(y_train, y_train_pred))
print('Acc Score :', accuracy_score(y_train, y_train_pred))
```

```
precision    recall  f1-score   support

         0.0       0.91      0.94      0.92      3238
         1.0       0.85      0.79      0.82       404
         2.0       0.69      0.72      0.71      3860
         3.0       0.92      0.94      0.93      7299
         4.0       0.68      0.64      0.66      2677
         5.0       0.91      0.87      0.89       640
         6.0       0.69      0.52      0.59       905
         7.0       0.97      0.99      0.98       116

    accuracy                           0.83     19139
   macro avg       0.83      0.80      0.81     19139
weighted avg       0.82      0.83      0.83     19139

Acc Score : 0.8273159517216155
```


```python
# 평가
print(classification_report(y_test, y_test_pred))
print('Acc Score :', accuracy_score(y_test, y_test_pred))
```

```
           precision    recall  f1-score   support

         0.0       0.91      0.93      0.92      1056
         1.0       0.83      0.79      0.81       151
         2.0       0.69      0.72      0.70      1329
         3.0       0.91      0.93      0.92      2381
         4.0       0.68      0.65      0.66       916
         5.0       0.93      0.89      0.91       226
         6.0       0.68      0.56      0.61       288
         7.0       0.92      1.00      0.96        33

    accuracy                           0.82      6380
   macro avg       0.82      0.81      0.81      6380
weighted avg       0.82      0.82      0.82      6380

Acc Score : 0.8236677115987461
```

# 최종 Confusion Matrix 시각화

```python
import itertools
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')  
```

```python
# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_test_pred)
# np.set_printoptions(precision=2)

from matplotlib import gridspec
fig = plt.figure(figsize=(15, 8)) 
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1]) 

## Plot non-normalized confusion matrix
plt.subplot(gs[0])
plot_confusion_matrix(cnf_matrix, title='Confusion matrix')

# Plot normalized confusion matrix
plt.subplot(gs[1])
plot_confusion_matrix(cnf_matrix, normalize=True, title='Normalized confusion matrix')

plt.tight_layout()
plt.show()
```

<Figure size 1500x800 with 4 Axes><img width="1454" height="790" alt="image" src="https://github.com/user-attachments/assets/fa4918c5-f125-4b06-b611-aa36e2274e07" />


훌륭한 분류 모델이 개발되었다.


기회가 생긴다면, 더 상위 분류 모델을 통해 검증해 볼 수 있겠음.
































