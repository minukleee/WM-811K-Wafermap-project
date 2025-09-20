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
```

`
waferMap 	dieSize 	lotName 	waferIndex 	trianTestLabel 	failureType
0 	[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,... 	1683.0 	lot1 	1.0 	[[Training]] 	[[none]]
1 	[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,... 	1683.0 	lot1 	2.0 	[[Training]] 	[[none]]
2 	[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,... 	1683.0 	lot1 	3.0 	[[Training]] 	[[none]]
3 	[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,... 	1683.0 	lot1 	4.0 	[[Training]] 	[[none]]
4 	[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,... 	1683.0 	lot1 	5.0 	[[Training]] 	[[none]]
`
```python
df.tail()
```


`
waferMap 	dieSize 	lotName 	waferIndex 	trianTestLabel 	failureType
811452 	[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 1, 1,... 	600.0 	lot47542 	23.0 	[[Test]] 	[[Edge-Ring]]
811453 	[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 1, 1,... 	600.0 	lot47542 	24.0 	[[Test]] 	[[Edge-Loc]]
811454 	[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 1, 1,... 	600.0 	lot47542 	25.0 	[[Test]] 	[[Edge-Ring]]
811455 	[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,... 	600.0 	lot47543 	1.0 	[] 	[]
811456 	[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 1, 1,... 	600.0 	lot47543 	2.0 	[] 	[]
`





# Step2: 데이터 시각화








































