# Case3
Trade Volume Prediction

## Data Set
* Time: 2020/10 - 2020/11
* Variables: Price, Trade volume, Outstanding shares, Daily limit, Disposal of stock, Pawn

## Data cleaning
```
import pandas as pd
%matplotlib inline
import matplotlib.pyplot as plt
import scipy
from pandasql import sqldf
import numpy as np
import statsmodels.api as sm

# 匯入資料
df1 = pd.read_csv('/content/drive/MyDrive/raw_data1.csv') # 股票資料
df2 = pd.read_csv('/content/drive/MyDrive/outstanding shares.csv') # 各公司在外流通股數

# 將資料位移
df1['當日成交量'] = df1['成交量(千股)'] # 除了這個欄位和開盤價其他的轉換為前一日的資料
df1['前一日合計買賣超(張)'] = 0.0
df1['前一日最高價與收盤價價差'] = 0.0
df1['前一日最低價與收盤價價差'] = 0.0
df1['前一日成交量'] = 0.0
df1['前一日漲跌停'] = None
df1['前一日高低價差%'] = 0.0
df1['前一日處置股票(D)'] = None
df1['前一日現沖+當沖比重'] = 0.0
for i in range(len(df1['年月日'])):
  df1['年月日'][i] = df1['年月日'][i].split('/')
for i in range(len(df1['年月日'])):
  if i+1 < len(df1['年月日']):
    if df1['證券代碼'][i+1] == df1['證券代碼'][i] and int(df1['年月日'][i+1][1]) == int(df1['年月日'][i][1]) and (int(df1['年月日'][i+1][2]) == (int(df1['年月日'][i][2])-1) or int(df1['年月日'][i+1][2]) == (int(df1['年月日'][i][2])-3)):
      df1['前一日合計買賣超(張)'][i] = df1['合計買賣超(張)'][i+1]
      df1['前一日最高價與收盤價價差'][i] = df1['最高價與收盤價價差'][i+1]
      df1['前一日最低價與收盤價價差'][i] = df1['最低價與收盤價價差'][i+1]
      df1['前一日成交量'][i] = df1['成交量(千股)'][i+1]
      df1['前一日漲跌停'][i] = df1['漲跌停'][i+1]
      df1['前一日高低價差%'][i] = df1['高低價差%'][i+1]
      df1['前一日處置股票(D)'][i] = df1['處置股票(D)'][i+1]
      df1['前一日現沖+當沖比重'][i] = df1['現沖+當沖比重'][i+1]
    else:
      df1['前一日合計買賣超(張)'][i] = None
      df1['前一日最高價與收盤價價差'][i] = None
      df1['前一日最低價與收盤價價差'][i] = None
      df1['前一日成交量'][i] = None
      df1['前一日漲跌停'][i] = None
      df1['前一日高低價差%'][i] = None
      df1['前一日處置股票(D)'][i] = None
      df1['前一日現沖+當沖比重'][i] = None

del df1['年月日']
del df1['最高價與收盤價價差']
del df1['最低價與收盤價價差']
del df1['成交量(千股)']
del df1['漲跌停']
del df1['高低價差%']
del df1['處置股票(D)']
del df1['合計買賣超(張)']
del df1['現沖+當沖比重']

# 將在外流通股數欄位併入df1，以計算交易量佔在外流通股數之比例
pysqldf = lambda q: sqldf(q, globals())
q = "SELECT a.*, b.流通股數 FROM df1 a LEFT JOIN df2 b ON a.證券代碼 = b.公司"
merge = pysqldf(q)

# dummy variable轉換
def dummy_ulimit(x):
  if x == '+':
    x = 1
  else:
    x = 0
  return x
def dummy_dlimit(x):
  if x == '-':
    x = 1
  else:
    x = 0
  return x
def dummy_d(x):
  if x == 'D':
    x = 1
  else:
    x = 0
  return x

# 以交易量比例創造新欄位，四捨五入到小數點第二位
merge['前一日成交量(比例)'] = round(((merge['前一日成交量'])/merge['流通股數']), 2)
merge['當日成交量(比例)'] = round(((merge['當日成交量'])/merge['流通股數']),2)
merge['前一日合計買賣超(萬分比例)'] = round((merge['前一日合計買賣超(張)']/(merge['前一日成交量']))*10000, 2)
merge['前一日高低價差%'] = round(merge['前一日高低價差%'],2)

del merge['當日成交量']
del merge['流通股數']
del merge['前一日成交量']
del merge['前一日合計買賣超(張)']

merge.dropna(axis=0, how='any', inplace=True)
```

## Model building
```
# 建立XY變數集
X = pd.DataFrame()
X['x1'] = merge['開盤價(元)']
X['x2'] = merge['前一日最高價與收盤價價差']
X['x3'] = merge['前一日最低價與收盤價價差']
X['x4'] = merge['前一日高低價差%']
X['x5'] = merge['前一日處置股票(D)']
X['x6'] = merge['前一日現沖+當沖比重']
X['x7'] = merge['前一日成交量(比例)']
X['x8'] = merge['前一日合計買賣超(萬分比例)']
X['x9'] = merge['前一日漲停']

Y = pd.DataFrame()
Y['y1'] = merge['當日成交量(比例)']

# 變數共線性檢視
def get_var_no_colinear(cutoff, df):
  corr_high = df.corr().applymap(lambda x: np.nan if x>cutoff else x).isnull()
  col_all = corr_high.columns.tolist()
  del_col = []
  i = 0
  while i < len(col_all)-1:
    ex_index = corr_high.iloc[:,i][i+1:].index[np.where(corr_high.iloc[:,i][i+1:])].tolist()
    for var in ex_index:
        col_all.remove(var)
    corr_high = corr_high.loc[col_all, col_all]
    i += 1
  return col_all
    
from statsmodels.stats.outliers_influence import variance_inflation_factor
def vif(X, thres=10.0):
  col = list(range(X.shape[1]))
  dropped = True
  while dropped:
    dropped = False
    vif = [variance_inflation_factor(X.iloc[:,col].values, ix)
            for ix in range(X.iloc[:,col].shape[1])]
    
    maxvif = max(vif)
    maxix = vif.index(maxvif)
    if maxvif > thres:
      del col[maxix]
      print('delete=',X.columns[col[maxix]],'  ', 'vif=',maxvif )
      dropped = True
  print('Remain Variables:', list(X.columns[col]))
  print('VIF:', vif)
  return list(X.columns[col])

vif(X, thres=10.0)

# 建立模型
model = sm.OLS(Y, X).fit()
print(model.summary())
```

## Validation
```
# 匯入資料並將其整理為前面X,Y相同格式
coef = np.array(model.params)
merge['predict'] = np.dot(np.array(X), coef)
merge['residual'] = merge['當日成交量(比例)'] - merge['predict']
merge.head(5)
```

![](https://i.imgur.com/bsCgNgI.png)
![](https://i.imgur.com/V0nXF2U.png)

```
# 觀察個別股票交易量預測表現
q = "SELECT DISTINCT 證券代碼 FROM merge"
stocks = pysqldf(q)
print(list(stocks['證券代碼']))
```

```
# 觀察 6706 惠特 預測表現
val = merge[merge['證券代碼'] == '6706 惠特']
pre = np.array(val['predict'])
rel = np.array(val['當日成交量(比例)'])
plt.figure(figsize=(12,5))
plt.plot(pre, label = 'Predicted')
plt.plot(rel, label = 'Real Trade Volume')
plt.legend(loc = 'best')
plt.title('Gap between predicted volume and real volume')
```
![](https://i.imgur.com/BxqHDKu.png)

```
# 觀察 6715 嘉基 預測表現→交易量較低預測表現較差
val = merge[merge['證券代碼'] == '6715 嘉基']
pre = np.array(val['predict'])
rel = np.array(val['當日成交量(比例)'])
plt.figure(figsize=(12,5))
plt.plot(pre, label = 'Predicted')
plt.plot(rel, label = 'Real Trade Volume')
plt.legend(loc = 'best')
plt.title('Gap between predicted volume and real volume')
```
![](https://i.imgur.com/wEkHud2.png)


```
# 觀察 8011 台通 預測表現
val = merge[merge['證券代碼'] == '8011 台通']
pre = np.array(val['predict'])
rel = np.array(val['當日成交量(比例)'])
plt.figure(figsize=(12,5))
plt.plot(pre, label = 'Predicted')
plt.plot(rel, label = 'Real Trade Volume')
plt.legend(loc = 'best')
plt.title('Gap between predicted volume and real volume')
```
![](https://i.imgur.com/NKXZ8vx.png)

```
# 觀察 8044 網家 預測表現
val = merge[merge['證券代碼'] == '8044 網家']
pre = np.array(val['predict'])
rel = np.array(val['當日成交量(比例)'])
plt.figure(figsize=(12,5))
plt.plot(pre, label = 'Predicted')
plt.plot(rel, label = 'Real Trade Volume')
plt.legend(loc = 'best')
plt.title('Gap between predicted volume and real volume')
```
![](https://i.imgur.com/3CtfcmG.png)

## Conclusions
* Larger trade volume has better prediction
* R-square:  0.616
* Prob (F-statistic): 0.00

![](https://i.imgur.com/0jJUuCs.png)
![](https://i.imgur.com/JH8tQja.png)
![](https://i.imgur.com/6Y17x8K.png)
![](https://i.imgur.com/MPV6GRU.png)
