```python
import pandas_datareader.data as web
import pandas as pd
import datetime
import tweepy
from textblob import TextBlob
import re
import numpy as np
from matplotlib import pyplot as plt
from pandas.plotting import scatter_matrix
from mpl_finance import candlestick_ohlc
from matplotlib.dates import DateFormatter, date2num, WeekdayLocator, DayLocator, MONDAY

```

    /Users/tylerbrown/opt/anaconda3/lib/python3.9/site-packages/mpl_finance.py:16: DeprecationWarning: 
    
      =================================================================
    
       WARNING: `mpl_finance` is deprecated:
    
        Please use `mplfinance` instead (no hyphen, no underscore).
    
        To install: `pip install --upgrade mplfinance` 
    
       For more information, see: https://pypi.org/project/mplfinance/
    
      =================================================================
    
      __warnings.warn('\n\n  ================================================================='+



```python
#Setting start and end dates
start = datetime.datetime(2020,1,1)
end = datetime.datetime(2022,10,15)
```


```python
#Reading in amazon and tesla history
Tesla = web.DataReader("TSLA", 'yahoo', start, end )
Amazon = web.DataReader("AMZN", 'yahoo', start, end )
```


```python
#Looking at top 5 most recent observations 
Amazon.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>High</th>
      <th>Low</th>
      <th>Open</th>
      <th>Close</th>
      <th>Volume</th>
      <th>Adj Close</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-01-02</th>
      <td>94.900497</td>
      <td>93.207497</td>
      <td>93.750000</td>
      <td>94.900497</td>
      <td>80580000.0</td>
      <td>94.900497</td>
    </tr>
    <tr>
      <th>2020-01-03</th>
      <td>94.309998</td>
      <td>93.224998</td>
      <td>93.224998</td>
      <td>93.748497</td>
      <td>75288000.0</td>
      <td>93.748497</td>
    </tr>
    <tr>
      <th>2020-01-06</th>
      <td>95.184502</td>
      <td>93.000000</td>
      <td>93.000000</td>
      <td>95.143997</td>
      <td>81236000.0</td>
      <td>95.143997</td>
    </tr>
    <tr>
      <th>2020-01-07</th>
      <td>95.694504</td>
      <td>94.601997</td>
      <td>95.224998</td>
      <td>95.343002</td>
      <td>80898000.0</td>
      <td>95.343002</td>
    </tr>
    <tr>
      <th>2020-01-08</th>
      <td>95.550003</td>
      <td>94.321999</td>
      <td>94.902000</td>
      <td>94.598503</td>
      <td>70160000.0</td>
      <td>94.598503</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Tesla head
Tesla.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>High</th>
      <th>Low</th>
      <th>Open</th>
      <th>Close</th>
      <th>Volume</th>
      <th>Adj Close</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-01-02</th>
      <td>28.713333</td>
      <td>28.114000</td>
      <td>28.299999</td>
      <td>28.684000</td>
      <td>142981500.0</td>
      <td>28.684000</td>
    </tr>
    <tr>
      <th>2020-01-03</th>
      <td>30.266666</td>
      <td>29.128000</td>
      <td>29.366667</td>
      <td>29.534000</td>
      <td>266677500.0</td>
      <td>29.534000</td>
    </tr>
    <tr>
      <th>2020-01-06</th>
      <td>30.104000</td>
      <td>29.333332</td>
      <td>29.364668</td>
      <td>30.102667</td>
      <td>151995000.0</td>
      <td>30.102667</td>
    </tr>
    <tr>
      <th>2020-01-07</th>
      <td>31.441999</td>
      <td>30.224001</td>
      <td>30.760000</td>
      <td>31.270666</td>
      <td>268231500.0</td>
      <td>31.270666</td>
    </tr>
    <tr>
      <th>2020-01-08</th>
      <td>33.232666</td>
      <td>31.215334</td>
      <td>31.580000</td>
      <td>32.809334</td>
      <td>467164500.0</td>
      <td>32.809334</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Plitting out a time series of open and close
Tesla['Open'].plot(label = 'TSLA Open price' )
Tesla['Close'].plot(label = 'TSLA Close price' )
Amazon['Open'].plot(label = 'Amazon price')
Amazon['Close'].plot(label = 'Amazon Low price')
plt.yscale("log")
plt.legend()
plt.ylabel('Stock Price')
plt.show()
```


    
![png](output_5_0.png)
    



```python
#Volume histogram
Tesla['Volume'].plot.hist(bins = 40)
plt.show()
```


    
![png](output_6_0.png)
    



```python
#Volume histogram
Amazon['Volume'].plot.hist(bins = 40)
plt.show()
```


    
![png](output_7_0.png)
    



```python
#Examing the volume of trades 
Amazon['Volume'].plot(label='Amazon',figsize=(15,7))
Tesla['Volume'].plot(label='Tesla')
plt.ylabel('Volume')
plt.legend()
plt.show()
```


    
![png](output_8_0.png)
    



```python
len(Amazon)
```




    703




```python
#Past 100 days of trading
Amazon.iloc[603:703].plot()
```




    <AxesSubplot:xlabel='Date'>




    
![png](output_10_1.png)
    



```python
#Past 100 days of trading
Tesla.iloc[603:703].plot()
```




    <AxesSubplot:xlabel='Date'>




    
![png](output_11_1.png)
    



```python
#Total traded
Tesla['Total Traded']= Tesla['Open'] * Tesla['Volume']
Amazon['Total Traded']= Amazon['Open'] * Amazon['Volume']
Tesla.head()
Bitcoin.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>High</th>
      <th>Low</th>
      <th>Open</th>
      <th>Close</th>
      <th>Volume</th>
      <th>Adj Close</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-01-02</th>
      <td>94.900497</td>
      <td>93.207497</td>
      <td>93.750000</td>
      <td>94.900497</td>
      <td>80580000</td>
      <td>94.900497</td>
    </tr>
    <tr>
      <th>2020-01-03</th>
      <td>94.309998</td>
      <td>93.224998</td>
      <td>93.224998</td>
      <td>93.748497</td>
      <td>75288000</td>
      <td>93.748497</td>
    </tr>
    <tr>
      <th>2020-01-06</th>
      <td>95.184502</td>
      <td>93.000000</td>
      <td>93.000000</td>
      <td>95.143997</td>
      <td>81236000</td>
      <td>95.143997</td>
    </tr>
    <tr>
      <th>2020-01-07</th>
      <td>95.694504</td>
      <td>94.601997</td>
      <td>95.224998</td>
      <td>95.343002</td>
      <td>80898000</td>
      <td>95.343002</td>
    </tr>
    <tr>
      <th>2020-01-08</th>
      <td>95.550003</td>
      <td>94.321999</td>
      <td>94.902000</td>
      <td>94.598503</td>
      <td>70160000</td>
      <td>94.598503</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Time series of total money traded using log scale
Tesla['Total Traded'].plot(label= 'Tesla')
Amazon['Total Traded'].plot(label = 'Amazon', figsize = (15,7))
plt.ylabel('Total Traded')
plt.legend()
plt.show()
```


    
![png](output_13_0.png)
    



```python
#Highest trading day
highest_trade_day = { 'Tesla': Tesla.iloc[[Tesla['Total Traded'].argmax()]],
                    'Amazon': Amazon.iloc[[Amazon['Total Traded'].argmax()]]}
highest_trade_day
```




    {'Tesla':                   High         Low       Open       Close       Volume  \
     Date                                                                     
     2020-12-18  231.666672  209.513336  222.96666  231.666672  666378600.0   
     
                  Adj Close  Total Traded  
     Date                                  
     2020-12-18  231.666672  1.485802e+11  ,
     'Amazon':                   High         Low        Open       Close       Volume  \
     Date                                                                      
     2022-02-04  161.199997  150.608002  155.606506  157.639496  253456000.0   
     
                  Adj Close  Total Traded  
     Date                                  
     2022-02-04  157.639496  3.943940e+10  }




```python
#Price at open comp
Tesla['Open'].plot(figsize = (15,7))
Amazon['Open'].plot(figsize = (15,7))
plt.title('Price at Open ')
```




    Text(0.5, 1.0, 'Price at Open ')




    
![png](output_15_1.png)
    



```python
#Rolling average for amazon 
Amazon['Open'].plot(figsize = (15,7))
Amazon['MA50'] = Amazon['Open'].rolling(100).mean()
Amazon['MA50'].plot(label = 'MA50')
plt.title('Rolling average for Amazon')

plt.legend()
```




    <matplotlib.legend.Legend at 0x7f8f69484760>




    
![png](output_16_1.png)
    



```python
#Rolling average for Tesla
Tesla['Open'].plot(figsize = (15,7))
Tesla['MA50'] = Tesla['Open'].rolling(100).mean()
Tesla['MA50'].plot(label = 'MA50')
plt.title('Rolling average for Tesla')
plt.legend()
```




    <matplotlib.legend.Legend at 0x7f8f898977f0>




    
![png](output_17_1.png)
    



```python
#Concating for scatter matrix
comp = pd.concat([Tesla['Open'], Amazon['Open']], axis = 1)
comp.columns= ['Tesla Open', 'Amazon Open']
```


```python
#Scatter 
scatter_matrix(comp, figsize = (8,8), hist_kwds = {'bins': 50} )

```




    array([[<AxesSubplot:xlabel='Tesla Open', ylabel='Tesla Open'>,
            <AxesSubplot:xlabel='Amazon Open', ylabel='Tesla Open'>],
           [<AxesSubplot:xlabel='Tesla Open', ylabel='Amazon Open'>,
            <AxesSubplot:xlabel='Amazon Open', ylabel='Amazon Open'>]],
          dtype=object)




    
![png](output_19_1.png)
    



```python
#Candlestick chart
TeslaReset = Tesla.loc['2020-01':'2022-01'].reset_index()
TeslaReset['date_ax'] = TeslaReset['Date'].apply(lambda date: date2num(date))
Tesla_values = [tuple(vals) for vals in TeslaReset[['date_ax', 'Open', 'High', 'Low', 'Close']].values]

mondays = WeekdayLocator(MONDAY)
alldays = DayLocator()
weekFormatter = DateFormatter('%b %d')
dayFormatter = DateFormatter('%d')

fig, ax = plt.subplots()
candlestick_ohlc(ax, Tesla_values, width = 0.6, colorup = 'g', colordown = 'r' )


```




    ([<matplotlib.lines.Line2D at 0x7f8f69817eb0>,
      <matplotlib.lines.Line2D at 0x7f8f58388220>,
      <matplotlib.lines.Line2D at 0x7f8f58388700>,
      <matplotlib.lines.Line2D at 0x7f8f58388be0>,
      <matplotlib.lines.Line2D at 0x7f8f7d8b7040>,
      <matplotlib.lines.Line2D at 0x7f8f7d8b75e0>,
      <matplotlib.lines.Line2D at 0x7f8f7d8b7ac0>,
      <matplotlib.lines.Line2D at 0x7f8f7d8b7fa0>,
      <matplotlib.lines.Line2D at 0x7f8f7d8c44c0>,
      <matplotlib.lines.Line2D at 0x7f8f7d8c49a0>,
      <matplotlib.lines.Line2D at 0x7f8f7d8c4e80>,
      <matplotlib.lines.Line2D at 0x7f8f7d8d13a0>,
      <matplotlib.lines.Line2D at 0x7f8f7d8d1880>,
      <matplotlib.lines.Line2D at 0x7f8f7d8d1d60>,
      <matplotlib.lines.Line2D at 0x7f8f7d8de280>,
      <matplotlib.lines.Line2D at 0x7f8f7d8de760>,
      <matplotlib.lines.Line2D at 0x7f8f7d8dec40>,
      <matplotlib.lines.Line2D at 0x7f8f7d8ec160>,
      <matplotlib.lines.Line2D at 0x7f8f7d8ec640>,
      <matplotlib.lines.Line2D at 0x7f8f7d8ecb20>,
      <matplotlib.lines.Line2D at 0x7f8f7d8ecfa0>,
      <matplotlib.lines.Line2D at 0x7f8f7d8fa520>,
      <matplotlib.lines.Line2D at 0x7f8f7d8faa00>,
      <matplotlib.lines.Line2D at 0x7f8f7d8faee0>,
      <matplotlib.lines.Line2D at 0x7f8f7d907400>,
      <matplotlib.lines.Line2D at 0x7f8f7d9078e0>,
      <matplotlib.lines.Line2D at 0x7f8f7d907dc0>,
      <matplotlib.lines.Line2D at 0x7f8f7d9142e0>,
      <matplotlib.lines.Line2D at 0x7f8f7d9147c0>,
      <matplotlib.lines.Line2D at 0x7f8f7d914ca0>,
      <matplotlib.lines.Line2D at 0x7f8f7d9221c0>,
      <matplotlib.lines.Line2D at 0x7f8f7d9226a0>,
      <matplotlib.lines.Line2D at 0x7f8f7d922b80>,
      <matplotlib.lines.Line2D at 0x7f8f7d922fd0>,
      <matplotlib.lines.Line2D at 0x7f8f7d92f580>,
      <matplotlib.lines.Line2D at 0x7f8f7d92fa60>,
      <matplotlib.lines.Line2D at 0x7f8f7d92ff40>,
      <matplotlib.lines.Line2D at 0x7f8f7d96d460>,
      <matplotlib.lines.Line2D at 0x7f8f7d96d940>,
      <matplotlib.lines.Line2D at 0x7f8f7d96de20>,
      <matplotlib.lines.Line2D at 0x7f8f7d97b340>,
      <matplotlib.lines.Line2D at 0x7f8f7d97b820>,
      <matplotlib.lines.Line2D at 0x7f8f7d97bd00>,
      <matplotlib.lines.Line2D at 0x7f8f7d988220>,
      <matplotlib.lines.Line2D at 0x7f8f7d988700>,
      <matplotlib.lines.Line2D at 0x7f8f7d988be0>,
      <matplotlib.lines.Line2D at 0x7f8f7d997040>,
      <matplotlib.lines.Line2D at 0x7f8f7d9975e0>,
      <matplotlib.lines.Line2D at 0x7f8f7d997ac0>,
      <matplotlib.lines.Line2D at 0x7f8f7d997fa0>,
      <matplotlib.lines.Line2D at 0x7f8f7d9a54c0>,
      <matplotlib.lines.Line2D at 0x7f8f7d9a59a0>,
      <matplotlib.lines.Line2D at 0x7f8f7d9a5e80>,
      <matplotlib.lines.Line2D at 0x7f8f7d9b03a0>,
      <matplotlib.lines.Line2D at 0x7f8f7d9b0880>,
      <matplotlib.lines.Line2D at 0x7f8f7d9b0d60>,
      <matplotlib.lines.Line2D at 0x7f8f7d9be280>,
      <matplotlib.lines.Line2D at 0x7f8f7d9be760>,
      <matplotlib.lines.Line2D at 0x7f8f7d9bec40>,
      <matplotlib.lines.Line2D at 0x7f8f7d9ce160>,
      <matplotlib.lines.Line2D at 0x7f8f7d9ce640>,
      <matplotlib.lines.Line2D at 0x7f8f7d9ceb20>,
      <matplotlib.lines.Line2D at 0x7f8f7d9cefa0>,
      <matplotlib.lines.Line2D at 0x7f8f7d9db520>,
      <matplotlib.lines.Line2D at 0x7f8f7d9dba00>,
      <matplotlib.lines.Line2D at 0x7f8f7d9dbee0>,
      <matplotlib.lines.Line2D at 0x7f8f7d9e8400>,
      <matplotlib.lines.Line2D at 0x7f8f7d9e88e0>,
      <matplotlib.lines.Line2D at 0x7f8f7d9e8dc0>,
      <matplotlib.lines.Line2D at 0x7f8f7d9f52e0>,
      <matplotlib.lines.Line2D at 0x7f8f7d9f57c0>,
      <matplotlib.lines.Line2D at 0x7f8f7d9f5ca0>,
      <matplotlib.lines.Line2D at 0x7f8f7da021c0>,
      <matplotlib.lines.Line2D at 0x7f8f7da026a0>,
      <matplotlib.lines.Line2D at 0x7f8f7da02b80>,
      <matplotlib.lines.Line2D at 0x7f8f7da02fd0>,
      <matplotlib.lines.Line2D at 0x7f8f7da13580>,
      <matplotlib.lines.Line2D at 0x7f8f7da13a60>,
      <matplotlib.lines.Line2D at 0x7f8f7da13f40>,
      <matplotlib.lines.Line2D at 0x7f8f7da1e460>,
      <matplotlib.lines.Line2D at 0x7f8f7da1e940>,
      <matplotlib.lines.Line2D at 0x7f8f7da1ee20>,
      <matplotlib.lines.Line2D at 0x7f8f7da2c340>,
      <matplotlib.lines.Line2D at 0x7f8f7da2c820>,
      <matplotlib.lines.Line2D at 0x7f8f7da2cd00>,
      <matplotlib.lines.Line2D at 0x7f8f7da3b220>,
      <matplotlib.lines.Line2D at 0x7f8f7da3b700>,
      <matplotlib.lines.Line2D at 0x7f8f7da3bbe0>,
      <matplotlib.lines.Line2D at 0x7f8f7da4a040>,
      <matplotlib.lines.Line2D at 0x7f8f7da4a5e0>,
      <matplotlib.lines.Line2D at 0x7f8f7da4aac0>,
      <matplotlib.lines.Line2D at 0x7f8f7da4afa0>,
      <matplotlib.lines.Line2D at 0x7f8f7da554c0>,
      <matplotlib.lines.Line2D at 0x7f8f7da559a0>,
      <matplotlib.lines.Line2D at 0x7f8f7da55e80>,
      <matplotlib.lines.Line2D at 0x7f8f7da623a0>,
      <matplotlib.lines.Line2D at 0x7f8f7da62880>,
      <matplotlib.lines.Line2D at 0x7f8f7da62d60>,
      <matplotlib.lines.Line2D at 0x7f8f7da70280>,
      <matplotlib.lines.Line2D at 0x7f8f7da70760>,
      <matplotlib.lines.Line2D at 0x7f8f7da70c40>,
      <matplotlib.lines.Line2D at 0x7f8f7da7e160>,
      <matplotlib.lines.Line2D at 0x7f8f7da7e640>,
      <matplotlib.lines.Line2D at 0x7f8f7da7eb20>,
      <matplotlib.lines.Line2D at 0x7f8f7da7efa0>,
      <matplotlib.lines.Line2D at 0x7f8f7da8d520>,
      <matplotlib.lines.Line2D at 0x7f8f7da8da00>,
      <matplotlib.lines.Line2D at 0x7f8f7da8dee0>,
      <matplotlib.lines.Line2D at 0x7f8f7da98430>,
      <matplotlib.lines.Line2D at 0x7f8f7da98910>,
      <matplotlib.lines.Line2D at 0x7f8f7da98df0>,
      <matplotlib.lines.Line2D at 0x7f8f7daa7310>,
      <matplotlib.lines.Line2D at 0x7f8f7daa77f0>,
      <matplotlib.lines.Line2D at 0x7f8f7daa7cd0>,
      <matplotlib.lines.Line2D at 0x7f8f7dab51f0>,
      <matplotlib.lines.Line2D at 0x7f8f7dab56d0>,
      <matplotlib.lines.Line2D at 0x7f8f7dab5bb0>,
      <matplotlib.lines.Line2D at 0x7f8f7dab5fd0>,
      <matplotlib.lines.Line2D at 0x7f8f7dac35b0>,
      <matplotlib.lines.Line2D at 0x7f8f7dac3a90>,
      <matplotlib.lines.Line2D at 0x7f8f7dac3f70>,
      <matplotlib.lines.Line2D at 0x7f8f7dad0490>,
      <matplotlib.lines.Line2D at 0x7f8f7dad0970>,
      <matplotlib.lines.Line2D at 0x7f8f7dad0e50>,
      <matplotlib.lines.Line2D at 0x7f8f7dadd370>,
      <matplotlib.lines.Line2D at 0x7f8f7dadd850>,
      <matplotlib.lines.Line2D at 0x7f8f7daddd30>,
      <matplotlib.lines.Line2D at 0x7f8f7daeb250>,
      <matplotlib.lines.Line2D at 0x7f8f7daeb730>,
      <matplotlib.lines.Line2D at 0x7f8f7daebc10>,
      <matplotlib.lines.Line2D at 0x7f8f7daf9130>,
      <matplotlib.lines.Line2D at 0x7f8f7daf9610>,
      <matplotlib.lines.Line2D at 0x7f8f7daf9af0>,
      <matplotlib.lines.Line2D at 0x7f8f7daf9fd0>,
      <matplotlib.lines.Line2D at 0x7f8f7db064f0>,
      <matplotlib.lines.Line2D at 0x7f8f7db069d0>,
      <matplotlib.lines.Line2D at 0x7f8f7db06eb0>,
      <matplotlib.lines.Line2D at 0x7f8f7db123d0>,
      <matplotlib.lines.Line2D at 0x7f8f7db128b0>,
      <matplotlib.lines.Line2D at 0x7f8f7db12d90>,
      <matplotlib.lines.Line2D at 0x7f8f7db222b0>,
      <matplotlib.lines.Line2D at 0x7f8f7db22790>,
      <matplotlib.lines.Line2D at 0x7f8f7db22c70>,
      <matplotlib.lines.Line2D at 0x7f8f7db2f190>,
      <matplotlib.lines.Line2D at 0x7f8f7db2f670>,
      <matplotlib.lines.Line2D at 0x7f8f7db2fb50>,
      <matplotlib.lines.Line2D at 0x7f8f7db2ffd0>,
      <matplotlib.lines.Line2D at 0x7f8f7db3e550>,
      <matplotlib.lines.Line2D at 0x7f8f7db3ea30>,
      <matplotlib.lines.Line2D at 0x7f8f7db3ef10>,
      <matplotlib.lines.Line2D at 0x7f8f7db4b430>,
      <matplotlib.lines.Line2D at 0x7f8f7db4b910>,
      <matplotlib.lines.Line2D at 0x7f8f7db4bdf0>,
      <matplotlib.lines.Line2D at 0x7f8f7db56310>,
      <matplotlib.lines.Line2D at 0x7f8f7db567f0>,
      <matplotlib.lines.Line2D at 0x7f8f7db56cd0>,
      <matplotlib.lines.Line2D at 0x7f8f7dcde1f0>,
      <matplotlib.lines.Line2D at 0x7f8f7dcde6d0>,
      <matplotlib.lines.Line2D at 0x7f8f7dcdebb0>,
      <matplotlib.lines.Line2D at 0x7f8f7dcdefd0>,
      <matplotlib.lines.Line2D at 0x7f8f7dcec5b0>,
      <matplotlib.lines.Line2D at 0x7f8f7dceca90>,
      <matplotlib.lines.Line2D at 0x7f8f7dcecf70>,
      <matplotlib.lines.Line2D at 0x7f8f7dcf8490>,
      <matplotlib.lines.Line2D at 0x7f8f7dcf8970>,
      <matplotlib.lines.Line2D at 0x7f8f7dcf8e50>,
      <matplotlib.lines.Line2D at 0x7f8f7dd06370>,
      <matplotlib.lines.Line2D at 0x7f8f7dd06850>,
      <matplotlib.lines.Line2D at 0x7f8f7dd06d30>,
      <matplotlib.lines.Line2D at 0x7f8f7dd13250>,
      <matplotlib.lines.Line2D at 0x7f8f7dd13730>,
      <matplotlib.lines.Line2D at 0x7f8f7dd13c10>,
      <matplotlib.lines.Line2D at 0x7f8f7dd21130>,
      <matplotlib.lines.Line2D at 0x7f8f7dd21610>,
      <matplotlib.lines.Line2D at 0x7f8f7dd21af0>,
      <matplotlib.lines.Line2D at 0x7f8f7dd21fd0>,
      <matplotlib.lines.Line2D at 0x7f8f7dd304f0>,
      <matplotlib.lines.Line2D at 0x7f8f7dd309d0>,
      <matplotlib.lines.Line2D at 0x7f8f7dd30eb0>,
      <matplotlib.lines.Line2D at 0x7f8f7dd3d3d0>,
      <matplotlib.lines.Line2D at 0x7f8f7dd3d8b0>,
      <matplotlib.lines.Line2D at 0x7f8f7dd3dd90>,
      <matplotlib.lines.Line2D at 0x7f8f7dd4a2b0>,
      <matplotlib.lines.Line2D at 0x7f8f7dd4a790>,
      <matplotlib.lines.Line2D at 0x7f8f7dd4ac70>,
      <matplotlib.lines.Line2D at 0x7f8f7dd5a190>,
      <matplotlib.lines.Line2D at 0x7f8f7dd5a670>,
      <matplotlib.lines.Line2D at 0x7f8f7dd5ab50>,
      <matplotlib.lines.Line2D at 0x7f8f7dd5afd0>,
      <matplotlib.lines.Line2D at 0x7f8f7dd66550>,
      <matplotlib.lines.Line2D at 0x7f8f7dd66a30>,
      <matplotlib.lines.Line2D at 0x7f8f7dd66f10>,
      <matplotlib.lines.Line2D at 0x7f8f7dd74430>,
      <matplotlib.lines.Line2D at 0x7f8f7dd74910>,
      <matplotlib.lines.Line2D at 0x7f8f7dd74df0>,
      <matplotlib.lines.Line2D at 0x7f8f7dd81310>,
      <matplotlib.lines.Line2D at 0x7f8f7dd817f0>,
      <matplotlib.lines.Line2D at 0x7f8f7dd81cd0>,
      <matplotlib.lines.Line2D at 0x7f8f7dd8d1f0>,
      <matplotlib.lines.Line2D at 0x7f8f7dd8d6d0>,
      <matplotlib.lines.Line2D at 0x7f8f7dd8dbb0>,
      <matplotlib.lines.Line2D at 0x7f8f7dd8dfd0>,
      <matplotlib.lines.Line2D at 0x7f8f7dd9d5b0>,
      <matplotlib.lines.Line2D at 0x7f8f7dd9da90>,
      <matplotlib.lines.Line2D at 0x7f8f7dd9df70>,
      <matplotlib.lines.Line2D at 0x7f8f7dda8490>,
      <matplotlib.lines.Line2D at 0x7f8f7dda8970>,
      <matplotlib.lines.Line2D at 0x7f8f7dda8e50>,
      <matplotlib.lines.Line2D at 0x7f8f7ddb8370>,
      <matplotlib.lines.Line2D at 0x7f8f7ddb8850>,
      <matplotlib.lines.Line2D at 0x7f8f7ddb8d30>,
      <matplotlib.lines.Line2D at 0x7f8f7ddc5250>,
      <matplotlib.lines.Line2D at 0x7f8f7ddc5730>,
      <matplotlib.lines.Line2D at 0x7f8f7ddc5c40>,
      <matplotlib.lines.Line2D at 0x7f8f7ddd3160>,
      <matplotlib.lines.Line2D at 0x7f8f7ddd3640>,
      <matplotlib.lines.Line2D at 0x7f8f7ddd3b20>,
      <matplotlib.lines.Line2D at 0x7f8f7ddd3fa0>,
      <matplotlib.lines.Line2D at 0x7f8f7dde0520>,
      <matplotlib.lines.Line2D at 0x7f8f7dde0a00>,
      <matplotlib.lines.Line2D at 0x7f8f7dde0ee0>,
      <matplotlib.lines.Line2D at 0x7f8f7ddee400>,
      <matplotlib.lines.Line2D at 0x7f8f7ddee8e0>,
      <matplotlib.lines.Line2D at 0x7f8f7ddeedc0>,
      <matplotlib.lines.Line2D at 0x7f8f7ddfb2e0>,
      <matplotlib.lines.Line2D at 0x7f8f7ddfb7c0>,
      <matplotlib.lines.Line2D at 0x7f8f7ddfbca0>,
      <matplotlib.lines.Line2D at 0x7f8f7de091c0>,
      <matplotlib.lines.Line2D at 0x7f8f7de096a0>,
      <matplotlib.lines.Line2D at 0x7f8f7de09b80>,
      <matplotlib.lines.Line2D at 0x7f8f7de09fd0>,
      <matplotlib.lines.Line2D at 0x7f8f7de17580>,
      <matplotlib.lines.Line2D at 0x7f8f7de17a60>,
      <matplotlib.lines.Line2D at 0x7f8f7de17f40>,
      <matplotlib.lines.Line2D at 0x7f8f7de26460>,
      <matplotlib.lines.Line2D at 0x7f8f7de26940>,
      <matplotlib.lines.Line2D at 0x7f8f7de26e20>,
      <matplotlib.lines.Line2D at 0x7f8f7de33340>,
      <matplotlib.lines.Line2D at 0x7f8f7de33820>,
      <matplotlib.lines.Line2D at 0x7f8f7de33d00>,
      <matplotlib.lines.Line2D at 0x7f8f7de40220>,
      <matplotlib.lines.Line2D at 0x7f8f7de40700>,
      <matplotlib.lines.Line2D at 0x7f8f7de40be0>,
      <matplotlib.lines.Line2D at 0x7f8f7de4d100>,
      <matplotlib.lines.Line2D at 0x7f8f7de4d5e0>,
      <matplotlib.lines.Line2D at 0x7f8f7de4dac0>,
      <matplotlib.lines.Line2D at 0x7f8f7de4dfa0>,
      <matplotlib.lines.Line2D at 0x7f8f7de5b4c0>,
      <matplotlib.lines.Line2D at 0x7f8f7de5b9a0>,
      <matplotlib.lines.Line2D at 0x7f8f7de5be80>,
      <matplotlib.lines.Line2D at 0x7f8f7de693a0>,
      <matplotlib.lines.Line2D at 0x7f8f7de69880>,
      <matplotlib.lines.Line2D at 0x7f8f7de69d60>,
      <matplotlib.lines.Line2D at 0x7f8f7de76280>,
      <matplotlib.lines.Line2D at 0x7f8f7de76760>,
      <matplotlib.lines.Line2D at 0x7f8f7de76c40>,
      <matplotlib.lines.Line2D at 0x7f8f7de83160>,
      <matplotlib.lines.Line2D at 0x7f8f7de83640>,
      <matplotlib.lines.Line2D at 0x7f8f7de83b20>,
      <matplotlib.lines.Line2D at 0x7f8f7de83fa0>,
      <matplotlib.lines.Line2D at 0x7f8f7de92520>,
      <matplotlib.lines.Line2D at 0x7f8f7de92a00>,
      <matplotlib.lines.Line2D at 0x7f8f7de92ee0>,
      <matplotlib.lines.Line2D at 0x7f8f7de9e400>,
      <matplotlib.lines.Line2D at 0x7f8f7de9e8e0>,
      <matplotlib.lines.Line2D at 0x7f8f7de9edc0>,
      <matplotlib.lines.Line2D at 0x7f8f7dead2e0>,
      <matplotlib.lines.Line2D at 0x7f8f7dead7c0>,
      <matplotlib.lines.Line2D at 0x7f8f7deadca0>,
      <matplotlib.lines.Line2D at 0x7f8f7debb1c0>,
      <matplotlib.lines.Line2D at 0x7f8f7debb6a0>,
      <matplotlib.lines.Line2D at 0x7f8f7debbb80>,
      <matplotlib.lines.Line2D at 0x7f8f7debbfd0>,
      <matplotlib.lines.Line2D at 0x7f8f7dec9580>,
      <matplotlib.lines.Line2D at 0x7f8f7dec9a60>,
      <matplotlib.lines.Line2D at 0x7f8f7dec9f40>,
      <matplotlib.lines.Line2D at 0x7f8f7ded7490>,
      <matplotlib.lines.Line2D at 0x7f8f7ded7970>,
      <matplotlib.lines.Line2D at 0x7f8f7ded7e50>,
      <matplotlib.lines.Line2D at 0x7f8f7dee2370>,
      <matplotlib.lines.Line2D at 0x7f8f7dee2880>,
      <matplotlib.lines.Line2D at 0x7f8f7dee2d60>,
      <matplotlib.lines.Line2D at 0x7f8f7def1280>,
      <matplotlib.lines.Line2D at 0x7f8f7def1760>,
      <matplotlib.lines.Line2D at 0x7f8f7def1c40>,
      <matplotlib.lines.Line2D at 0x7f8f7df00160>,
      <matplotlib.lines.Line2D at 0x7f8f7df00640>,
      <matplotlib.lines.Line2D at 0x7f8f7df00b20>,
      <matplotlib.lines.Line2D at 0x7f8f7df00fa0>,
      <matplotlib.lines.Line2D at 0x7f8f7df0d520>,
      <matplotlib.lines.Line2D at 0x7f8f7df0da00>,
      <matplotlib.lines.Line2D at 0x7f8f7df0dee0>,
      <matplotlib.lines.Line2D at 0x7f8f7df1a400>,
      <matplotlib.lines.Line2D at 0x7f8f7df1a8e0>,
      <matplotlib.lines.Line2D at 0x7f8f7df1adc0>,
      <matplotlib.lines.Line2D at 0x7f8f7df262e0>,
      <matplotlib.lines.Line2D at 0x7f8f7df267c0>,
      <matplotlib.lines.Line2D at 0x7f8f7df26ca0>,
      <matplotlib.lines.Line2D at 0x7f8f7df351c0>,
      <matplotlib.lines.Line2D at 0x7f8f7df356a0>,
      <matplotlib.lines.Line2D at 0x7f8f7df35b80>,
      <matplotlib.lines.Line2D at 0x7f8f7df35fd0>,
      <matplotlib.lines.Line2D at 0x7f8f7df43580>,
      <matplotlib.lines.Line2D at 0x7f8f7df43a60>,
      <matplotlib.lines.Line2D at 0x7f8f7df43f40>,
      <matplotlib.lines.Line2D at 0x7f8f7df50460>,
      <matplotlib.lines.Line2D at 0x7f8f7df50940>,
      <matplotlib.lines.Line2D at 0x7f8f7df50e20>,
      <matplotlib.lines.Line2D at 0x7f8f7df5d340>,
      <matplotlib.lines.Line2D at 0x7f8f7df5d820>,
      <matplotlib.lines.Line2D at 0x7f8f7df5dd00>,
      <matplotlib.lines.Line2D at 0x7f8f7df6c220>,
      <matplotlib.lines.Line2D at 0x7f8f7df6c700>,
      <matplotlib.lines.Line2D at 0x7f8f7df6cbe0>,
      <matplotlib.lines.Line2D at 0x7f8f7df79040>,
      <matplotlib.lines.Line2D at 0x7f8f7df795e0>,
      <matplotlib.lines.Line2D at 0x7f8f7df79ac0>,
      <matplotlib.lines.Line2D at 0x7f8f7df79fa0>,
      <matplotlib.lines.Line2D at 0x7f8f7df864c0>,
      <matplotlib.lines.Line2D at 0x7f8f7df869a0>,
      <matplotlib.lines.Line2D at 0x7f8f7df86e80>,
      <matplotlib.lines.Line2D at 0x7f8f7df933a0>,
      <matplotlib.lines.Line2D at 0x7f8f7df93880>,
      <matplotlib.lines.Line2D at 0x7f8f7df93d60>,
      <matplotlib.lines.Line2D at 0x7f8f7dfa2280>,
      <matplotlib.lines.Line2D at 0x7f8f7dfa2760>,
      <matplotlib.lines.Line2D at 0x7f8f7dfa2c40>,
      <matplotlib.lines.Line2D at 0x7f8f7dfb1160>,
      <matplotlib.lines.Line2D at 0x7f8f7dfb1640>,
      <matplotlib.lines.Line2D at 0x7f8f7dfb1b20>,
      <matplotlib.lines.Line2D at 0x7f8f7dfb1fa0>,
      <matplotlib.lines.Line2D at 0x7f8f7dfbd520>,
      <matplotlib.lines.Line2D at 0x7f8f7dfbda00>,
      <matplotlib.lines.Line2D at 0x7f8f7dfbdee0>,
      <matplotlib.lines.Line2D at 0x7f8f7dfc9400>,
      <matplotlib.lines.Line2D at 0x7f8f7dfc98e0>,
      <matplotlib.lines.Line2D at 0x7f8f7dfc9dc0>,
      <matplotlib.lines.Line2D at 0x7f8f7dfd92e0>,
      <matplotlib.lines.Line2D at 0x7f8f7dfd97c0>,
      <matplotlib.lines.Line2D at 0x7f8f7dfd9ca0>,
      <matplotlib.lines.Line2D at 0x7f8f7dfe71c0>,
      <matplotlib.lines.Line2D at 0x7f8f7dfe76a0>,
      <matplotlib.lines.Line2D at 0x7f8f7dfe7b80>,
      <matplotlib.lines.Line2D at 0x7f8f7dfe7fd0>,
      <matplotlib.lines.Line2D at 0x7f8f7dff5580>,
      <matplotlib.lines.Line2D at 0x7f8f7dff5a60>,
      <matplotlib.lines.Line2D at 0x7f8f7dff5f40>,
      <matplotlib.lines.Line2D at 0x7f8f7e000460>,
      <matplotlib.lines.Line2D at 0x7f8f7e000940>,
      <matplotlib.lines.Line2D at 0x7f8f7e000e20>,
      <matplotlib.lines.Line2D at 0x7f8f7e00f340>,
      <matplotlib.lines.Line2D at 0x7f8f7e00f820>,
      <matplotlib.lines.Line2D at 0x7f8f7e00fd00>,
      <matplotlib.lines.Line2D at 0x7f8f7e01d220>,
      <matplotlib.lines.Line2D at 0x7f8f7e01d700>,
      <matplotlib.lines.Line2D at 0x7f8f7e01dbe0>,
      <matplotlib.lines.Line2D at 0x7f8f7e02c040>,
      <matplotlib.lines.Line2D at 0x7f8f7e02c5e0>,
      <matplotlib.lines.Line2D at 0x7f8f7e02cac0>,
      <matplotlib.lines.Line2D at 0x7f8f7e02cfa0>,
      <matplotlib.lines.Line2D at 0x7f8f7e0384c0>,
      <matplotlib.lines.Line2D at 0x7f8f7e0389a0>,
      <matplotlib.lines.Line2D at 0x7f8f7e038e80>,
      <matplotlib.lines.Line2D at 0x7f8f7e0483a0>,
      <matplotlib.lines.Line2D at 0x7f8f7e048880>,
      <matplotlib.lines.Line2D at 0x7f8f7e048d60>,
      <matplotlib.lines.Line2D at 0x7f8f7e053280>,
      <matplotlib.lines.Line2D at 0x7f8f7e053760>,
      <matplotlib.lines.Line2D at 0x7f8f7e053c40>,
      <matplotlib.lines.Line2D at 0x7f8f7e060160>,
      <matplotlib.lines.Line2D at 0x7f8f7e060640>,
      <matplotlib.lines.Line2D at 0x7f8f7e060b20>,
      <matplotlib.lines.Line2D at 0x7f8f7e060fa0>,
      <matplotlib.lines.Line2D at 0x7f8f7e06e520>,
      <matplotlib.lines.Line2D at 0x7f8f7e06ea00>,
      <matplotlib.lines.Line2D at 0x7f8f7e06eee0>,
      <matplotlib.lines.Line2D at 0x7f8f7e07d400>,
      <matplotlib.lines.Line2D at 0x7f8f7e07d8e0>,
      <matplotlib.lines.Line2D at 0x7f8f7e07ddc0>,
      <matplotlib.lines.Line2D at 0x7f8f7e0882e0>,
      <matplotlib.lines.Line2D at 0x7f8f7e0887c0>,
      <matplotlib.lines.Line2D at 0x7f8f7e088ca0>,
      <matplotlib.lines.Line2D at 0x7f8f7e0971c0>,
      <matplotlib.lines.Line2D at 0x7f8f7e0976a0>,
      <matplotlib.lines.Line2D at 0x7f8f7e097b80>,
      <matplotlib.lines.Line2D at 0x7f8f7e097fd0>,
      <matplotlib.lines.Line2D at 0x7f8f7e0a6580>,
      <matplotlib.lines.Line2D at 0x7f8f7e0a6a60>,
      <matplotlib.lines.Line2D at 0x7f8f7e0a6f40>,
      <matplotlib.lines.Line2D at 0x7f8f7e0b2460>,
      <matplotlib.lines.Line2D at 0x7f8f7e0b2940>,
      <matplotlib.lines.Line2D at 0x7f8f7e0b2e20>,
      <matplotlib.lines.Line2D at 0x7f8f7e0c1340>,
      <matplotlib.lines.Line2D at 0x7f8f7e0c1820>,
      <matplotlib.lines.Line2D at 0x7f8f7e0c1d00>,
      <matplotlib.lines.Line2D at 0x7f8f7e0cf220>,
      <matplotlib.lines.Line2D at 0x7f8f7e0cf700>,
      <matplotlib.lines.Line2D at 0x7f8f7e0cfbe0>,
      <matplotlib.lines.Line2D at 0x7f8f7e0dd040>,
      <matplotlib.lines.Line2D at 0x7f8f7e0dd5e0>,
      <matplotlib.lines.Line2D at 0x7f8f7e0ddac0>,
      <matplotlib.lines.Line2D at 0x7f8f7e0ddfa0>,
      <matplotlib.lines.Line2D at 0x7f8f7e0eb4c0>,
      <matplotlib.lines.Line2D at 0x7f8f7e0eb9a0>,
      <matplotlib.lines.Line2D at 0x7f8f7e0ebe80>,
      <matplotlib.lines.Line2D at 0x7f8f7e0f63a0>,
      <matplotlib.lines.Line2D at 0x7f8f7e0f6880>,
      <matplotlib.lines.Line2D at 0x7f8f7e0f6d60>,
      <matplotlib.lines.Line2D at 0x7f8f7e105280>,
      <matplotlib.lines.Line2D at 0x7f8f7e105760>,
      <matplotlib.lines.Line2D at 0x7f8f7e105c40>,
      <matplotlib.lines.Line2D at 0x7f8f7e113160>,
      <matplotlib.lines.Line2D at 0x7f8f7e113640>,
      <matplotlib.lines.Line2D at 0x7f8f7e113b20>,
      <matplotlib.lines.Line2D at 0x7f8f7e113fa0>,
      <matplotlib.lines.Line2D at 0x7f8f7e120520>,
      <matplotlib.lines.Line2D at 0x7f8f7e120a00>,
      <matplotlib.lines.Line2D at 0x7f8f7e120ee0>,
      <matplotlib.lines.Line2D at 0x7f8f7e12f400>,
      <matplotlib.lines.Line2D at 0x7f8f7e12f8e0>,
      <matplotlib.lines.Line2D at 0x7f8f7e12fdc0>,
      <matplotlib.lines.Line2D at 0x7f8f7e13a2e0>,
      <matplotlib.lines.Line2D at 0x7f8f7e13a7c0>,
      <matplotlib.lines.Line2D at 0x7f8f7e13aca0>,
      <matplotlib.lines.Line2D at 0x7f8f7e1471c0>,
      <matplotlib.lines.Line2D at 0x7f8f7e1476a0>,
      <matplotlib.lines.Line2D at 0x7f8f7e147b80>,
      <matplotlib.lines.Line2D at 0x7f8f7e147fd0>,
      <matplotlib.lines.Line2D at 0x7f8f7e157580>,
      <matplotlib.lines.Line2D at 0x7f8f7e157a60>,
      <matplotlib.lines.Line2D at 0x7f8f7e157f70>,
      <matplotlib.lines.Line2D at 0x7f8f7e164490>,
      <matplotlib.lines.Line2D at 0x7f8f7e164970>,
      <matplotlib.lines.Line2D at 0x7f8f7e164e50>,
      <matplotlib.lines.Line2D at 0x7f8f7e171370>,
      <matplotlib.lines.Line2D at 0x7f8f7e171850>,
      <matplotlib.lines.Line2D at 0x7f8f7e171d30>,
      <matplotlib.lines.Line2D at 0x7f8f7e180250>,
      <matplotlib.lines.Line2D at 0x7f8f7e180730>,
      <matplotlib.lines.Line2D at 0x7f8f7e180c10>,
      <matplotlib.lines.Line2D at 0x7f8f7e18c130>,
      <matplotlib.lines.Line2D at 0x7f8f7e18c610>,
      <matplotlib.lines.Line2D at 0x7f8f7e18caf0>,
      <matplotlib.lines.Line2D at 0x7f8f7e18cfd0>,
      <matplotlib.lines.Line2D at 0x7f8fa852a4f0>,
      <matplotlib.lines.Line2D at 0x7f8fa852a9d0>,
      <matplotlib.lines.Line2D at 0x7f8fa852aeb0>,
      <matplotlib.lines.Line2D at 0x7f8fa85373d0>,
      <matplotlib.lines.Line2D at 0x7f8fa85378b0>,
      <matplotlib.lines.Line2D at 0x7f8fa8537d90>,
      <matplotlib.lines.Line2D at 0x7f8fa85452b0>,
      <matplotlib.lines.Line2D at 0x7f8fa8545790>,
      <matplotlib.lines.Line2D at 0x7f8fa8545c70>,
      <matplotlib.lines.Line2D at 0x7f8fa8556190>,
      <matplotlib.lines.Line2D at 0x7f8fa8556670>,
      <matplotlib.lines.Line2D at 0x7f8fa8556b50>,
      <matplotlib.lines.Line2D at 0x7f8fa8556fd0>,
      <matplotlib.lines.Line2D at 0x7f8fa8563550>,
      <matplotlib.lines.Line2D at 0x7f8fa8563a30>,
      <matplotlib.lines.Line2D at 0x7f8fa8563f10>,
      <matplotlib.lines.Line2D at 0x7f8fa856e430>,
      <matplotlib.lines.Line2D at 0x7f8fa856e910>,
      <matplotlib.lines.Line2D at 0x7f8fa856edf0>,
      <matplotlib.lines.Line2D at 0x7f8fa857c310>,
      <matplotlib.lines.Line2D at 0x7f8fa857c7f0>,
      <matplotlib.lines.Line2D at 0x7f8fa857ccd0>,
      <matplotlib.lines.Line2D at 0x7f8fa858b1f0>,
      <matplotlib.lines.Line2D at 0x7f8fa858b6d0>,
      <matplotlib.lines.Line2D at 0x7f8fa858bbb0>,
      <matplotlib.lines.Line2D at 0x7f8fa858bfd0>,
      <matplotlib.lines.Line2D at 0x7f8fa85995b0>,
      <matplotlib.lines.Line2D at 0x7f8fa8599a90>,
      <matplotlib.lines.Line2D at 0x7f8fa8599f70>,
      <matplotlib.lines.Line2D at 0x7f8fa85a5490>,
      <matplotlib.lines.Line2D at 0x7f8fa85a5970>,
      <matplotlib.lines.Line2D at 0x7f8fa85a5e50>,
      <matplotlib.lines.Line2D at 0x7f8fa85b2370>,
      <matplotlib.lines.Line2D at 0x7f8fa85b2850>,
      <matplotlib.lines.Line2D at 0x7f8fa85b2d30>,
      <matplotlib.lines.Line2D at 0x7f8fa85c1250>,
      <matplotlib.lines.Line2D at 0x7f8fa85c1730>,
      <matplotlib.lines.Line2D at 0x7f8fa85c1c10>,
      <matplotlib.lines.Line2D at 0x7f8fa85ce130>,
      <matplotlib.lines.Line2D at 0x7f8fa85ce610>,
      <matplotlib.lines.Line2D at 0x7f8fa85ceaf0>,
      <matplotlib.lines.Line2D at 0x7f8fa85cefd0>,
      <matplotlib.lines.Line2D at 0x7f8fa85dc4f0>,
      <matplotlib.lines.Line2D at 0x7f8fa85dc9d0>,
      <matplotlib.lines.Line2D at 0x7f8fa85dceb0>,
      <matplotlib.lines.Line2D at 0x7f8fa85ea3d0>,
      <matplotlib.lines.Line2D at 0x7f8fa85ea8b0>,
      <matplotlib.lines.Line2D at 0x7f8fa85ead90>,
      <matplotlib.lines.Line2D at 0x7f8fa85f82b0>,
      <matplotlib.lines.Line2D at 0x7f8fa85f8790>,
      <matplotlib.lines.Line2D at 0x7f8fa85f8ca0>,
      <matplotlib.lines.Line2D at 0x7f8fa86041c0>,
      <matplotlib.lines.Line2D at 0x7f8fa86046a0>,
      <matplotlib.lines.Line2D at 0x7f8fa8604b80>,
      <matplotlib.lines.Line2D at 0x7f8fa8604fd0>,
      <matplotlib.lines.Line2D at 0x7f8fa8612580>,
      <matplotlib.lines.Line2D at 0x7f8fa8612a60>,
      <matplotlib.lines.Line2D at 0x7f8fa8612f40>,
      <matplotlib.lines.Line2D at 0x7f8fa8621460>,
      <matplotlib.lines.Line2D at 0x7f8fa8621940>,
      <matplotlib.lines.Line2D at 0x7f8fa8621e20>,
      <matplotlib.lines.Line2D at 0x7f8fa862d340>,
      <matplotlib.lines.Line2D at 0x7f8fa862d820>,
      <matplotlib.lines.Line2D at 0x7f8fa862dd00>,
      <matplotlib.lines.Line2D at 0x7f8fa863c220>,
      <matplotlib.lines.Line2D at 0x7f8fa863c700>,
      <matplotlib.lines.Line2D at 0x7f8fa863cbe0>,
      <matplotlib.lines.Line2D at 0x7f8fa864b100>,
      <matplotlib.lines.Line2D at 0x7f8fa864b5e0>,
      <matplotlib.lines.Line2D at 0x7f8fa864bac0>,
      <matplotlib.lines.Line2D at 0x7f8fa864bfa0>,
      <matplotlib.lines.Line2D at 0x7f8fa86544c0>,
      <matplotlib.lines.Line2D at 0x7f8fa86549a0>,
      <matplotlib.lines.Line2D at 0x7f8fa8654e80>,
      <matplotlib.lines.Line2D at 0x7f8fa86653a0>,
      <matplotlib.lines.Line2D at 0x7f8fa8665880>,
      <matplotlib.lines.Line2D at 0x7f8fa8665d60>,
      <matplotlib.lines.Line2D at 0x7f8fa8673280>,
      <matplotlib.lines.Line2D at 0x7f8fa8673760>,
      <matplotlib.lines.Line2D at 0x7f8fa8673c40>,
      <matplotlib.lines.Line2D at 0x7f8fa867e160>],
     [<matplotlib.patches.Rectangle at 0x7f8f5837ce80>,
      <matplotlib.patches.Rectangle at 0x7f8f583883a0>,
      <matplotlib.patches.Rectangle at 0x7f8f58388880>,
      <matplotlib.patches.Rectangle at 0x7f8f58388d60>,
      <matplotlib.patches.Rectangle at 0x7f8f7d8b7160>,
      <matplotlib.patches.Rectangle at 0x7f8f7d8b7760>,
      <matplotlib.patches.Rectangle at 0x7f8f7d8b7c40>,
      <matplotlib.patches.Rectangle at 0x7f8f7d8c4040>,
      <matplotlib.patches.Rectangle at 0x7f8f7d8c4640>,
      <matplotlib.patches.Rectangle at 0x7f8f7d8c4b20>,
      <matplotlib.patches.Rectangle at 0x7f8f7d8c4f40>,
      <matplotlib.patches.Rectangle at 0x7f8f7d8d1520>,
      <matplotlib.patches.Rectangle at 0x7f8f7d8d1a00>,
      <matplotlib.patches.Rectangle at 0x7f8f7d8d1ee0>,
      <matplotlib.patches.Rectangle at 0x7f8f7d8de400>,
      <matplotlib.patches.Rectangle at 0x7f8f7d8de8e0>,
      <matplotlib.patches.Rectangle at 0x7f8f7d8dedc0>,
      <matplotlib.patches.Rectangle at 0x7f8f7d8ec2e0>,
      <matplotlib.patches.Rectangle at 0x7f8f7d8ec7c0>,
      <matplotlib.patches.Rectangle at 0x7f8f7d8ecca0>,
      <matplotlib.patches.Rectangle at 0x7f8f7d8fa0a0>,
      <matplotlib.patches.Rectangle at 0x7f8f7d8fa6a0>,
      <matplotlib.patches.Rectangle at 0x7f8f7d8fab80>,
      <matplotlib.patches.Rectangle at 0x7f8f7d8faf40>,
      <matplotlib.patches.Rectangle at 0x7f8f7d907580>,
      <matplotlib.patches.Rectangle at 0x7f8f7d907a60>,
      <matplotlib.patches.Rectangle at 0x7f8f7d907f40>,
      <matplotlib.patches.Rectangle at 0x7f8f7d914460>,
      <matplotlib.patches.Rectangle at 0x7f8f7d914940>,
      <matplotlib.patches.Rectangle at 0x7f8f7d914e20>,
      <matplotlib.patches.Rectangle at 0x7f8f7d922340>,
      <matplotlib.patches.Rectangle at 0x7f8f7d922820>,
      <matplotlib.patches.Rectangle at 0x7f8f7d922d00>,
      <matplotlib.patches.Rectangle at 0x7f8f7d92f220>,
      <matplotlib.patches.Rectangle at 0x7f8f7d92f700>,
      <matplotlib.patches.Rectangle at 0x7f8f7d92fbe0>,
      <matplotlib.patches.Rectangle at 0x7f8f7d92ffa0>,
      <matplotlib.patches.Rectangle at 0x7f8f7d96d5e0>,
      <matplotlib.patches.Rectangle at 0x7f8f7d96dac0>,
      <matplotlib.patches.Rectangle at 0x7f8f7d96dfa0>,
      <matplotlib.patches.Rectangle at 0x7f8f7d97b4c0>,
      <matplotlib.patches.Rectangle at 0x7f8f7d97b9a0>,
      <matplotlib.patches.Rectangle at 0x7f8f7d97be80>,
      <matplotlib.patches.Rectangle at 0x7f8f7d9883a0>,
      <matplotlib.patches.Rectangle at 0x7f8f7d988880>,
      <matplotlib.patches.Rectangle at 0x7f8f7d988d60>,
      <matplotlib.patches.Rectangle at 0x7f8f7d997160>,
      <matplotlib.patches.Rectangle at 0x7f8f7d997760>,
      <matplotlib.patches.Rectangle at 0x7f8f7d997c40>,
      <matplotlib.patches.Rectangle at 0x7f8f7d9a5160>,
      <matplotlib.patches.Rectangle at 0x7f8f7d9a5640>,
      <matplotlib.patches.Rectangle at 0x7f8f7d9a5b20>,
      <matplotlib.patches.Rectangle at 0x7f8f7d9a5f40>,
      <matplotlib.patches.Rectangle at 0x7f8f7d9b0520>,
      <matplotlib.patches.Rectangle at 0x7f8f7d9b0a00>,
      <matplotlib.patches.Rectangle at 0x7f8f7d9b0ee0>,
      <matplotlib.patches.Rectangle at 0x7f8f7d9be400>,
      <matplotlib.patches.Rectangle at 0x7f8f7d9be8e0>,
      <matplotlib.patches.Rectangle at 0x7f8f7d9bedc0>,
      <matplotlib.patches.Rectangle at 0x7f8f7d9ce2e0>,
      <matplotlib.patches.Rectangle at 0x7f8f7d9ce7c0>,
      <matplotlib.patches.Rectangle at 0x7f8f7d9ceca0>,
      <matplotlib.patches.Rectangle at 0x7f8f7d9db0a0>,
      <matplotlib.patches.Rectangle at 0x7f8f7d9db6a0>,
      <matplotlib.patches.Rectangle at 0x7f8f7d9dbb80>,
      <matplotlib.patches.Rectangle at 0x7f8f7d9dbf40>,
      <matplotlib.patches.Rectangle at 0x7f8f7d9e8580>,
      <matplotlib.patches.Rectangle at 0x7f8f7d9e8a60>,
      <matplotlib.patches.Rectangle at 0x7f8f7d9e8f40>,
      <matplotlib.patches.Rectangle at 0x7f8f7d9f5460>,
      <matplotlib.patches.Rectangle at 0x7f8f7d9f5940>,
      <matplotlib.patches.Rectangle at 0x7f8f7d9f5e20>,
      <matplotlib.patches.Rectangle at 0x7f8f7da02340>,
      <matplotlib.patches.Rectangle at 0x7f8f7da02820>,
      <matplotlib.patches.Rectangle at 0x7f8f7da02d00>,
      <matplotlib.patches.Rectangle at 0x7f8f7da13220>,
      <matplotlib.patches.Rectangle at 0x7f8f7da13700>,
      <matplotlib.patches.Rectangle at 0x7f8f7da13be0>,
      <matplotlib.patches.Rectangle at 0x7f8f7da13fa0>,
      <matplotlib.patches.Rectangle at 0x7f8f7da1e5e0>,
      <matplotlib.patches.Rectangle at 0x7f8f7da1eac0>,
      <matplotlib.patches.Rectangle at 0x7f8f7da1efa0>,
      <matplotlib.patches.Rectangle at 0x7f8f7da2c4c0>,
      <matplotlib.patches.Rectangle at 0x7f8f7da2c9a0>,
      <matplotlib.patches.Rectangle at 0x7f8f7da2ce80>,
      <matplotlib.patches.Rectangle at 0x7f8f7da3b3a0>,
      <matplotlib.patches.Rectangle at 0x7f8f7da3b880>,
      <matplotlib.patches.Rectangle at 0x7f8f7da3bd60>,
      <matplotlib.patches.Rectangle at 0x7f8f7da4a160>,
      <matplotlib.patches.Rectangle at 0x7f8f7da4a760>,
      <matplotlib.patches.Rectangle at 0x7f8f7da4ac40>,
      <matplotlib.patches.Rectangle at 0x7f8f7da55160>,
      <matplotlib.patches.Rectangle at 0x7f8f7da55640>,
      <matplotlib.patches.Rectangle at 0x7f8f7da55b20>,
      <matplotlib.patches.Rectangle at 0x7f8f7da55f40>,
      <matplotlib.patches.Rectangle at 0x7f8f7da62520>,
      <matplotlib.patches.Rectangle at 0x7f8f7da62a00>,
      <matplotlib.patches.Rectangle at 0x7f8f7da62ee0>,
      <matplotlib.patches.Rectangle at 0x7f8f7da70400>,
      <matplotlib.patches.Rectangle at 0x7f8f7da708e0>,
      <matplotlib.patches.Rectangle at 0x7f8f7da70dc0>,
      <matplotlib.patches.Rectangle at 0x7f8f7da7e2e0>,
      <matplotlib.patches.Rectangle at 0x7f8f7da7e7c0>,
      <matplotlib.patches.Rectangle at 0x7f8f7da7eca0>,
      <matplotlib.patches.Rectangle at 0x7f8f7da8d0a0>,
      <matplotlib.patches.Rectangle at 0x7f8f7da8d6a0>,
      <matplotlib.patches.Rectangle at 0x7f8f7da8db80>,
      <matplotlib.patches.Rectangle at 0x7f8f7da8df40>,
      <matplotlib.patches.Rectangle at 0x7f8f7da985b0>,
      <matplotlib.patches.Rectangle at 0x7f8f7da98a90>,
      <matplotlib.patches.Rectangle at 0x7f8f7da98f70>,
      <matplotlib.patches.Rectangle at 0x7f8f7daa7490>,
      <matplotlib.patches.Rectangle at 0x7f8f7daa7970>,
      <matplotlib.patches.Rectangle at 0x7f8f7daa7e50>,
      <matplotlib.patches.Rectangle at 0x7f8f7dab5370>,
      <matplotlib.patches.Rectangle at 0x7f8f7dab5850>,
      <matplotlib.patches.Rectangle at 0x7f8f7dab5d30>,
      <matplotlib.patches.Rectangle at 0x7f8f7dac3250>,
      <matplotlib.patches.Rectangle at 0x7f8f7dac3730>,
      <matplotlib.patches.Rectangle at 0x7f8f7dac3c10>,
      <matplotlib.patches.Rectangle at 0x7f8f7dac3fd0>,
      <matplotlib.patches.Rectangle at 0x7f8f7dad0610>,
      <matplotlib.patches.Rectangle at 0x7f8f7dad0af0>,
      <matplotlib.patches.Rectangle at 0x7f8f7dad0fd0>,
      <matplotlib.patches.Rectangle at 0x7f8f7dadd4f0>,
      <matplotlib.patches.Rectangle at 0x7f8f7dadd9d0>,
      <matplotlib.patches.Rectangle at 0x7f8f7daddeb0>,
      <matplotlib.patches.Rectangle at 0x7f8f7daeb3d0>,
      <matplotlib.patches.Rectangle at 0x7f8f7daeb8b0>,
      <matplotlib.patches.Rectangle at 0x7f8f7daebd90>,
      <matplotlib.patches.Rectangle at 0x7f8f7daf92b0>,
      <matplotlib.patches.Rectangle at 0x7f8f7daf9790>,
      <matplotlib.patches.Rectangle at 0x7f8f7daf9c70>,
      <matplotlib.patches.Rectangle at 0x7f8f7db06190>,
      <matplotlib.patches.Rectangle at 0x7f8f7db06670>,
      <matplotlib.patches.Rectangle at 0x7f8f7db06b50>,
      <matplotlib.patches.Rectangle at 0x7f8f7db06fa0>,
      <matplotlib.patches.Rectangle at 0x7f8f7db12550>,
      <matplotlib.patches.Rectangle at 0x7f8f7db12a30>,
      <matplotlib.patches.Rectangle at 0x7f8f7db12f10>,
      <matplotlib.patches.Rectangle at 0x7f8f7db22430>,
      <matplotlib.patches.Rectangle at 0x7f8f7db22910>,
      <matplotlib.patches.Rectangle at 0x7f8f7db22df0>,
      <matplotlib.patches.Rectangle at 0x7f8f7db2f310>,
      <matplotlib.patches.Rectangle at 0x7f8f7db2f7f0>,
      <matplotlib.patches.Rectangle at 0x7f8f7db2fcd0>,
      <matplotlib.patches.Rectangle at 0x7f8f7db3e0d0>,
      <matplotlib.patches.Rectangle at 0x7f8f7db3e6d0>,
      <matplotlib.patches.Rectangle at 0x7f8f7db3ebb0>,
      <matplotlib.patches.Rectangle at 0x7f8f7db3efd0>,
      <matplotlib.patches.Rectangle at 0x7f8f7db4b5b0>,
      <matplotlib.patches.Rectangle at 0x7f8f7db4ba90>,
      <matplotlib.patches.Rectangle at 0x7f8f7db4bf70>,
      <matplotlib.patches.Rectangle at 0x7f8f7db56490>,
      <matplotlib.patches.Rectangle at 0x7f8f7db56970>,
      <matplotlib.patches.Rectangle at 0x7f8f7db56e50>,
      <matplotlib.patches.Rectangle at 0x7f8f7dcde370>,
      <matplotlib.patches.Rectangle at 0x7f8f7dcde850>,
      <matplotlib.patches.Rectangle at 0x7f8f7dcded30>,
      <matplotlib.patches.Rectangle at 0x7f8f7dcec250>,
      <matplotlib.patches.Rectangle at 0x7f8f7dcec730>,
      <matplotlib.patches.Rectangle at 0x7f8f7dcecc10>,
      <matplotlib.patches.Rectangle at 0x7f8f7dcecfd0>,
      <matplotlib.patches.Rectangle at 0x7f8f7dcf8610>,
      <matplotlib.patches.Rectangle at 0x7f8f7dcf8af0>,
      <matplotlib.patches.Rectangle at 0x7f8f7dcf8fd0>,
      <matplotlib.patches.Rectangle at 0x7f8f7dd064f0>,
      <matplotlib.patches.Rectangle at 0x7f8f7dd069d0>,
      <matplotlib.patches.Rectangle at 0x7f8f7dd06eb0>,
      <matplotlib.patches.Rectangle at 0x7f8f7dd133d0>,
      <matplotlib.patches.Rectangle at 0x7f8f7dd138b0>,
      <matplotlib.patches.Rectangle at 0x7f8f7dd13d90>,
      <matplotlib.patches.Rectangle at 0x7f8f7dd212b0>,
      <matplotlib.patches.Rectangle at 0x7f8f7dd21790>,
      <matplotlib.patches.Rectangle at 0x7f8f7dd21c70>,
      <matplotlib.patches.Rectangle at 0x7f8f7dd30190>,
      <matplotlib.patches.Rectangle at 0x7f8f7dd30670>,
      <matplotlib.patches.Rectangle at 0x7f8f7dd30b50>,
      <matplotlib.patches.Rectangle at 0x7f8f7dd30fa0>,
      <matplotlib.patches.Rectangle at 0x7f8f7dd3d550>,
      <matplotlib.patches.Rectangle at 0x7f8f7dd3da30>,
      <matplotlib.patches.Rectangle at 0x7f8f7dd3df10>,
      <matplotlib.patches.Rectangle at 0x7f8f7dd4a430>,
      <matplotlib.patches.Rectangle at 0x7f8f7dd4a910>,
      <matplotlib.patches.Rectangle at 0x7f8f7dd4adf0>,
      <matplotlib.patches.Rectangle at 0x7f8f7dd5a310>,
      <matplotlib.patches.Rectangle at 0x7f8f7dd5a7f0>,
      <matplotlib.patches.Rectangle at 0x7f8f7dd5acd0>,
      <matplotlib.patches.Rectangle at 0x7f8f7dd660d0>,
      <matplotlib.patches.Rectangle at 0x7f8f7dd666d0>,
      <matplotlib.patches.Rectangle at 0x7f8f7dd66bb0>,
      <matplotlib.patches.Rectangle at 0x7f8f7dd66fd0>,
      <matplotlib.patches.Rectangle at 0x7f8f7dd745b0>,
      <matplotlib.patches.Rectangle at 0x7f8f7dd74a90>,
      <matplotlib.patches.Rectangle at 0x7f8f7dd74f70>,
      <matplotlib.patches.Rectangle at 0x7f8f7dd81490>,
      <matplotlib.patches.Rectangle at 0x7f8f7dd81970>,
      <matplotlib.patches.Rectangle at 0x7f8f7dd81e50>,
      <matplotlib.patches.Rectangle at 0x7f8f7dd8d370>,
      <matplotlib.patches.Rectangle at 0x7f8f7dd8d850>,
      <matplotlib.patches.Rectangle at 0x7f8f7dd8dd30>,
      <matplotlib.patches.Rectangle at 0x7f8f7dd9d250>,
      <matplotlib.patches.Rectangle at 0x7f8f7dd9d730>,
      <matplotlib.patches.Rectangle at 0x7f8f7dd9dc10>,
      <matplotlib.patches.Rectangle at 0x7f8f7dd9dfd0>,
      <matplotlib.patches.Rectangle at 0x7f8f7dda8610>,
      <matplotlib.patches.Rectangle at 0x7f8f7dda8af0>,
      <matplotlib.patches.Rectangle at 0x7f8f7dda8fd0>,
      <matplotlib.patches.Rectangle at 0x7f8f7ddb84f0>,
      <matplotlib.patches.Rectangle at 0x7f8f7ddb89d0>,
      <matplotlib.patches.Rectangle at 0x7f8f7ddb8eb0>,
      <matplotlib.patches.Rectangle at 0x7f8f7ddc53d0>,
      <matplotlib.patches.Rectangle at 0x7f8f7ddc58b0>,
      <matplotlib.patches.Rectangle at 0x7f8f7ddc5dc0>,
      <matplotlib.patches.Rectangle at 0x7f8f7ddd32e0>,
      <matplotlib.patches.Rectangle at 0x7f8f7ddd37c0>,
      <matplotlib.patches.Rectangle at 0x7f8f7ddd3ca0>,
      <matplotlib.patches.Rectangle at 0x7f8f7dde00a0>,
      <matplotlib.patches.Rectangle at 0x7f8f7dde06a0>,
      <matplotlib.patches.Rectangle at 0x7f8f7dde0b80>,
      <matplotlib.patches.Rectangle at 0x7f8f7dde0f40>,
      <matplotlib.patches.Rectangle at 0x7f8f7ddee580>,
      <matplotlib.patches.Rectangle at 0x7f8f7ddeea60>,
      <matplotlib.patches.Rectangle at 0x7f8f7ddeef40>,
      <matplotlib.patches.Rectangle at 0x7f8f7ddfb460>,
      <matplotlib.patches.Rectangle at 0x7f8f7ddfb940>,
      <matplotlib.patches.Rectangle at 0x7f8f7ddfbe20>,
      <matplotlib.patches.Rectangle at 0x7f8f7de09340>,
      <matplotlib.patches.Rectangle at 0x7f8f7de09820>,
      <matplotlib.patches.Rectangle at 0x7f8f7de09d00>,
      <matplotlib.patches.Rectangle at 0x7f8f7de17100>,
      <matplotlib.patches.Rectangle at 0x7f8f7de17700>,
      <matplotlib.patches.Rectangle at 0x7f8f7de17be0>,
      <matplotlib.patches.Rectangle at 0x7f8f7de17fa0>,
      <matplotlib.patches.Rectangle at 0x7f8f7de265e0>,
      <matplotlib.patches.Rectangle at 0x7f8f7de26ac0>,
      <matplotlib.patches.Rectangle at 0x7f8f7de26fa0>,
      <matplotlib.patches.Rectangle at 0x7f8f7de334c0>,
      <matplotlib.patches.Rectangle at 0x7f8f7de339a0>,
      <matplotlib.patches.Rectangle at 0x7f8f7de33e80>,
      <matplotlib.patches.Rectangle at 0x7f8f7de403a0>,
      <matplotlib.patches.Rectangle at 0x7f8f7de40880>,
      <matplotlib.patches.Rectangle at 0x7f8f7de40d60>,
      <matplotlib.patches.Rectangle at 0x7f8f7de4d280>,
      <matplotlib.patches.Rectangle at 0x7f8f7de4d760>,
      <matplotlib.patches.Rectangle at 0x7f8f7de4dc40>,
      <matplotlib.patches.Rectangle at 0x7f8f7de5b040>,
      <matplotlib.patches.Rectangle at 0x7f8f7de5b640>,
      <matplotlib.patches.Rectangle at 0x7f8f7de5bb20>,
      <matplotlib.patches.Rectangle at 0x7f8f7de5bf40>,
      <matplotlib.patches.Rectangle at 0x7f8f7de69520>,
      <matplotlib.patches.Rectangle at 0x7f8f7de69a00>,
      <matplotlib.patches.Rectangle at 0x7f8f7de69ee0>,
      <matplotlib.patches.Rectangle at 0x7f8f7de76400>,
      <matplotlib.patches.Rectangle at 0x7f8f7de768e0>,
      <matplotlib.patches.Rectangle at 0x7f8f7de76dc0>,
      <matplotlib.patches.Rectangle at 0x7f8f7de832e0>,
      <matplotlib.patches.Rectangle at 0x7f8f7de837c0>,
      <matplotlib.patches.Rectangle at 0x7f8f7de83ca0>,
      <matplotlib.patches.Rectangle at 0x7f8f7de921c0>,
      <matplotlib.patches.Rectangle at 0x7f8f7de926a0>,
      <matplotlib.patches.Rectangle at 0x7f8f7de92b80>,
      <matplotlib.patches.Rectangle at 0x7f8f7de92f40>,
      <matplotlib.patches.Rectangle at 0x7f8f7de9e580>,
      <matplotlib.patches.Rectangle at 0x7f8f7de9ea60>,
      <matplotlib.patches.Rectangle at 0x7f8f7de9ef40>,
      <matplotlib.patches.Rectangle at 0x7f8f7dead460>,
      <matplotlib.patches.Rectangle at 0x7f8f7dead940>,
      <matplotlib.patches.Rectangle at 0x7f8f7deade20>,
      <matplotlib.patches.Rectangle at 0x7f8f7debb340>,
      <matplotlib.patches.Rectangle at 0x7f8f7debb820>,
      <matplotlib.patches.Rectangle at 0x7f8f7debbd00>,
      <matplotlib.patches.Rectangle at 0x7f8f7dec9100>,
      <matplotlib.patches.Rectangle at 0x7f8f7dec9700>,
      <matplotlib.patches.Rectangle at 0x7f8f7dec9be0>,
      <matplotlib.patches.Rectangle at 0x7f8f7dec9fa0>,
      <matplotlib.patches.Rectangle at 0x7f8f7ded7610>,
      <matplotlib.patches.Rectangle at 0x7f8f7ded7af0>,
      <matplotlib.patches.Rectangle at 0x7f8f7ded7fd0>,
      <matplotlib.patches.Rectangle at 0x7f8f7dee24f0>,
      <matplotlib.patches.Rectangle at 0x7f8f7dee2a00>,
      <matplotlib.patches.Rectangle at 0x7f8f7dee2ee0>,
      <matplotlib.patches.Rectangle at 0x7f8f7def1400>,
      <matplotlib.patches.Rectangle at 0x7f8f7def18e0>,
      <matplotlib.patches.Rectangle at 0x7f8f7def1dc0>,
      <matplotlib.patches.Rectangle at 0x7f8f7df002e0>,
      <matplotlib.patches.Rectangle at 0x7f8f7df007c0>,
      <matplotlib.patches.Rectangle at 0x7f8f7df00ca0>,
      <matplotlib.patches.Rectangle at 0x7f8f7df0d0a0>,
      <matplotlib.patches.Rectangle at 0x7f8f7df0d6a0>,
      <matplotlib.patches.Rectangle at 0x7f8f7df0db80>,
      <matplotlib.patches.Rectangle at 0x7f8f7df0df40>,
      <matplotlib.patches.Rectangle at 0x7f8f7df1a580>,
      <matplotlib.patches.Rectangle at 0x7f8f7df1aa60>,
      <matplotlib.patches.Rectangle at 0x7f8f7df1af40>,
      <matplotlib.patches.Rectangle at 0x7f8f7df26460>,
      <matplotlib.patches.Rectangle at 0x7f8f7df26940>,
      <matplotlib.patches.Rectangle at 0x7f8f7df26e20>,
      <matplotlib.patches.Rectangle at 0x7f8f7df35340>,
      <matplotlib.patches.Rectangle at 0x7f8f7df35820>,
      <matplotlib.patches.Rectangle at 0x7f8f7df35d00>,
      <matplotlib.patches.Rectangle at 0x7f8f7df43220>,
      <matplotlib.patches.Rectangle at 0x7f8f7df43700>,
      <matplotlib.patches.Rectangle at 0x7f8f7df43be0>,
      <matplotlib.patches.Rectangle at 0x7f8f7df43fa0>,
      <matplotlib.patches.Rectangle at 0x7f8f7df505e0>,
      <matplotlib.patches.Rectangle at 0x7f8f7df50ac0>,
      <matplotlib.patches.Rectangle at 0x7f8f7df50fa0>,
      <matplotlib.patches.Rectangle at 0x7f8f7df5d4c0>,
      <matplotlib.patches.Rectangle at 0x7f8f7df5d9a0>,
      <matplotlib.patches.Rectangle at 0x7f8f7df5de80>,
      <matplotlib.patches.Rectangle at 0x7f8f7df6c3a0>,
      <matplotlib.patches.Rectangle at 0x7f8f7df6c880>,
      <matplotlib.patches.Rectangle at 0x7f8f7df6cd60>,
      <matplotlib.patches.Rectangle at 0x7f8f7df79160>,
      <matplotlib.patches.Rectangle at 0x7f8f7df79760>,
      <matplotlib.patches.Rectangle at 0x7f8f7df79c40>,
      <matplotlib.patches.Rectangle at 0x7f8f7df86160>,
      <matplotlib.patches.Rectangle at 0x7f8f7df86640>,
      <matplotlib.patches.Rectangle at 0x7f8f7df86b20>,
      <matplotlib.patches.Rectangle at 0x7f8f7df86f40>,
      <matplotlib.patches.Rectangle at 0x7f8f7df93520>,
      <matplotlib.patches.Rectangle at 0x7f8f7df93a00>,
      <matplotlib.patches.Rectangle at 0x7f8f7df93ee0>,
      <matplotlib.patches.Rectangle at 0x7f8f7dfa2400>,
      <matplotlib.patches.Rectangle at 0x7f8f7dfa28e0>,
      <matplotlib.patches.Rectangle at 0x7f8f7dfa2dc0>,
      <matplotlib.patches.Rectangle at 0x7f8f7dfb12e0>,
      <matplotlib.patches.Rectangle at 0x7f8f7dfb17c0>,
      <matplotlib.patches.Rectangle at 0x7f8f7dfb1ca0>,
      <matplotlib.patches.Rectangle at 0x7f8f7dfbd0a0>,
      <matplotlib.patches.Rectangle at 0x7f8f7dfbd6a0>,
      <matplotlib.patches.Rectangle at 0x7f8f7dfbdb80>,
      <matplotlib.patches.Rectangle at 0x7f8f7dfbdf40>,
      <matplotlib.patches.Rectangle at 0x7f8f7dfc9580>,
      <matplotlib.patches.Rectangle at 0x7f8f7dfc9a60>,
      <matplotlib.patches.Rectangle at 0x7f8f7dfc9f40>,
      <matplotlib.patches.Rectangle at 0x7f8f7dfd9460>,
      <matplotlib.patches.Rectangle at 0x7f8f7dfd9940>,
      <matplotlib.patches.Rectangle at 0x7f8f7dfd9e20>,
      <matplotlib.patches.Rectangle at 0x7f8f7dfe7340>,
      <matplotlib.patches.Rectangle at 0x7f8f7dfe7820>,
      <matplotlib.patches.Rectangle at 0x7f8f7dfe7d00>,
      <matplotlib.patches.Rectangle at 0x7f8f7dff5220>,
      <matplotlib.patches.Rectangle at 0x7f8f7dff5700>,
      <matplotlib.patches.Rectangle at 0x7f8f7dff5be0>,
      <matplotlib.patches.Rectangle at 0x7f8f7dff5fa0>,
      <matplotlib.patches.Rectangle at 0x7f8f7e0005e0>,
      <matplotlib.patches.Rectangle at 0x7f8f7e000ac0>,
      <matplotlib.patches.Rectangle at 0x7f8f7e000fa0>,
      <matplotlib.patches.Rectangle at 0x7f8f7e00f4c0>,
      <matplotlib.patches.Rectangle at 0x7f8f7e00f9a0>,
      <matplotlib.patches.Rectangle at 0x7f8f7e00fe80>,
      <matplotlib.patches.Rectangle at 0x7f8f7e01d3a0>,
      <matplotlib.patches.Rectangle at 0x7f8f7e01d880>,
      <matplotlib.patches.Rectangle at 0x7f8f7e01dd60>,
      <matplotlib.patches.Rectangle at 0x7f8f7e02c160>,
      <matplotlib.patches.Rectangle at 0x7f8f7e02c760>,
      <matplotlib.patches.Rectangle at 0x7f8f7e02cc40>,
      <matplotlib.patches.Rectangle at 0x7f8f7e038160>,
      <matplotlib.patches.Rectangle at 0x7f8f7e038640>,
      <matplotlib.patches.Rectangle at 0x7f8f7e038b20>,
      <matplotlib.patches.Rectangle at 0x7f8f7e038f40>,
      <matplotlib.patches.Rectangle at 0x7f8f7e048520>,
      <matplotlib.patches.Rectangle at 0x7f8f7e048a00>,
      <matplotlib.patches.Rectangle at 0x7f8f7e048ee0>,
      <matplotlib.patches.Rectangle at 0x7f8f7e053400>,
      <matplotlib.patches.Rectangle at 0x7f8f7e0538e0>,
      <matplotlib.patches.Rectangle at 0x7f8f7e053dc0>,
      <matplotlib.patches.Rectangle at 0x7f8f7e0602e0>,
      <matplotlib.patches.Rectangle at 0x7f8f7e0607c0>,
      <matplotlib.patches.Rectangle at 0x7f8f7e060ca0>,
      <matplotlib.patches.Rectangle at 0x7f8f7e06e0a0>,
      <matplotlib.patches.Rectangle at 0x7f8f7e06e6a0>,
      <matplotlib.patches.Rectangle at 0x7f8f7e06eb80>,
      <matplotlib.patches.Rectangle at 0x7f8f7e06ef40>,
      <matplotlib.patches.Rectangle at 0x7f8f7e07d580>,
      <matplotlib.patches.Rectangle at 0x7f8f7e07da60>,
      <matplotlib.patches.Rectangle at 0x7f8f7e07df40>,
      <matplotlib.patches.Rectangle at 0x7f8f7e088460>,
      <matplotlib.patches.Rectangle at 0x7f8f7e088940>,
      <matplotlib.patches.Rectangle at 0x7f8f7e088e20>,
      <matplotlib.patches.Rectangle at 0x7f8f7e097340>,
      <matplotlib.patches.Rectangle at 0x7f8f7e097820>,
      <matplotlib.patches.Rectangle at 0x7f8f7e097d00>,
      <matplotlib.patches.Rectangle at 0x7f8f7e0a6100>,
      <matplotlib.patches.Rectangle at 0x7f8f7e0a6700>,
      <matplotlib.patches.Rectangle at 0x7f8f7e0a6be0>,
      <matplotlib.patches.Rectangle at 0x7f8f7e0a6fa0>,
      <matplotlib.patches.Rectangle at 0x7f8f7e0b25e0>,
      <matplotlib.patches.Rectangle at 0x7f8f7e0b2ac0>,
      <matplotlib.patches.Rectangle at 0x7f8f7e0b2fa0>,
      <matplotlib.patches.Rectangle at 0x7f8f7e0c14c0>,
      <matplotlib.patches.Rectangle at 0x7f8f7e0c19a0>,
      <matplotlib.patches.Rectangle at 0x7f8f7e0c1e80>,
      <matplotlib.patches.Rectangle at 0x7f8f7e0cf3a0>,
      <matplotlib.patches.Rectangle at 0x7f8f7e0cf880>,
      <matplotlib.patches.Rectangle at 0x7f8f7e0cfd60>,
      <matplotlib.patches.Rectangle at 0x7f8f7e0dd160>,
      <matplotlib.patches.Rectangle at 0x7f8f7e0dd760>,
      <matplotlib.patches.Rectangle at 0x7f8f7e0ddc40>,
      <matplotlib.patches.Rectangle at 0x7f8f7e0eb160>,
      <matplotlib.patches.Rectangle at 0x7f8f7e0eb640>,
      <matplotlib.patches.Rectangle at 0x7f8f7e0ebb20>,
      <matplotlib.patches.Rectangle at 0x7f8f7e0ebf40>,
      <matplotlib.patches.Rectangle at 0x7f8f7e0f6520>,
      <matplotlib.patches.Rectangle at 0x7f8f7e0f6a00>,
      <matplotlib.patches.Rectangle at 0x7f8f7e0f6ee0>,
      <matplotlib.patches.Rectangle at 0x7f8f7e105400>,
      <matplotlib.patches.Rectangle at 0x7f8f7e1058e0>,
      <matplotlib.patches.Rectangle at 0x7f8f7e105dc0>,
      <matplotlib.patches.Rectangle at 0x7f8f7e1132e0>,
      <matplotlib.patches.Rectangle at 0x7f8f7e1137c0>,
      <matplotlib.patches.Rectangle at 0x7f8f7e113ca0>,
      <matplotlib.patches.Rectangle at 0x7f8f7e1200a0>,
      <matplotlib.patches.Rectangle at 0x7f8f7e1206a0>,
      <matplotlib.patches.Rectangle at 0x7f8f7e120b80>,
      <matplotlib.patches.Rectangle at 0x7f8f7e120f40>,
      <matplotlib.patches.Rectangle at 0x7f8f7e12f580>,
      <matplotlib.patches.Rectangle at 0x7f8f7e12fa60>,
      <matplotlib.patches.Rectangle at 0x7f8f7e12ff40>,
      <matplotlib.patches.Rectangle at 0x7f8f7e13a460>,
      <matplotlib.patches.Rectangle at 0x7f8f7e13a940>,
      <matplotlib.patches.Rectangle at 0x7f8f7e13ae20>,
      <matplotlib.patches.Rectangle at 0x7f8f7e147340>,
      <matplotlib.patches.Rectangle at 0x7f8f7e147820>,
      <matplotlib.patches.Rectangle at 0x7f8f7e147d00>,
      <matplotlib.patches.Rectangle at 0x7f8f7e157220>,
      <matplotlib.patches.Rectangle at 0x7f8f7e157700>,
      <matplotlib.patches.Rectangle at 0x7f8f7e157be0>,
      <matplotlib.patches.Rectangle at 0x7f8f7e157fd0>,
      <matplotlib.patches.Rectangle at 0x7f8f7e164610>,
      <matplotlib.patches.Rectangle at 0x7f8f7e164af0>,
      <matplotlib.patches.Rectangle at 0x7f8f7e164fd0>,
      <matplotlib.patches.Rectangle at 0x7f8f7e1714f0>,
      <matplotlib.patches.Rectangle at 0x7f8f7e1719d0>,
      <matplotlib.patches.Rectangle at 0x7f8f7e171eb0>,
      <matplotlib.patches.Rectangle at 0x7f8f7e1803d0>,
      <matplotlib.patches.Rectangle at 0x7f8f7e1808b0>,
      <matplotlib.patches.Rectangle at 0x7f8f7e180d90>,
      <matplotlib.patches.Rectangle at 0x7f8f7e18c2b0>,
      <matplotlib.patches.Rectangle at 0x7f8f7e18c790>,
      <matplotlib.patches.Rectangle at 0x7f8f7e18cc70>,
      <matplotlib.patches.Rectangle at 0x7f8fa852a190>,
      <matplotlib.patches.Rectangle at 0x7f8fa852a670>,
      <matplotlib.patches.Rectangle at 0x7f8fa852ab50>,
      <matplotlib.patches.Rectangle at 0x7f8fa852afa0>,
      <matplotlib.patches.Rectangle at 0x7f8fa8537550>,
      <matplotlib.patches.Rectangle at 0x7f8fa8537a30>,
      <matplotlib.patches.Rectangle at 0x7f8fa8537f10>,
      <matplotlib.patches.Rectangle at 0x7f8fa8545430>,
      <matplotlib.patches.Rectangle at 0x7f8fa8545910>,
      <matplotlib.patches.Rectangle at 0x7f8fa8545df0>,
      <matplotlib.patches.Rectangle at 0x7f8fa8556310>,
      <matplotlib.patches.Rectangle at 0x7f8fa85567f0>,
      <matplotlib.patches.Rectangle at 0x7f8fa8556cd0>,
      <matplotlib.patches.Rectangle at 0x7f8fa85630d0>,
      <matplotlib.patches.Rectangle at 0x7f8fa85636d0>,
      <matplotlib.patches.Rectangle at 0x7f8fa8563bb0>,
      <matplotlib.patches.Rectangle at 0x7f8fa8563fd0>,
      <matplotlib.patches.Rectangle at 0x7f8fa856e5b0>,
      <matplotlib.patches.Rectangle at 0x7f8fa856ea90>,
      <matplotlib.patches.Rectangle at 0x7f8fa856ef70>,
      <matplotlib.patches.Rectangle at 0x7f8fa857c490>,
      <matplotlib.patches.Rectangle at 0x7f8fa857c970>,
      <matplotlib.patches.Rectangle at 0x7f8fa857ce50>,
      <matplotlib.patches.Rectangle at 0x7f8fa858b370>,
      <matplotlib.patches.Rectangle at 0x7f8fa858b850>,
      <matplotlib.patches.Rectangle at 0x7f8fa858bd30>,
      <matplotlib.patches.Rectangle at 0x7f8fa8599250>,
      <matplotlib.patches.Rectangle at 0x7f8fa8599730>,
      <matplotlib.patches.Rectangle at 0x7f8fa8599c10>,
      <matplotlib.patches.Rectangle at 0x7f8fa8599fd0>,
      <matplotlib.patches.Rectangle at 0x7f8fa85a5610>,
      <matplotlib.patches.Rectangle at 0x7f8fa85a5af0>,
      <matplotlib.patches.Rectangle at 0x7f8fa85a5fd0>,
      <matplotlib.patches.Rectangle at 0x7f8fa85b24f0>,
      <matplotlib.patches.Rectangle at 0x7f8fa85b29d0>,
      <matplotlib.patches.Rectangle at 0x7f8fa85b2eb0>,
      <matplotlib.patches.Rectangle at 0x7f8fa85c13d0>,
      <matplotlib.patches.Rectangle at 0x7f8fa85c18b0>,
      <matplotlib.patches.Rectangle at 0x7f8fa85c1d90>,
      <matplotlib.patches.Rectangle at 0x7f8fa85ce2b0>,
      <matplotlib.patches.Rectangle at 0x7f8fa85ce790>,
      <matplotlib.patches.Rectangle at 0x7f8fa85cec70>,
      <matplotlib.patches.Rectangle at 0x7f8fa85dc190>,
      <matplotlib.patches.Rectangle at 0x7f8fa85dc670>,
      <matplotlib.patches.Rectangle at 0x7f8fa85dcb50>,
      <matplotlib.patches.Rectangle at 0x7f8fa85dcfa0>,
      <matplotlib.patches.Rectangle at 0x7f8fa85ea550>,
      <matplotlib.patches.Rectangle at 0x7f8fa85eaa30>,
      <matplotlib.patches.Rectangle at 0x7f8fa85eaf10>,
      <matplotlib.patches.Rectangle at 0x7f8fa85f8430>,
      <matplotlib.patches.Rectangle at 0x7f8fa85f8910>,
      <matplotlib.patches.Rectangle at 0x7f8fa85f8e20>,
      <matplotlib.patches.Rectangle at 0x7f8fa8604340>,
      <matplotlib.patches.Rectangle at 0x7f8fa8604820>,
      <matplotlib.patches.Rectangle at 0x7f8fa8604d00>,
      <matplotlib.patches.Rectangle at 0x7f8fa8612100>,
      <matplotlib.patches.Rectangle at 0x7f8fa8612700>,
      <matplotlib.patches.Rectangle at 0x7f8fa8612be0>,
      <matplotlib.patches.Rectangle at 0x7f8fa8612fa0>,
      <matplotlib.patches.Rectangle at 0x7f8fa86215e0>,
      <matplotlib.patches.Rectangle at 0x7f8fa8621ac0>,
      <matplotlib.patches.Rectangle at 0x7f8fa8621fa0>,
      <matplotlib.patches.Rectangle at 0x7f8fa862d4c0>,
      <matplotlib.patches.Rectangle at 0x7f8fa862d9a0>,
      <matplotlib.patches.Rectangle at 0x7f8fa862de80>,
      <matplotlib.patches.Rectangle at 0x7f8fa863c3a0>,
      <matplotlib.patches.Rectangle at 0x7f8fa863c880>,
      <matplotlib.patches.Rectangle at 0x7f8fa863cd60>,
      <matplotlib.patches.Rectangle at 0x7f8fa864b280>,
      <matplotlib.patches.Rectangle at 0x7f8fa864b760>,
      <matplotlib.patches.Rectangle at 0x7f8fa864bc40>,
      <matplotlib.patches.Rectangle at 0x7f8fa8654040>,
      <matplotlib.patches.Rectangle at 0x7f8fa8654640>,
      <matplotlib.patches.Rectangle at 0x7f8fa8654b20>,
      <matplotlib.patches.Rectangle at 0x7f8fa8654f40>,
      <matplotlib.patches.Rectangle at 0x7f8fa8665520>,
      <matplotlib.patches.Rectangle at 0x7f8fa8665a00>,
      <matplotlib.patches.Rectangle at 0x7f8fa8665ee0>,
      <matplotlib.patches.Rectangle at 0x7f8fa8673400>,
      <matplotlib.patches.Rectangle at 0x7f8fa86738e0>,
      <matplotlib.patches.Rectangle at 0x7f8fa8673dc0>,
      <matplotlib.patches.Rectangle at 0x7f8fa867e2e0>])




    
![png](output_20_1.png)
    



```python
#Calculating returns on close
Tesla['returns'] = (Tesla['Close']/Tesla['Close'].shift(1)) - 1
```


```python
#Calculating returns on close
Amazon['returns'] = (Amazon['Close']/Amazon['Close'].shift(1)) - 1
```


```python
#Return hist
Tesla['returns'].hist(bins=50)
```




    <AxesSubplot:>




    
![png](output_23_1.png)
    



```python
#Return hist
Amazon['returns'].hist(bins=50)

```




    <AxesSubplot:>




    
![png](output_24_1.png)
    



```python
#Overlaying returns
Amazon['returns'].hist(bins=50, alpha = .2, figsize = (13,6), label = 'Bitcoin')
Tesla['returns'].hist(bins=50, alpha = .2, label = 'Tesla')
plt.legend()
```




    <matplotlib.legend.Legend at 0x7f8f7d3b42b0>




    
![png](output_25_1.png)
    



```python
#Amazon day over day return
Amazon['returns'].plot( label = 'Bitcoin')
```




    <AxesSubplot:xlabel='Date'>




    
![png](output_26_1.png)
    



```python
#Day over day return 
Amazon['returns'].plot( label = 'Bitcoin')
Tesla['returns'].plot(label = 'Tesla')
plt.legend()

```




    <matplotlib.legend.Legend at 0x7f8f7bfc0bb0>




    
![png](output_27_1.png)
    



```python
#Centeral tendency
box_df = pd.concat([Tesla['returns'], Amazon['returns']], axis = 1)
box_df.columns=['Tesla Returns', 'Amazon Returns']
box_df.plot(kind= "box", figsize = (16,6))
```




    <AxesSubplot:>




    
![png](output_28_1.png)
    



```python
#Scatter matrix 
scatter_matrix(box_df, figsize = (8,8), hist_kwds = {'bins':50}, alpha = 0.3)
```




    array([[<AxesSubplot:xlabel='Tesla Returns', ylabel='Tesla Returns'>,
            <AxesSubplot:xlabel='Amazon Returns', ylabel='Tesla Returns'>],
           [<AxesSubplot:xlabel='Tesla Returns', ylabel='Amazon Returns'>,
            <AxesSubplot:xlabel='Amazon Returns', ylabel='Amazon Returns'>]],
          dtype=object)




    
![png](output_29_1.png)
    



```python
#Calculating cumulative return
Tesla['Cumulative Return'] = (1+ Tesla['returns']).cumprod()
Amazon['Cumulative Return'] = (1+ Amazon['returns']).cumprod()
```


```python
#Charting cumulative return
Tesla['Cumulative Return']. plot(label = 'Tesla' , figsize = (15,7))
Amazon['Cumulative Return'].plot(label = "Amazon")
plt.legend()
```




    <matplotlib.legend.Legend at 0x7f8f7d7e5b80>




    
![png](output_31_1.png)
    

