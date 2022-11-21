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
start = datetime.datetime(2010,6,29)
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
      <th>2010-06-29</th>
      <td>5.8240</td>
      <td>5.3005</td>
      <td>5.8130</td>
      <td>5.4305</td>
      <td>257326000.0</td>
      <td>5.4305</td>
    </tr>
    <tr>
      <th>2010-06-30</th>
      <td>5.6340</td>
      <td>5.4055</td>
      <td>5.4290</td>
      <td>5.4630</td>
      <td>194814000.0</td>
      <td>5.4630</td>
    </tr>
    <tr>
      <th>2010-07-01</th>
      <td>5.5845</td>
      <td>5.3350</td>
      <td>5.4450</td>
      <td>5.5480</td>
      <td>170596000.0</td>
      <td>5.5480</td>
    </tr>
    <tr>
      <th>2010-07-02</th>
      <td>5.5645</td>
      <td>5.4280</td>
      <td>5.5460</td>
      <td>5.4570</td>
      <td>89542000.0</td>
      <td>5.4570</td>
    </tr>
    <tr>
      <th>2010-07-06</th>
      <td>5.6265</td>
      <td>5.4500</td>
      <td>5.5325</td>
      <td>5.5030</td>
      <td>104386000.0</td>
      <td>5.5030</td>
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
      <th>2010-06-29</th>
      <td>1.666667</td>
      <td>1.169333</td>
      <td>1.266667</td>
      <td>1.592667</td>
      <td>281494500.0</td>
      <td>1.592667</td>
    </tr>
    <tr>
      <th>2010-06-30</th>
      <td>2.028000</td>
      <td>1.553333</td>
      <td>1.719333</td>
      <td>1.588667</td>
      <td>257806500.0</td>
      <td>1.588667</td>
    </tr>
    <tr>
      <th>2010-07-01</th>
      <td>1.728000</td>
      <td>1.351333</td>
      <td>1.666667</td>
      <td>1.464000</td>
      <td>123282000.0</td>
      <td>1.464000</td>
    </tr>
    <tr>
      <th>2010-07-02</th>
      <td>1.540000</td>
      <td>1.247333</td>
      <td>1.533333</td>
      <td>1.280000</td>
      <td>77097000.0</td>
      <td>1.280000</td>
    </tr>
    <tr>
      <th>2010-07-06</th>
      <td>1.333333</td>
      <td>1.055333</td>
      <td>1.333333</td>
      <td>1.074000</td>
      <td>103003500.0</td>
      <td>1.074000</td>
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
plt.title('Tesla Volume')
plt.show()
```


    
![png](output_6_0.png)
    



```python
#Volume histogram
Amazon['Volume'].plot.hist(bins = 40)
plt.title('Amazon Volume')
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




    3097




```python
#Past 100 days of trading
Amazon.iloc[603:703].plot()
```




    <AxesSubplot:xlabel='Date'>




    
![png](output_10_1.png)
    



```python
#Past 100 days of trading
Tesla.iloc[603:703].plot()
plt.title('Tesla Past 100 Days')

```




    Text(0.5, 1.0, 'Tesla Past 100 Days')




    
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
plt.title('Trade Volume')
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




    <matplotlib.legend.Legend at 0x7f8fb9aadeb0>




    
![png](output_16_1.png)
    



```python
#Rolling average for Tesla
Tesla['Open'].plot(figsize = (15,7))
Tesla['MA50'] = Tesla['Open'].rolling(100).mean()
Tesla['MA50'].plot(label = 'MA50')
plt.title('Rolling average for Tesla')
plt.legend()
```




    <matplotlib.legend.Legend at 0x7f8f7e63b8b0>




    
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




    ([<matplotlib.lines.Line2D at 0x7f8fb94dd130>,
      <matplotlib.lines.Line2D at 0x7f8fa89cd460>,
      <matplotlib.lines.Line2D at 0x7f8fa89cd940>,
      <matplotlib.lines.Line2D at 0x7f8fa89cde20>,
      <matplotlib.lines.Line2D at 0x7f8f6ad94340>,
      <matplotlib.lines.Line2D at 0x7f8f6ad94820>,
      <matplotlib.lines.Line2D at 0x7f8f6ad94d00>,
      <matplotlib.lines.Line2D at 0x7f8f6adc7220>,
      <matplotlib.lines.Line2D at 0x7f8f6adc7700>,
      <matplotlib.lines.Line2D at 0x7f8f6adc7be0>,
      <matplotlib.lines.Line2D at 0x7f8f6ada4100>,
      <matplotlib.lines.Line2D at 0x7f8f6ada45e0>,
      <matplotlib.lines.Line2D at 0x7f8f6ada4ac0>,
      <matplotlib.lines.Line2D at 0x7f8f6ada4fa0>,
      <matplotlib.lines.Line2D at 0x7f8f6adb84c0>,
      <matplotlib.lines.Line2D at 0x7f8f6adb89a0>,
      <matplotlib.lines.Line2D at 0x7f8f6adb8e80>,
      <matplotlib.lines.Line2D at 0x7f8f6ad8d3a0>,
      <matplotlib.lines.Line2D at 0x7f8f6ad8d880>,
      <matplotlib.lines.Line2D at 0x7f8f6ad8dd60>,
      <matplotlib.lines.Line2D at 0x7f8f6af7e280>,
      <matplotlib.lines.Line2D at 0x7f8f6af7e760>,
      <matplotlib.lines.Line2D at 0x7f8f6af7ec40>,
      <matplotlib.lines.Line2D at 0x7f8f6af64160>,
      <matplotlib.lines.Line2D at 0x7f8f6af64640>,
      <matplotlib.lines.Line2D at 0x7f8f6af64b20>,
      <matplotlib.lines.Line2D at 0x7f8f6af64fa0>,
      <matplotlib.lines.Line2D at 0x7f8f6af67520>,
      <matplotlib.lines.Line2D at 0x7f8f6af67a00>,
      <matplotlib.lines.Line2D at 0x7f8f6af67ee0>,
      <matplotlib.lines.Line2D at 0x7f8f6af8d400>,
      <matplotlib.lines.Line2D at 0x7f8f6af8d8e0>,
      <matplotlib.lines.Line2D at 0x7f8f6af8ddc0>,
      <matplotlib.lines.Line2D at 0x7f8f89e622e0>,
      <matplotlib.lines.Line2D at 0x7f8f89e627c0>,
      <matplotlib.lines.Line2D at 0x7f8f89e62ca0>,
      <matplotlib.lines.Line2D at 0x7f8f89e731c0>,
      <matplotlib.lines.Line2D at 0x7f8f89e736a0>,
      <matplotlib.lines.Line2D at 0x7f8f89e73b80>,
      <matplotlib.lines.Line2D at 0x7f8f89e73fd0>,
      <matplotlib.lines.Line2D at 0x7f8f89e64580>,
      <matplotlib.lines.Line2D at 0x7f8f89e64a60>,
      <matplotlib.lines.Line2D at 0x7f8f89e64f40>,
      <matplotlib.lines.Line2D at 0x7f8f89e4f460>,
      <matplotlib.lines.Line2D at 0x7f8f89e4f940>,
      <matplotlib.lines.Line2D at 0x7f8f89e4fe20>,
      <matplotlib.lines.Line2D at 0x7f8f89e50340>,
      <matplotlib.lines.Line2D at 0x7f8f89e50820>,
      <matplotlib.lines.Line2D at 0x7f8f89e50d00>,
      <matplotlib.lines.Line2D at 0x7f8f6b004220>,
      <matplotlib.lines.Line2D at 0x7f8f6b004700>,
      <matplotlib.lines.Line2D at 0x7f8f6b004be0>,
      <matplotlib.lines.Line2D at 0x7f8f6aff8100>,
      <matplotlib.lines.Line2D at 0x7f8f6aff85e0>,
      <matplotlib.lines.Line2D at 0x7f8f6aff8ac0>,
      <matplotlib.lines.Line2D at 0x7f8f6aff8fa0>,
      <matplotlib.lines.Line2D at 0x7f8f6afe84c0>,
      <matplotlib.lines.Line2D at 0x7f8f6afe89a0>,
      <matplotlib.lines.Line2D at 0x7f8f6afe8e80>,
      <matplotlib.lines.Line2D at 0x7f8f6afda3a0>,
      <matplotlib.lines.Line2D at 0x7f8f6afda880>,
      <matplotlib.lines.Line2D at 0x7f8f6afdad60>,
      <matplotlib.lines.Line2D at 0x7f8f9936e280>,
      <matplotlib.lines.Line2D at 0x7f8f9936e760>,
      <matplotlib.lines.Line2D at 0x7f8f9936ec40>,
      <matplotlib.lines.Line2D at 0x7f8f99351160>,
      <matplotlib.lines.Line2D at 0x7f8f99351640>,
      <matplotlib.lines.Line2D at 0x7f8f99351b20>,
      <matplotlib.lines.Line2D at 0x7f8f99351fa0>,
      <matplotlib.lines.Line2D at 0x7f8f99353520>,
      <matplotlib.lines.Line2D at 0x7f8f99353a00>,
      <matplotlib.lines.Line2D at 0x7f8f99353ee0>,
      <matplotlib.lines.Line2D at 0x7f8f9934c400>,
      <matplotlib.lines.Line2D at 0x7f8f9934c8e0>,
      <matplotlib.lines.Line2D at 0x7f8f9934cdc0>,
      <matplotlib.lines.Line2D at 0x7f8f9934e2e0>,
      <matplotlib.lines.Line2D at 0x7f8f9934e7c0>,
      <matplotlib.lines.Line2D at 0x7f8f9934eca0>,
      <matplotlib.lines.Line2D at 0x7f8fa8b1a1c0>,
      <matplotlib.lines.Line2D at 0x7f8fa8b1a6a0>,
      <matplotlib.lines.Line2D at 0x7f8fa8b1ab80>,
      <matplotlib.lines.Line2D at 0x7f8fa8b1afd0>,
      <matplotlib.lines.Line2D at 0x7f8fa8b2f580>,
      <matplotlib.lines.Line2D at 0x7f8fa8b2fa60>,
      <matplotlib.lines.Line2D at 0x7f8fa8b2ff40>,
      <matplotlib.lines.Line2D at 0x7f8fa8b2e460>,
      <matplotlib.lines.Line2D at 0x7f8fa8b2e940>,
      <matplotlib.lines.Line2D at 0x7f8fa8b2ee20>,
      <matplotlib.lines.Line2D at 0x7f8fa8b0a340>,
      <matplotlib.lines.Line2D at 0x7f8fa8b0a820>,
      <matplotlib.lines.Line2D at 0x7f8fa8b0ad00>,
      <matplotlib.lines.Line2D at 0x7f8fa8b18220>,
      <matplotlib.lines.Line2D at 0x7f8fa8b18700>,
      <matplotlib.lines.Line2D at 0x7f8fa8b18be0>,
      <matplotlib.lines.Line2D at 0x7f8fa8ba3100>,
      <matplotlib.lines.Line2D at 0x7f8fa8ba35e0>,
      <matplotlib.lines.Line2D at 0x7f8fa8ba3ac0>,
      <matplotlib.lines.Line2D at 0x7f8fa8ba3fa0>,
      <matplotlib.lines.Line2D at 0x7f8fa8b964c0>,
      <matplotlib.lines.Line2D at 0x7f8fa8b969a0>,
      <matplotlib.lines.Line2D at 0x7f8fa8b96e80>,
      <matplotlib.lines.Line2D at 0x7f8fa8b883a0>,
      <matplotlib.lines.Line2D at 0x7f8fa8b88880>,
      <matplotlib.lines.Line2D at 0x7f8fa8b88d60>,
      <matplotlib.lines.Line2D at 0x7f8fa8b7a280>,
      <matplotlib.lines.Line2D at 0x7f8fa8b7a760>,
      <matplotlib.lines.Line2D at 0x7f8fa8b7ac40>,
      <matplotlib.lines.Line2D at 0x7f8f99769160>,
      <matplotlib.lines.Line2D at 0x7f8f99769640>,
      <matplotlib.lines.Line2D at 0x7f8f99769b20>,
      <matplotlib.lines.Line2D at 0x7f8f99769fa0>,
      <matplotlib.lines.Line2D at 0x7f8f99777520>,
      <matplotlib.lines.Line2D at 0x7f8f99777a00>,
      <matplotlib.lines.Line2D at 0x7f8f99777ee0>,
      <matplotlib.lines.Line2D at 0x7f8f99784400>,
      <matplotlib.lines.Line2D at 0x7f8f997848e0>,
      <matplotlib.lines.Line2D at 0x7f8f99784dc0>,
      <matplotlib.lines.Line2D at 0x7f8f997912e0>,
      <matplotlib.lines.Line2D at 0x7f8f997917c0>,
      <matplotlib.lines.Line2D at 0x7f8f99791ca0>,
      <matplotlib.lines.Line2D at 0x7f8f997a11c0>,
      <matplotlib.lines.Line2D at 0x7f8f997a16a0>,
      <matplotlib.lines.Line2D at 0x7f8f997a1b80>,
      <matplotlib.lines.Line2D at 0x7f8f997a1fd0>,
      <matplotlib.lines.Line2D at 0x7f8f997ae580>,
      <matplotlib.lines.Line2D at 0x7f8f997aea60>,
      <matplotlib.lines.Line2D at 0x7f8f997aef40>,
      <matplotlib.lines.Line2D at 0x7f8f997bd460>,
      <matplotlib.lines.Line2D at 0x7f8f997bd940>,
      <matplotlib.lines.Line2D at 0x7f8f997bde20>,
      <matplotlib.lines.Line2D at 0x7f8f997c9340>,
      <matplotlib.lines.Line2D at 0x7f8f997c9820>,
      <matplotlib.lines.Line2D at 0x7f8f997c9d00>,
      <matplotlib.lines.Line2D at 0x7f8f997d5220>,
      <matplotlib.lines.Line2D at 0x7f8f997d5700>,
      <matplotlib.lines.Line2D at 0x7f8f997d5be0>,
      <matplotlib.lines.Line2D at 0x7f8f997e5100>,
      <matplotlib.lines.Line2D at 0x7f8f997e55e0>,
      <matplotlib.lines.Line2D at 0x7f8f997e5ac0>,
      <matplotlib.lines.Line2D at 0x7f8f997e5fa0>,
      <matplotlib.lines.Line2D at 0x7f8f997f44c0>,
      <matplotlib.lines.Line2D at 0x7f8f997f49a0>,
      <matplotlib.lines.Line2D at 0x7f8f997f4e80>,
      <matplotlib.lines.Line2D at 0x7f8f997fe3a0>,
      <matplotlib.lines.Line2D at 0x7f8f997fe880>,
      <matplotlib.lines.Line2D at 0x7f8f997fed60>,
      <matplotlib.lines.Line2D at 0x7f8f9980e280>,
      <matplotlib.lines.Line2D at 0x7f8f9980e760>,
      <matplotlib.lines.Line2D at 0x7f8f9980ec40>,
      <matplotlib.lines.Line2D at 0x7f8f9981c160>,
      <matplotlib.lines.Line2D at 0x7f8f9981c640>,
      <matplotlib.lines.Line2D at 0x7f8f9981cb20>,
      <matplotlib.lines.Line2D at 0x7f8f9981cfa0>,
      <matplotlib.lines.Line2D at 0x7f8f99f21520>,
      <matplotlib.lines.Line2D at 0x7f8f99f21a00>,
      <matplotlib.lines.Line2D at 0x7f8f99f21ee0>,
      <matplotlib.lines.Line2D at 0x7f8f99f2e400>,
      <matplotlib.lines.Line2D at 0x7f8f99f2e8e0>,
      <matplotlib.lines.Line2D at 0x7f8f99f2edc0>,
      <matplotlib.lines.Line2D at 0x7f8f99f3c2e0>,
      <matplotlib.lines.Line2D at 0x7f8f99f3c7c0>,
      <matplotlib.lines.Line2D at 0x7f8f99f3cca0>,
      <matplotlib.lines.Line2D at 0x7f8f99f491c0>,
      <matplotlib.lines.Line2D at 0x7f8f99f496a0>,
      <matplotlib.lines.Line2D at 0x7f8f99f49b80>,
      <matplotlib.lines.Line2D at 0x7f8f99f49fd0>,
      <matplotlib.lines.Line2D at 0x7f8f99f58580>,
      <matplotlib.lines.Line2D at 0x7f8f99f58a60>,
      <matplotlib.lines.Line2D at 0x7f8f99f58f40>,
      <matplotlib.lines.Line2D at 0x7f8f99f64460>,
      <matplotlib.lines.Line2D at 0x7f8f99f64940>,
      <matplotlib.lines.Line2D at 0x7f8f99f64e20>,
      <matplotlib.lines.Line2D at 0x7f8f99f73340>,
      <matplotlib.lines.Line2D at 0x7f8f99f73820>,
      <matplotlib.lines.Line2D at 0x7f8f99f73d00>,
      <matplotlib.lines.Line2D at 0x7f8f99f7f220>,
      <matplotlib.lines.Line2D at 0x7f8f99f7f700>,
      <matplotlib.lines.Line2D at 0x7f8f99f7fbe0>,
      <matplotlib.lines.Line2D at 0x7f8f99f8e100>,
      <matplotlib.lines.Line2D at 0x7f8f99f8e5e0>,
      <matplotlib.lines.Line2D at 0x7f8f99f8eac0>,
      <matplotlib.lines.Line2D at 0x7f8f99f8efa0>,
      <matplotlib.lines.Line2D at 0x7f8f99f9d4c0>,
      <matplotlib.lines.Line2D at 0x7f8f99f9d9a0>,
      <matplotlib.lines.Line2D at 0x7f8f99f9de80>,
      <matplotlib.lines.Line2D at 0x7f8f99fa83a0>,
      <matplotlib.lines.Line2D at 0x7f8f99fa8880>,
      <matplotlib.lines.Line2D at 0x7f8f99fa8d60>,
      <matplotlib.lines.Line2D at 0x7f8f99fb6280>,
      <matplotlib.lines.Line2D at 0x7f8f99fb6760>,
      <matplotlib.lines.Line2D at 0x7f8f99fb6c40>,
      <matplotlib.lines.Line2D at 0x7f8f99fc5160>,
      <matplotlib.lines.Line2D at 0x7f8f99fc5640>,
      <matplotlib.lines.Line2D at 0x7f8f99fc5b20>,
      <matplotlib.lines.Line2D at 0x7f8f99fc5fa0>,
      <matplotlib.lines.Line2D at 0x7f8f99fd3520>,
      <matplotlib.lines.Line2D at 0x7f8f99fd3a00>,
      <matplotlib.lines.Line2D at 0x7f8f99fd3ee0>,
      <matplotlib.lines.Line2D at 0x7f8f99fde400>,
      <matplotlib.lines.Line2D at 0x7f8f99fde8e0>,
      <matplotlib.lines.Line2D at 0x7f8f99fdedc0>,
      <matplotlib.lines.Line2D at 0x7f8f99fed2e0>,
      <matplotlib.lines.Line2D at 0x7f8f99fed7c0>,
      <matplotlib.lines.Line2D at 0x7f8f99fedca0>,
      <matplotlib.lines.Line2D at 0x7f8f99ffb1c0>,
      <matplotlib.lines.Line2D at 0x7f8f99ffb6a0>,
      <matplotlib.lines.Line2D at 0x7f8f99ffbb80>,
      <matplotlib.lines.Line2D at 0x7f8f99ffbfd0>,
      <matplotlib.lines.Line2D at 0x7f8f9a008580>,
      <matplotlib.lines.Line2D at 0x7f8f9a008a60>,
      <matplotlib.lines.Line2D at 0x7f8f9a008f40>,
      <matplotlib.lines.Line2D at 0x7f8f9a016460>,
      <matplotlib.lines.Line2D at 0x7f8f9a016940>,
      <matplotlib.lines.Line2D at 0x7f8f9a016e20>,
      <matplotlib.lines.Line2D at 0x7f8f9a023340>,
      <matplotlib.lines.Line2D at 0x7f8f9a023820>,
      <matplotlib.lines.Line2D at 0x7f8f9a023d00>,
      <matplotlib.lines.Line2D at 0x7f8f9a033220>,
      <matplotlib.lines.Line2D at 0x7f8f9a033700>,
      <matplotlib.lines.Line2D at 0x7f8f9a033be0>,
      <matplotlib.lines.Line2D at 0x7f8f9a040100>,
      <matplotlib.lines.Line2D at 0x7f8f9a0405e0>,
      <matplotlib.lines.Line2D at 0x7f8f9a040ac0>,
      <matplotlib.lines.Line2D at 0x7f8f9a040fa0>,
      <matplotlib.lines.Line2D at 0x7f8f9a04c4c0>,
      <matplotlib.lines.Line2D at 0x7f8f9a04c9a0>,
      <matplotlib.lines.Line2D at 0x7f8f9a04ce80>,
      <matplotlib.lines.Line2D at 0x7f8f9a05a3a0>,
      <matplotlib.lines.Line2D at 0x7f8f9a05a880>,
      <matplotlib.lines.Line2D at 0x7f8f9a05ad60>,
      <matplotlib.lines.Line2D at 0x7f8f9a069280>,
      <matplotlib.lines.Line2D at 0x7f8f9a069760>,
      <matplotlib.lines.Line2D at 0x7f8f9a069c40>,
      <matplotlib.lines.Line2D at 0x7f8f9a076160>,
      <matplotlib.lines.Line2D at 0x7f8f9a076640>,
      <matplotlib.lines.Line2D at 0x7f8f9a076b20>,
      <matplotlib.lines.Line2D at 0x7f8f9a076fa0>,
      <matplotlib.lines.Line2D at 0x7f8f9a084520>,
      <matplotlib.lines.Line2D at 0x7f8f9a084a00>,
      <matplotlib.lines.Line2D at 0x7f8f9a084ee0>,
      <matplotlib.lines.Line2D at 0x7f8f9a08f400>,
      <matplotlib.lines.Line2D at 0x7f8f9a08f8e0>,
      <matplotlib.lines.Line2D at 0x7f8f9a08fdc0>,
      <matplotlib.lines.Line2D at 0x7f8f9a09d2e0>,
      <matplotlib.lines.Line2D at 0x7f8f9a09d7c0>,
      <matplotlib.lines.Line2D at 0x7f8f9a09dca0>,
      <matplotlib.lines.Line2D at 0x7f8f9a0ad1c0>,
      <matplotlib.lines.Line2D at 0x7f8f9a0ad6a0>,
      <matplotlib.lines.Line2D at 0x7f8f9a0adb80>,
      <matplotlib.lines.Line2D at 0x7f8f9a0adfd0>,
      <matplotlib.lines.Line2D at 0x7f8f9a0ba580>,
      <matplotlib.lines.Line2D at 0x7f8f9a0baa60>,
      <matplotlib.lines.Line2D at 0x7f8f9a0baf40>,
      <matplotlib.lines.Line2D at 0x7f8f9a0c9460>,
      <matplotlib.lines.Line2D at 0x7f8f9a0c9940>,
      <matplotlib.lines.Line2D at 0x7f8f9a0c9e20>,
      <matplotlib.lines.Line2D at 0x7f8f9a0d5340>,
      <matplotlib.lines.Line2D at 0x7f8f9a0d5820>,
      <matplotlib.lines.Line2D at 0x7f8f9a0d5d00>,
      <matplotlib.lines.Line2D at 0x7f8f9a0e1220>,
      <matplotlib.lines.Line2D at 0x7f8f9a0e1700>,
      <matplotlib.lines.Line2D at 0x7f8f9a0e1be0>,
      <matplotlib.lines.Line2D at 0x7f8f9a0f1100>,
      <matplotlib.lines.Line2D at 0x7f8f9a0f15e0>,
      <matplotlib.lines.Line2D at 0x7f8f9a0f1ac0>,
      <matplotlib.lines.Line2D at 0x7f8f9a0f1fa0>,
      <matplotlib.lines.Line2D at 0x7f8f9a0ff4c0>,
      <matplotlib.lines.Line2D at 0x7f8f9a0ff9a0>,
      <matplotlib.lines.Line2D at 0x7f8f9a0ffe80>,
      <matplotlib.lines.Line2D at 0x7f8f9a10c3a0>,
      <matplotlib.lines.Line2D at 0x7f8f9a10c880>,
      <matplotlib.lines.Line2D at 0x7f8f9a10cd60>,
      <matplotlib.lines.Line2D at 0x7f8f9a11b280>,
      <matplotlib.lines.Line2D at 0x7f8f9a11b760>,
      <matplotlib.lines.Line2D at 0x7f8f9a11bc40>,
      <matplotlib.lines.Line2D at 0x7f8f9a128160>,
      <matplotlib.lines.Line2D at 0x7f8f9a128640>,
      <matplotlib.lines.Line2D at 0x7f8f9a128b20>,
      <matplotlib.lines.Line2D at 0x7f8f9a128fa0>,
      <matplotlib.lines.Line2D at 0x7f8f9a133520>,
      <matplotlib.lines.Line2D at 0x7f8f9a133a00>,
      <matplotlib.lines.Line2D at 0x7f8f9a133ee0>,
      <matplotlib.lines.Line2D at 0x7f8f9a142400>,
      <matplotlib.lines.Line2D at 0x7f8f9a1428e0>,
      <matplotlib.lines.Line2D at 0x7f8f9a142dc0>,
      <matplotlib.lines.Line2D at 0x7f8f9a1502e0>,
      <matplotlib.lines.Line2D at 0x7f8f9a1507c0>,
      <matplotlib.lines.Line2D at 0x7f8f9a150ca0>,
      <matplotlib.lines.Line2D at 0x7f8f9a15d1c0>,
      <matplotlib.lines.Line2D at 0x7f8f9a15d6a0>,
      <matplotlib.lines.Line2D at 0x7f8f9a15db80>,
      <matplotlib.lines.Line2D at 0x7f8f9a15dfd0>,
      <matplotlib.lines.Line2D at 0x7f8f9a16c580>,
      <matplotlib.lines.Line2D at 0x7f8f9a16ca60>,
      <matplotlib.lines.Line2D at 0x7f8f9a16cf40>,
      <matplotlib.lines.Line2D at 0x7f8f9a178460>,
      <matplotlib.lines.Line2D at 0x7f8f9a178940>,
      <matplotlib.lines.Line2D at 0x7f8f9a178e20>,
      <matplotlib.lines.Line2D at 0x7f8f9a184340>,
      <matplotlib.lines.Line2D at 0x7f8f9a184820>,
      <matplotlib.lines.Line2D at 0x7f8f9a184d00>,
      <matplotlib.lines.Line2D at 0x7f8f9a193220>,
      <matplotlib.lines.Line2D at 0x7f8f9a193700>,
      <matplotlib.lines.Line2D at 0x7f8f9a193be0>,
      <matplotlib.lines.Line2D at 0x7f8f9a1a1100>,
      <matplotlib.lines.Line2D at 0x7f8f9a1a15e0>,
      <matplotlib.lines.Line2D at 0x7f8f9a1a1ac0>,
      <matplotlib.lines.Line2D at 0x7f8f9a1a1fa0>,
      <matplotlib.lines.Line2D at 0x7f8f9a1b24c0>,
      <matplotlib.lines.Line2D at 0x7f8f9a1b29a0>,
      <matplotlib.lines.Line2D at 0x7f8f9a1b2e80>,
      <matplotlib.lines.Line2D at 0x7f8f9a1bc3a0>,
      <matplotlib.lines.Line2D at 0x7f8f9a1bc880>,
      <matplotlib.lines.Line2D at 0x7f8f9a1bcd60>,
      <matplotlib.lines.Line2D at 0x7f8f9a1ca280>,
      <matplotlib.lines.Line2D at 0x7f8f9a1ca760>,
      <matplotlib.lines.Line2D at 0x7f8f9a1cac40>,
      <matplotlib.lines.Line2D at 0x7f8f9a1d8160>,
      <matplotlib.lines.Line2D at 0x7f8f9a1d8640>,
      <matplotlib.lines.Line2D at 0x7f8f9a1d8b20>,
      <matplotlib.lines.Line2D at 0x7f8f9a1d8fa0>,
      <matplotlib.lines.Line2D at 0x7f8f9a1e7520>,
      <matplotlib.lines.Line2D at 0x7f8f9a1e7a00>,
      <matplotlib.lines.Line2D at 0x7f8f9a1e7ee0>,
      <matplotlib.lines.Line2D at 0x7f8f9a1f3400>,
      <matplotlib.lines.Line2D at 0x7f8f9a1f38e0>,
      <matplotlib.lines.Line2D at 0x7f8f9a1f3dc0>,
      <matplotlib.lines.Line2D at 0x7f8f9a2012e0>,
      <matplotlib.lines.Line2D at 0x7f8f9a2017c0>,
      <matplotlib.lines.Line2D at 0x7f8f9a201ca0>,
      <matplotlib.lines.Line2D at 0x7f8f9a20f1c0>,
      <matplotlib.lines.Line2D at 0x7f8f9a20f6a0>,
      <matplotlib.lines.Line2D at 0x7f8f9a20fb80>,
      <matplotlib.lines.Line2D at 0x7f8f9a20ffd0>,
      <matplotlib.lines.Line2D at 0x7f8f9a21c580>,
      <matplotlib.lines.Line2D at 0x7f8f9a21ca60>,
      <matplotlib.lines.Line2D at 0x7f8f9a21cf40>,
      <matplotlib.lines.Line2D at 0x7f8f9a22a460>,
      <matplotlib.lines.Line2D at 0x7f8f9a22a940>,
      <matplotlib.lines.Line2D at 0x7f8f9a22ae20>,
      <matplotlib.lines.Line2D at 0x7f8f9a237340>,
      <matplotlib.lines.Line2D at 0x7f8f9a237820>,
      <matplotlib.lines.Line2D at 0x7f8f9a237d00>,
      <matplotlib.lines.Line2D at 0x7f8f9a246220>,
      <matplotlib.lines.Line2D at 0x7f8f9a246700>,
      <matplotlib.lines.Line2D at 0x7f8f9a246be0>,
      <matplotlib.lines.Line2D at 0x7f8f9a253100>,
      <matplotlib.lines.Line2D at 0x7f8f9a2535e0>,
      <matplotlib.lines.Line2D at 0x7f8f9a253ac0>,
      <matplotlib.lines.Line2D at 0x7f8f9a253fa0>,
      <matplotlib.lines.Line2D at 0x7f8fa88e04c0>,
      <matplotlib.lines.Line2D at 0x7f8fa88e09a0>,
      <matplotlib.lines.Line2D at 0x7f8fa88e0e80>,
      <matplotlib.lines.Line2D at 0x7f8fa88ee3a0>,
      <matplotlib.lines.Line2D at 0x7f8fa88ee880>,
      <matplotlib.lines.Line2D at 0x7f8fa88eed60>,
      <matplotlib.lines.Line2D at 0x7f8fa88fb280>,
      <matplotlib.lines.Line2D at 0x7f8fa88fb760>,
      <matplotlib.lines.Line2D at 0x7f8fa88fbc40>,
      <matplotlib.lines.Line2D at 0x7f8fa890a160>,
      <matplotlib.lines.Line2D at 0x7f8fa890a640>,
      <matplotlib.lines.Line2D at 0x7f8fa890ab20>,
      <matplotlib.lines.Line2D at 0x7f8fa890afa0>,
      <matplotlib.lines.Line2D at 0x7f8fa8918520>,
      <matplotlib.lines.Line2D at 0x7f8fa8918a00>,
      <matplotlib.lines.Line2D at 0x7f8fa8918ee0>,
      <matplotlib.lines.Line2D at 0x7f8fa8d8c400>,
      <matplotlib.lines.Line2D at 0x7f8fa8d8c8e0>,
      <matplotlib.lines.Line2D at 0x7f8fa8d8cdc0>,
      <matplotlib.lines.Line2D at 0x7f8fa8d982e0>,
      <matplotlib.lines.Line2D at 0x7f8fa8d987c0>,
      <matplotlib.lines.Line2D at 0x7f8fa8d98ca0>,
      <matplotlib.lines.Line2D at 0x7f8fa8da91c0>,
      <matplotlib.lines.Line2D at 0x7f8fa8da96a0>,
      <matplotlib.lines.Line2D at 0x7f8fa8da9b80>,
      <matplotlib.lines.Line2D at 0x7f8fa8da9fd0>,
      <matplotlib.lines.Line2D at 0x7f8fa8db6580>,
      <matplotlib.lines.Line2D at 0x7f8fa8db6a60>,
      <matplotlib.lines.Line2D at 0x7f8fa8db6f40>,
      <matplotlib.lines.Line2D at 0x7f8fa8dc5460>,
      <matplotlib.lines.Line2D at 0x7f8fa8dc5940>,
      <matplotlib.lines.Line2D at 0x7f8fa8dc5e20>,
      <matplotlib.lines.Line2D at 0x7f8fa8730340>,
      <matplotlib.lines.Line2D at 0x7f8fa8730820>,
      <matplotlib.lines.Line2D at 0x7f8fa8730d00>,
      <matplotlib.lines.Line2D at 0x7f8fa873d220>,
      <matplotlib.lines.Line2D at 0x7f8fa873d700>,
      <matplotlib.lines.Line2D at 0x7f8fa873dbe0>,
      <matplotlib.lines.Line2D at 0x7f8fa874d100>,
      <matplotlib.lines.Line2D at 0x7f8fa874d5e0>,
      <matplotlib.lines.Line2D at 0x7f8fa874dac0>,
      <matplotlib.lines.Line2D at 0x7f8fa874dfa0>,
      <matplotlib.lines.Line2D at 0x7f8fa875a4c0>,
      <matplotlib.lines.Line2D at 0x7f8fa875a9a0>,
      <matplotlib.lines.Line2D at 0x7f8fa875ae80>,
      <matplotlib.lines.Line2D at 0x7f8fa87673a0>,
      <matplotlib.lines.Line2D at 0x7f8fa8767880>,
      <matplotlib.lines.Line2D at 0x7f8fa8767d60>,
      <matplotlib.lines.Line2D at 0x7f8fa8e1f280>,
      <matplotlib.lines.Line2D at 0x7f8fa8e1f760>,
      <matplotlib.lines.Line2D at 0x7f8fa8e1fc40>,
      <matplotlib.lines.Line2D at 0x7f8fa8e2c160>,
      <matplotlib.lines.Line2D at 0x7f8fa8e2c640>,
      <matplotlib.lines.Line2D at 0x7f8fa8e2cb20>,
      <matplotlib.lines.Line2D at 0x7f8fa8e2cfa0>,
      <matplotlib.lines.Line2D at 0x7f8fa8e37520>,
      <matplotlib.lines.Line2D at 0x7f8fa8e37a00>,
      <matplotlib.lines.Line2D at 0x7f8fa8e37ee0>,
      <matplotlib.lines.Line2D at 0x7f8fa8e45400>,
      <matplotlib.lines.Line2D at 0x7f8fa8e458e0>,
      <matplotlib.lines.Line2D at 0x7f8fa8e45dc0>,
      <matplotlib.lines.Line2D at 0x7f8f8a1ca2e0>,
      <matplotlib.lines.Line2D at 0x7f8f8a1ca7c0>,
      <matplotlib.lines.Line2D at 0x7f8f8a1caca0>,
      <matplotlib.lines.Line2D at 0x7f8f8a1d91c0>,
      <matplotlib.lines.Line2D at 0x7f8f8a1d96a0>,
      <matplotlib.lines.Line2D at 0x7f8f8a1d9b80>,
      <matplotlib.lines.Line2D at 0x7f8f8a1d9fd0>,
      <matplotlib.lines.Line2D at 0x7f8f8a1e8580>,
      <matplotlib.lines.Line2D at 0x7f8f8a1e8a60>,
      <matplotlib.lines.Line2D at 0x7f8f8a1e8f40>,
      <matplotlib.lines.Line2D at 0x7f8f8a1f4460>,
      <matplotlib.lines.Line2D at 0x7f8f8a1f4940>,
      <matplotlib.lines.Line2D at 0x7f8f8a1f4e20>,
      <matplotlib.lines.Line2D at 0x7f8f8a202340>,
      <matplotlib.lines.Line2D at 0x7f8f8a202820>,
      <matplotlib.lines.Line2D at 0x7f8f8a202d00>,
      <matplotlib.lines.Line2D at 0x7f8f8a20f220>,
      <matplotlib.lines.Line2D at 0x7f8f8a20f700>,
      <matplotlib.lines.Line2D at 0x7f8f8a20fbe0>,
      <matplotlib.lines.Line2D at 0x7f8f8a21d100>,
      <matplotlib.lines.Line2D at 0x7f8f8a21d5e0>,
      <matplotlib.lines.Line2D at 0x7f8f8a21dac0>,
      <matplotlib.lines.Line2D at 0x7f8f8a21dfa0>,
      <matplotlib.lines.Line2D at 0x7f8f8a22d4c0>,
      <matplotlib.lines.Line2D at 0x7f8f8a22d9a0>,
      <matplotlib.lines.Line2D at 0x7f8f8a22de80>,
      <matplotlib.lines.Line2D at 0x7f8f8a2393a0>,
      <matplotlib.lines.Line2D at 0x7f8f8a239880>,
      <matplotlib.lines.Line2D at 0x7f8f8a239d60>,
      <matplotlib.lines.Line2D at 0x7f8f8a246280>,
      <matplotlib.lines.Line2D at 0x7f8f8a246760>,
      <matplotlib.lines.Line2D at 0x7f8f8a246c40>,
      <matplotlib.lines.Line2D at 0x7f8f8a254160>,
      <matplotlib.lines.Line2D at 0x7f8f8a254640>,
      <matplotlib.lines.Line2D at 0x7f8f8a254b20>,
      <matplotlib.lines.Line2D at 0x7f8f8a254fa0>,
      <matplotlib.lines.Line2D at 0x7f8f8a262520>,
      <matplotlib.lines.Line2D at 0x7f8f8a262a00>,
      <matplotlib.lines.Line2D at 0x7f8f8a262ee0>,
      <matplotlib.lines.Line2D at 0x7f8f8a26f400>,
      <matplotlib.lines.Line2D at 0x7f8f8a26f8e0>,
      <matplotlib.lines.Line2D at 0x7f8f8a26fdc0>,
      <matplotlib.lines.Line2D at 0x7f8f8a27e2e0>,
      <matplotlib.lines.Line2D at 0x7f8f8a27e7c0>,
      <matplotlib.lines.Line2D at 0x7f8f8a27eca0>,
      <matplotlib.lines.Line2D at 0x7f8f8a28c1c0>,
      <matplotlib.lines.Line2D at 0x7f8f8a28c6a0>,
      <matplotlib.lines.Line2D at 0x7f8f8a28cb80>,
      <matplotlib.lines.Line2D at 0x7f8f8a28cfd0>,
      <matplotlib.lines.Line2D at 0x7f8f8a297580>,
      <matplotlib.lines.Line2D at 0x7f8f8a297a60>,
      <matplotlib.lines.Line2D at 0x7f8f8a297f40>,
      <matplotlib.lines.Line2D at 0x7f8f8a2a6460>,
      <matplotlib.lines.Line2D at 0x7f8f8a2a6940>,
      <matplotlib.lines.Line2D at 0x7f8f8a2a6e20>,
      <matplotlib.lines.Line2D at 0x7f8f8a2b3340>,
      <matplotlib.lines.Line2D at 0x7f8f8a2b3820>,
      <matplotlib.lines.Line2D at 0x7f8f8a2b3d00>,
      <matplotlib.lines.Line2D at 0x7f8f8a2c1220>,
      <matplotlib.lines.Line2D at 0x7f8f8a2c1700>,
      <matplotlib.lines.Line2D at 0x7f8f8a2c1be0>,
      <matplotlib.lines.Line2D at 0x7f8f8a2cf100>,
      <matplotlib.lines.Line2D at 0x7f8f8a2cf5e0>,
      <matplotlib.lines.Line2D at 0x7f8f8a2cfac0>,
      <matplotlib.lines.Line2D at 0x7f8f8a2cffa0>,
      <matplotlib.lines.Line2D at 0x7f8f8a2de4c0>,
      <matplotlib.lines.Line2D at 0x7f8f8a2de9a0>,
      <matplotlib.lines.Line2D at 0x7f8f8a2dee80>,
      <matplotlib.lines.Line2D at 0x7f8f8a2e93a0>,
      <matplotlib.lines.Line2D at 0x7f8f8a2e9880>,
      <matplotlib.lines.Line2D at 0x7f8f8a2e9d60>,
      <matplotlib.lines.Line2D at 0x7f8f8a2f7280>,
      <matplotlib.lines.Line2D at 0x7f8f8a2f7760>,
      <matplotlib.lines.Line2D at 0x7f8f8a2f7c40>,
      <matplotlib.lines.Line2D at 0x7f8f8a305160>,
      <matplotlib.lines.Line2D at 0x7f8f8a305640>,
      <matplotlib.lines.Line2D at 0x7f8f8a305b20>,
      <matplotlib.lines.Line2D at 0x7f8f8a305fa0>,
      <matplotlib.lines.Line2D at 0x7f8f8a314520>,
      <matplotlib.lines.Line2D at 0x7f8f8a314a00>,
      <matplotlib.lines.Line2D at 0x7f8f8a314ee0>,
      <matplotlib.lines.Line2D at 0x7f8f8a322400>,
      <matplotlib.lines.Line2D at 0x7f8f8a3228e0>,
      <matplotlib.lines.Line2D at 0x7f8f8a322dc0>,
      <matplotlib.lines.Line2D at 0x7f8f8a32c2e0>,
      <matplotlib.lines.Line2D at 0x7f8f8a32c7c0>,
      <matplotlib.lines.Line2D at 0x7f8f8a32cca0>,
      <matplotlib.lines.Line2D at 0x7f8f8a33c1c0>,
      <matplotlib.lines.Line2D at 0x7f8f8a33c6a0>,
      <matplotlib.lines.Line2D at 0x7f8f8a33cb80>,
      <matplotlib.lines.Line2D at 0x7f8f8a33cfd0>,
      <matplotlib.lines.Line2D at 0x7f8f8a34a580>,
      <matplotlib.lines.Line2D at 0x7f8f8a34aa60>,
      <matplotlib.lines.Line2D at 0x7f8f8a34af40>,
      <matplotlib.lines.Line2D at 0x7f8f8a357460>,
      <matplotlib.lines.Line2D at 0x7f8f8a357940>,
      <matplotlib.lines.Line2D at 0x7f8f8a357e20>,
      <matplotlib.lines.Line2D at 0x7f8f8a365340>,
      <matplotlib.lines.Line2D at 0x7f8f8a365820>,
      <matplotlib.lines.Line2D at 0x7f8f8a365d00>,
      <matplotlib.lines.Line2D at 0x7f8f8a371220>,
      <matplotlib.lines.Line2D at 0x7f8f8a371700>,
      <matplotlib.lines.Line2D at 0x7f8f8a371be0>,
      <matplotlib.lines.Line2D at 0x7f8f8a382100>,
      <matplotlib.lines.Line2D at 0x7f8f8a3825e0>,
      <matplotlib.lines.Line2D at 0x7f8f8a382ac0>,
      <matplotlib.lines.Line2D at 0x7f8f8a382fa0>,
      <matplotlib.lines.Line2D at 0x7f8f8a38e4c0>,
      <matplotlib.lines.Line2D at 0x7f8f8a38e9a0>,
      <matplotlib.lines.Line2D at 0x7f8f8a38ee80>,
      <matplotlib.lines.Line2D at 0x7f8f8a39b3a0>,
      <matplotlib.lines.Line2D at 0x7f8f8a39b880>,
      <matplotlib.lines.Line2D at 0x7f8f8a39bd60>,
      <matplotlib.lines.Line2D at 0x7f8f8a3aa280>],
     [<matplotlib.patches.Rectangle at 0x7f8fa89cd100>,
      <matplotlib.patches.Rectangle at 0x7f8fa89cd5e0>,
      <matplotlib.patches.Rectangle at 0x7f8fa89cdac0>,
      <matplotlib.patches.Rectangle at 0x7f8fa89cdfa0>,
      <matplotlib.patches.Rectangle at 0x7f8f6ad944c0>,
      <matplotlib.patches.Rectangle at 0x7f8f6ad949a0>,
      <matplotlib.patches.Rectangle at 0x7f8f6ad94e80>,
      <matplotlib.patches.Rectangle at 0x7f8f6adc73a0>,
      <matplotlib.patches.Rectangle at 0x7f8f6adc7880>,
      <matplotlib.patches.Rectangle at 0x7f8f6adc7d60>,
      <matplotlib.patches.Rectangle at 0x7f8f6ada4280>,
      <matplotlib.patches.Rectangle at 0x7f8f6ada4760>,
      <matplotlib.patches.Rectangle at 0x7f8f6ada4c40>,
      <matplotlib.patches.Rectangle at 0x7f8f6adb8040>,
      <matplotlib.patches.Rectangle at 0x7f8f6adb8640>,
      <matplotlib.patches.Rectangle at 0x7f8f6adb8b20>,
      <matplotlib.patches.Rectangle at 0x7f8f6adb8f40>,
      <matplotlib.patches.Rectangle at 0x7f8f6ad8d520>,
      <matplotlib.patches.Rectangle at 0x7f8f6ad8da00>,
      <matplotlib.patches.Rectangle at 0x7f8f6ad8dee0>,
      <matplotlib.patches.Rectangle at 0x7f8f6af7e400>,
      <matplotlib.patches.Rectangle at 0x7f8f6af7e8e0>,
      <matplotlib.patches.Rectangle at 0x7f8f6af7edc0>,
      <matplotlib.patches.Rectangle at 0x7f8f6af642e0>,
      <matplotlib.patches.Rectangle at 0x7f8f6af647c0>,
      <matplotlib.patches.Rectangle at 0x7f8f6af64ca0>,
      <matplotlib.patches.Rectangle at 0x7f8f6af671c0>,
      <matplotlib.patches.Rectangle at 0x7f8f6af676a0>,
      <matplotlib.patches.Rectangle at 0x7f8f6af67b80>,
      <matplotlib.patches.Rectangle at 0x7f8f6af67f40>,
      <matplotlib.patches.Rectangle at 0x7f8f6af8d580>,
      <matplotlib.patches.Rectangle at 0x7f8f6af8da60>,
      <matplotlib.patches.Rectangle at 0x7f8f6af8df40>,
      <matplotlib.patches.Rectangle at 0x7f8f89e62460>,
      <matplotlib.patches.Rectangle at 0x7f8f89e62940>,
      <matplotlib.patches.Rectangle at 0x7f8f89e62e20>,
      <matplotlib.patches.Rectangle at 0x7f8f89e73340>,
      <matplotlib.patches.Rectangle at 0x7f8f89e73820>,
      <matplotlib.patches.Rectangle at 0x7f8f89e73d00>,
      <matplotlib.patches.Rectangle at 0x7f8f89e64220>,
      <matplotlib.patches.Rectangle at 0x7f8f89e64700>,
      <matplotlib.patches.Rectangle at 0x7f8f89e64be0>,
      <matplotlib.patches.Rectangle at 0x7f8f89e64fa0>,
      <matplotlib.patches.Rectangle at 0x7f8f89e4f5e0>,
      <matplotlib.patches.Rectangle at 0x7f8f89e4fac0>,
      <matplotlib.patches.Rectangle at 0x7f8f89e4ffa0>,
      <matplotlib.patches.Rectangle at 0x7f8f89e504c0>,
      <matplotlib.patches.Rectangle at 0x7f8f89e509a0>,
      <matplotlib.patches.Rectangle at 0x7f8f89e50e80>,
      <matplotlib.patches.Rectangle at 0x7f8f6b0043a0>,
      <matplotlib.patches.Rectangle at 0x7f8f6b004880>,
      <matplotlib.patches.Rectangle at 0x7f8f6b004d60>,
      <matplotlib.patches.Rectangle at 0x7f8f6aff8280>,
      <matplotlib.patches.Rectangle at 0x7f8f6aff8760>,
      <matplotlib.patches.Rectangle at 0x7f8f6aff8c40>,
      <matplotlib.patches.Rectangle at 0x7f8f6afe8040>,
      <matplotlib.patches.Rectangle at 0x7f8f6afe8640>,
      <matplotlib.patches.Rectangle at 0x7f8f6afe8b20>,
      <matplotlib.patches.Rectangle at 0x7f8f6afe8f40>,
      <matplotlib.patches.Rectangle at 0x7f8f6afda520>,
      <matplotlib.patches.Rectangle at 0x7f8f6afdaa00>,
      <matplotlib.patches.Rectangle at 0x7f8f6afdaee0>,
      <matplotlib.patches.Rectangle at 0x7f8f9936e400>,
      <matplotlib.patches.Rectangle at 0x7f8f9936e8e0>,
      <matplotlib.patches.Rectangle at 0x7f8f9936edc0>,
      <matplotlib.patches.Rectangle at 0x7f8f993512e0>,
      <matplotlib.patches.Rectangle at 0x7f8f993517c0>,
      <matplotlib.patches.Rectangle at 0x7f8f99351ca0>,
      <matplotlib.patches.Rectangle at 0x7f8f993531c0>,
      <matplotlib.patches.Rectangle at 0x7f8f993536a0>,
      <matplotlib.patches.Rectangle at 0x7f8f99353b80>,
      <matplotlib.patches.Rectangle at 0x7f8f99353f40>,
      <matplotlib.patches.Rectangle at 0x7f8f9934c580>,
      <matplotlib.patches.Rectangle at 0x7f8f9934ca60>,
      <matplotlib.patches.Rectangle at 0x7f8f9934cf40>,
      <matplotlib.patches.Rectangle at 0x7f8f9934e460>,
      <matplotlib.patches.Rectangle at 0x7f8f9934e940>,
      <matplotlib.patches.Rectangle at 0x7f8f9934ee20>,
      <matplotlib.patches.Rectangle at 0x7f8fa8b1a340>,
      <matplotlib.patches.Rectangle at 0x7f8fa8b1a820>,
      <matplotlib.patches.Rectangle at 0x7f8fa8b1ad00>,
      <matplotlib.patches.Rectangle at 0x7f8fa8b2f220>,
      <matplotlib.patches.Rectangle at 0x7f8fa8b2f700>,
      <matplotlib.patches.Rectangle at 0x7f8fa8b2fbe0>,
      <matplotlib.patches.Rectangle at 0x7f8fa8b2ffa0>,
      <matplotlib.patches.Rectangle at 0x7f8fa8b2e5e0>,
      <matplotlib.patches.Rectangle at 0x7f8fa8b2eac0>,
      <matplotlib.patches.Rectangle at 0x7f8fa8b2efa0>,
      <matplotlib.patches.Rectangle at 0x7f8fa8b0a4c0>,
      <matplotlib.patches.Rectangle at 0x7f8fa8b0a9a0>,
      <matplotlib.patches.Rectangle at 0x7f8fa8b0ae80>,
      <matplotlib.patches.Rectangle at 0x7f8fa8b183a0>,
      <matplotlib.patches.Rectangle at 0x7f8fa8b18880>,
      <matplotlib.patches.Rectangle at 0x7f8fa8b18d60>,
      <matplotlib.patches.Rectangle at 0x7f8fa8ba3280>,
      <matplotlib.patches.Rectangle at 0x7f8fa8ba3760>,
      <matplotlib.patches.Rectangle at 0x7f8fa8ba3c40>,
      <matplotlib.patches.Rectangle at 0x7f8fa8b96040>,
      <matplotlib.patches.Rectangle at 0x7f8fa8b96640>,
      <matplotlib.patches.Rectangle at 0x7f8fa8b96b20>,
      <matplotlib.patches.Rectangle at 0x7f8fa8b96f40>,
      <matplotlib.patches.Rectangle at 0x7f8fa8b88520>,
      <matplotlib.patches.Rectangle at 0x7f8fa8b88a00>,
      <matplotlib.patches.Rectangle at 0x7f8fa8b88ee0>,
      <matplotlib.patches.Rectangle at 0x7f8fa8b7a400>,
      <matplotlib.patches.Rectangle at 0x7f8fa8b7a8e0>,
      <matplotlib.patches.Rectangle at 0x7f8fa8b7adc0>,
      <matplotlib.patches.Rectangle at 0x7f8f997692e0>,
      <matplotlib.patches.Rectangle at 0x7f8f997697c0>,
      <matplotlib.patches.Rectangle at 0x7f8f99769ca0>,
      <matplotlib.patches.Rectangle at 0x7f8f997770a0>,
      <matplotlib.patches.Rectangle at 0x7f8f997776a0>,
      <matplotlib.patches.Rectangle at 0x7f8f99777b80>,
      <matplotlib.patches.Rectangle at 0x7f8f99777f40>,
      <matplotlib.patches.Rectangle at 0x7f8f99784580>,
      <matplotlib.patches.Rectangle at 0x7f8f99784a60>,
      <matplotlib.patches.Rectangle at 0x7f8f99784f40>,
      <matplotlib.patches.Rectangle at 0x7f8f99791460>,
      <matplotlib.patches.Rectangle at 0x7f8f99791940>,
      <matplotlib.patches.Rectangle at 0x7f8f99791e20>,
      <matplotlib.patches.Rectangle at 0x7f8f997a1340>,
      <matplotlib.patches.Rectangle at 0x7f8f997a1820>,
      <matplotlib.patches.Rectangle at 0x7f8f997a1d00>,
      <matplotlib.patches.Rectangle at 0x7f8f997ae100>,
      <matplotlib.patches.Rectangle at 0x7f8f997ae700>,
      <matplotlib.patches.Rectangle at 0x7f8f997aebe0>,
      <matplotlib.patches.Rectangle at 0x7f8f997aefa0>,
      <matplotlib.patches.Rectangle at 0x7f8f997bd5e0>,
      <matplotlib.patches.Rectangle at 0x7f8f997bdac0>,
      <matplotlib.patches.Rectangle at 0x7f8f997bdfa0>,
      <matplotlib.patches.Rectangle at 0x7f8f997c94c0>,
      <matplotlib.patches.Rectangle at 0x7f8f997c99a0>,
      <matplotlib.patches.Rectangle at 0x7f8f997c9e80>,
      <matplotlib.patches.Rectangle at 0x7f8f997d53a0>,
      <matplotlib.patches.Rectangle at 0x7f8f997d5880>,
      <matplotlib.patches.Rectangle at 0x7f8f997d5d60>,
      <matplotlib.patches.Rectangle at 0x7f8f997e5280>,
      <matplotlib.patches.Rectangle at 0x7f8f997e5760>,
      <matplotlib.patches.Rectangle at 0x7f8f997e5c40>,
      <matplotlib.patches.Rectangle at 0x7f8f997f4040>,
      <matplotlib.patches.Rectangle at 0x7f8f997f4640>,
      <matplotlib.patches.Rectangle at 0x7f8f997f4b20>,
      <matplotlib.patches.Rectangle at 0x7f8f997f4f40>,
      <matplotlib.patches.Rectangle at 0x7f8f997fe520>,
      <matplotlib.patches.Rectangle at 0x7f8f997fea00>,
      <matplotlib.patches.Rectangle at 0x7f8f997feee0>,
      <matplotlib.patches.Rectangle at 0x7f8f9980e400>,
      <matplotlib.patches.Rectangle at 0x7f8f9980e8e0>,
      <matplotlib.patches.Rectangle at 0x7f8f9980edc0>,
      <matplotlib.patches.Rectangle at 0x7f8f9981c2e0>,
      <matplotlib.patches.Rectangle at 0x7f8f9981c7c0>,
      <matplotlib.patches.Rectangle at 0x7f8f9981cca0>,
      <matplotlib.patches.Rectangle at 0x7f8f99f211c0>,
      <matplotlib.patches.Rectangle at 0x7f8f99f216a0>,
      <matplotlib.patches.Rectangle at 0x7f8f99f21b80>,
      <matplotlib.patches.Rectangle at 0x7f8f99f21f40>,
      <matplotlib.patches.Rectangle at 0x7f8f99f2e580>,
      <matplotlib.patches.Rectangle at 0x7f8f99f2ea60>,
      <matplotlib.patches.Rectangle at 0x7f8f99f2ef40>,
      <matplotlib.patches.Rectangle at 0x7f8f99f3c460>,
      <matplotlib.patches.Rectangle at 0x7f8f99f3c940>,
      <matplotlib.patches.Rectangle at 0x7f8f99f3ce20>,
      <matplotlib.patches.Rectangle at 0x7f8f99f49340>,
      <matplotlib.patches.Rectangle at 0x7f8f99f49820>,
      <matplotlib.patches.Rectangle at 0x7f8f99f49d00>,
      <matplotlib.patches.Rectangle at 0x7f8f99f58220>,
      <matplotlib.patches.Rectangle at 0x7f8f99f58700>,
      <matplotlib.patches.Rectangle at 0x7f8f99f58be0>,
      <matplotlib.patches.Rectangle at 0x7f8f99f58fa0>,
      <matplotlib.patches.Rectangle at 0x7f8f99f645e0>,
      <matplotlib.patches.Rectangle at 0x7f8f99f64ac0>,
      <matplotlib.patches.Rectangle at 0x7f8f99f64fa0>,
      <matplotlib.patches.Rectangle at 0x7f8f99f734c0>,
      <matplotlib.patches.Rectangle at 0x7f8f99f739a0>,
      <matplotlib.patches.Rectangle at 0x7f8f99f73e80>,
      <matplotlib.patches.Rectangle at 0x7f8f99f7f3a0>,
      <matplotlib.patches.Rectangle at 0x7f8f99f7f880>,
      <matplotlib.patches.Rectangle at 0x7f8f99f7fd60>,
      <matplotlib.patches.Rectangle at 0x7f8f99f8e280>,
      <matplotlib.patches.Rectangle at 0x7f8f99f8e760>,
      <matplotlib.patches.Rectangle at 0x7f8f99f8ec40>,
      <matplotlib.patches.Rectangle at 0x7f8f99f9d040>,
      <matplotlib.patches.Rectangle at 0x7f8f99f9d640>,
      <matplotlib.patches.Rectangle at 0x7f8f99f9db20>,
      <matplotlib.patches.Rectangle at 0x7f8f99f9df40>,
      <matplotlib.patches.Rectangle at 0x7f8f99fa8520>,
      <matplotlib.patches.Rectangle at 0x7f8f99fa8a00>,
      <matplotlib.patches.Rectangle at 0x7f8f99fa8ee0>,
      <matplotlib.patches.Rectangle at 0x7f8f99fb6400>,
      <matplotlib.patches.Rectangle at 0x7f8f99fb68e0>,
      <matplotlib.patches.Rectangle at 0x7f8f99fb6dc0>,
      <matplotlib.patches.Rectangle at 0x7f8f99fc52e0>,
      <matplotlib.patches.Rectangle at 0x7f8f99fc57c0>,
      <matplotlib.patches.Rectangle at 0x7f8f99fc5ca0>,
      <matplotlib.patches.Rectangle at 0x7f8f99fd31c0>,
      <matplotlib.patches.Rectangle at 0x7f8f99fd36a0>,
      <matplotlib.patches.Rectangle at 0x7f8f99fd3b80>,
      <matplotlib.patches.Rectangle at 0x7f8f99fd3f40>,
      <matplotlib.patches.Rectangle at 0x7f8f99fde580>,
      <matplotlib.patches.Rectangle at 0x7f8f99fdea60>,
      <matplotlib.patches.Rectangle at 0x7f8f99fdef40>,
      <matplotlib.patches.Rectangle at 0x7f8f99fed460>,
      <matplotlib.patches.Rectangle at 0x7f8f99fed940>,
      <matplotlib.patches.Rectangle at 0x7f8f99fede20>,
      <matplotlib.patches.Rectangle at 0x7f8f99ffb340>,
      <matplotlib.patches.Rectangle at 0x7f8f99ffb820>,
      <matplotlib.patches.Rectangle at 0x7f8f99ffbd00>,
      <matplotlib.patches.Rectangle at 0x7f8f9a008220>,
      <matplotlib.patches.Rectangle at 0x7f8f9a008700>,
      <matplotlib.patches.Rectangle at 0x7f8f9a008be0>,
      <matplotlib.patches.Rectangle at 0x7f8f9a008fa0>,
      <matplotlib.patches.Rectangle at 0x7f8f9a0165e0>,
      <matplotlib.patches.Rectangle at 0x7f8f9a016ac0>,
      <matplotlib.patches.Rectangle at 0x7f8f9a016fa0>,
      <matplotlib.patches.Rectangle at 0x7f8f9a0234c0>,
      <matplotlib.patches.Rectangle at 0x7f8f9a0239a0>,
      <matplotlib.patches.Rectangle at 0x7f8f9a023e80>,
      <matplotlib.patches.Rectangle at 0x7f8f9a0333a0>,
      <matplotlib.patches.Rectangle at 0x7f8f9a033880>,
      <matplotlib.patches.Rectangle at 0x7f8f9a033d60>,
      <matplotlib.patches.Rectangle at 0x7f8f9a040280>,
      <matplotlib.patches.Rectangle at 0x7f8f9a040760>,
      <matplotlib.patches.Rectangle at 0x7f8f9a040c40>,
      <matplotlib.patches.Rectangle at 0x7f8f9a04c040>,
      <matplotlib.patches.Rectangle at 0x7f8f9a04c640>,
      <matplotlib.patches.Rectangle at 0x7f8f9a04cb20>,
      <matplotlib.patches.Rectangle at 0x7f8f9a04cf40>,
      <matplotlib.patches.Rectangle at 0x7f8f9a05a520>,
      <matplotlib.patches.Rectangle at 0x7f8f9a05aa00>,
      <matplotlib.patches.Rectangle at 0x7f8f9a05aee0>,
      <matplotlib.patches.Rectangle at 0x7f8f9a069400>,
      <matplotlib.patches.Rectangle at 0x7f8f9a0698e0>,
      <matplotlib.patches.Rectangle at 0x7f8f9a069dc0>,
      <matplotlib.patches.Rectangle at 0x7f8f9a0762e0>,
      <matplotlib.patches.Rectangle at 0x7f8f9a0767c0>,
      <matplotlib.patches.Rectangle at 0x7f8f9a076ca0>,
      <matplotlib.patches.Rectangle at 0x7f8f9a0841c0>,
      <matplotlib.patches.Rectangle at 0x7f8f9a0846a0>,
      <matplotlib.patches.Rectangle at 0x7f8f9a084b80>,
      <matplotlib.patches.Rectangle at 0x7f8f9a084f40>,
      <matplotlib.patches.Rectangle at 0x7f8f9a08f580>,
      <matplotlib.patches.Rectangle at 0x7f8f9a08fa60>,
      <matplotlib.patches.Rectangle at 0x7f8f9a08ff40>,
      <matplotlib.patches.Rectangle at 0x7f8f9a09d460>,
      <matplotlib.patches.Rectangle at 0x7f8f9a09d940>,
      <matplotlib.patches.Rectangle at 0x7f8f9a09de20>,
      <matplotlib.patches.Rectangle at 0x7f8f9a0ad340>,
      <matplotlib.patches.Rectangle at 0x7f8f9a0ad820>,
      <matplotlib.patches.Rectangle at 0x7f8f9a0add00>,
      <matplotlib.patches.Rectangle at 0x7f8f9a0ba220>,
      <matplotlib.patches.Rectangle at 0x7f8f9a0ba700>,
      <matplotlib.patches.Rectangle at 0x7f8f9a0babe0>,
      <matplotlib.patches.Rectangle at 0x7f8f9a0bafa0>,
      <matplotlib.patches.Rectangle at 0x7f8f9a0c95e0>,
      <matplotlib.patches.Rectangle at 0x7f8f9a0c9ac0>,
      <matplotlib.patches.Rectangle at 0x7f8f9a0c9fa0>,
      <matplotlib.patches.Rectangle at 0x7f8f9a0d54c0>,
      <matplotlib.patches.Rectangle at 0x7f8f9a0d59a0>,
      <matplotlib.patches.Rectangle at 0x7f8f9a0d5e80>,
      <matplotlib.patches.Rectangle at 0x7f8f9a0e13a0>,
      <matplotlib.patches.Rectangle at 0x7f8f9a0e1880>,
      <matplotlib.patches.Rectangle at 0x7f8f9a0e1d60>,
      <matplotlib.patches.Rectangle at 0x7f8f9a0f1280>,
      <matplotlib.patches.Rectangle at 0x7f8f9a0f1760>,
      <matplotlib.patches.Rectangle at 0x7f8f9a0f1c40>,
      <matplotlib.patches.Rectangle at 0x7f8f9a0ff040>,
      <matplotlib.patches.Rectangle at 0x7f8f9a0ff640>,
      <matplotlib.patches.Rectangle at 0x7f8f9a0ffb20>,
      <matplotlib.patches.Rectangle at 0x7f8f9a0fff40>,
      <matplotlib.patches.Rectangle at 0x7f8f9a10c520>,
      <matplotlib.patches.Rectangle at 0x7f8f9a10ca00>,
      <matplotlib.patches.Rectangle at 0x7f8f9a10cee0>,
      <matplotlib.patches.Rectangle at 0x7f8f9a11b400>,
      <matplotlib.patches.Rectangle at 0x7f8f9a11b8e0>,
      <matplotlib.patches.Rectangle at 0x7f8f9a11bdc0>,
      <matplotlib.patches.Rectangle at 0x7f8f9a1282e0>,
      <matplotlib.patches.Rectangle at 0x7f8f9a1287c0>,
      <matplotlib.patches.Rectangle at 0x7f8f9a128ca0>,
      <matplotlib.patches.Rectangle at 0x7f8f9a1331c0>,
      <matplotlib.patches.Rectangle at 0x7f8f9a1336a0>,
      <matplotlib.patches.Rectangle at 0x7f8f9a133b80>,
      <matplotlib.patches.Rectangle at 0x7f8f9a133f40>,
      <matplotlib.patches.Rectangle at 0x7f8f9a142580>,
      <matplotlib.patches.Rectangle at 0x7f8f9a142a60>,
      <matplotlib.patches.Rectangle at 0x7f8f9a142f40>,
      <matplotlib.patches.Rectangle at 0x7f8f9a150460>,
      <matplotlib.patches.Rectangle at 0x7f8f9a150940>,
      <matplotlib.patches.Rectangle at 0x7f8f9a150e20>,
      <matplotlib.patches.Rectangle at 0x7f8f9a15d340>,
      <matplotlib.patches.Rectangle at 0x7f8f9a15d820>,
      <matplotlib.patches.Rectangle at 0x7f8f9a15dd00>,
      <matplotlib.patches.Rectangle at 0x7f8f9a16c220>,
      <matplotlib.patches.Rectangle at 0x7f8f9a16c700>,
      <matplotlib.patches.Rectangle at 0x7f8f9a16cbe0>,
      <matplotlib.patches.Rectangle at 0x7f8f9a16cfa0>,
      <matplotlib.patches.Rectangle at 0x7f8f9a1785e0>,
      <matplotlib.patches.Rectangle at 0x7f8f9a178ac0>,
      <matplotlib.patches.Rectangle at 0x7f8f9a178fa0>,
      <matplotlib.patches.Rectangle at 0x7f8f9a1844c0>,
      <matplotlib.patches.Rectangle at 0x7f8f9a1849a0>,
      <matplotlib.patches.Rectangle at 0x7f8f9a184e80>,
      <matplotlib.patches.Rectangle at 0x7f8f9a1933a0>,
      <matplotlib.patches.Rectangle at 0x7f8f9a193880>,
      <matplotlib.patches.Rectangle at 0x7f8f9a193d60>,
      <matplotlib.patches.Rectangle at 0x7f8f9a1a1280>,
      <matplotlib.patches.Rectangle at 0x7f8f9a1a1760>,
      <matplotlib.patches.Rectangle at 0x7f8f9a1a1c40>,
      <matplotlib.patches.Rectangle at 0x7f8f9a1b2040>,
      <matplotlib.patches.Rectangle at 0x7f8f9a1b2640>,
      <matplotlib.patches.Rectangle at 0x7f8f9a1b2b20>,
      <matplotlib.patches.Rectangle at 0x7f8f9a1b2f40>,
      <matplotlib.patches.Rectangle at 0x7f8f9a1bc520>,
      <matplotlib.patches.Rectangle at 0x7f8f9a1bca00>,
      <matplotlib.patches.Rectangle at 0x7f8f9a1bcee0>,
      <matplotlib.patches.Rectangle at 0x7f8f9a1ca400>,
      <matplotlib.patches.Rectangle at 0x7f8f9a1ca8e0>,
      <matplotlib.patches.Rectangle at 0x7f8f9a1cadc0>,
      <matplotlib.patches.Rectangle at 0x7f8f9a1d82e0>,
      <matplotlib.patches.Rectangle at 0x7f8f9a1d87c0>,
      <matplotlib.patches.Rectangle at 0x7f8f9a1d8ca0>,
      <matplotlib.patches.Rectangle at 0x7f8f9a1e71c0>,
      <matplotlib.patches.Rectangle at 0x7f8f9a1e76a0>,
      <matplotlib.patches.Rectangle at 0x7f8f9a1e7b80>,
      <matplotlib.patches.Rectangle at 0x7f8f9a1e7f40>,
      <matplotlib.patches.Rectangle at 0x7f8f9a1f3580>,
      <matplotlib.patches.Rectangle at 0x7f8f9a1f3a60>,
      <matplotlib.patches.Rectangle at 0x7f8f9a1f3f40>,
      <matplotlib.patches.Rectangle at 0x7f8f9a201460>,
      <matplotlib.patches.Rectangle at 0x7f8f9a201940>,
      <matplotlib.patches.Rectangle at 0x7f8f9a201e20>,
      <matplotlib.patches.Rectangle at 0x7f8f9a20f340>,
      <matplotlib.patches.Rectangle at 0x7f8f9a20f820>,
      <matplotlib.patches.Rectangle at 0x7f8f9a20fd00>,
      <matplotlib.patches.Rectangle at 0x7f8f9a21c220>,
      <matplotlib.patches.Rectangle at 0x7f8f9a21c700>,
      <matplotlib.patches.Rectangle at 0x7f8f9a21cbe0>,
      <matplotlib.patches.Rectangle at 0x7f8f9a21cfa0>,
      <matplotlib.patches.Rectangle at 0x7f8f9a22a5e0>,
      <matplotlib.patches.Rectangle at 0x7f8f9a22aac0>,
      <matplotlib.patches.Rectangle at 0x7f8f9a22afa0>,
      <matplotlib.patches.Rectangle at 0x7f8f9a2374c0>,
      <matplotlib.patches.Rectangle at 0x7f8f9a2379a0>,
      <matplotlib.patches.Rectangle at 0x7f8f9a237e80>,
      <matplotlib.patches.Rectangle at 0x7f8f9a2463a0>,
      <matplotlib.patches.Rectangle at 0x7f8f9a246880>,
      <matplotlib.patches.Rectangle at 0x7f8f9a246d60>,
      <matplotlib.patches.Rectangle at 0x7f8f9a253280>,
      <matplotlib.patches.Rectangle at 0x7f8f9a253760>,
      <matplotlib.patches.Rectangle at 0x7f8f9a253c40>,
      <matplotlib.patches.Rectangle at 0x7f8fa88e0040>,
      <matplotlib.patches.Rectangle at 0x7f8fa88e0640>,
      <matplotlib.patches.Rectangle at 0x7f8fa88e0b20>,
      <matplotlib.patches.Rectangle at 0x7f8fa88e0f40>,
      <matplotlib.patches.Rectangle at 0x7f8fa88ee520>,
      <matplotlib.patches.Rectangle at 0x7f8fa88eea00>,
      <matplotlib.patches.Rectangle at 0x7f8fa88eeee0>,
      <matplotlib.patches.Rectangle at 0x7f8fa88fb400>,
      <matplotlib.patches.Rectangle at 0x7f8fa88fb8e0>,
      <matplotlib.patches.Rectangle at 0x7f8fa88fbdc0>,
      <matplotlib.patches.Rectangle at 0x7f8fa890a2e0>,
      <matplotlib.patches.Rectangle at 0x7f8fa890a7c0>,
      <matplotlib.patches.Rectangle at 0x7f8fa890aca0>,
      <matplotlib.patches.Rectangle at 0x7f8fa89181c0>,
      <matplotlib.patches.Rectangle at 0x7f8fa89186a0>,
      <matplotlib.patches.Rectangle at 0x7f8fa8918b80>,
      <matplotlib.patches.Rectangle at 0x7f8fa8918f40>,
      <matplotlib.patches.Rectangle at 0x7f8fa8d8c580>,
      <matplotlib.patches.Rectangle at 0x7f8fa8d8ca60>,
      <matplotlib.patches.Rectangle at 0x7f8fa8d8cf40>,
      <matplotlib.patches.Rectangle at 0x7f8fa8d98460>,
      <matplotlib.patches.Rectangle at 0x7f8fa8d98940>,
      <matplotlib.patches.Rectangle at 0x7f8fa8d98e20>,
      <matplotlib.patches.Rectangle at 0x7f8fa8da9340>,
      <matplotlib.patches.Rectangle at 0x7f8fa8da9820>,
      <matplotlib.patches.Rectangle at 0x7f8fa8da9d00>,
      <matplotlib.patches.Rectangle at 0x7f8fa8db6220>,
      <matplotlib.patches.Rectangle at 0x7f8fa8db6700>,
      <matplotlib.patches.Rectangle at 0x7f8fa8db6be0>,
      <matplotlib.patches.Rectangle at 0x7f8fa8db6fa0>,
      <matplotlib.patches.Rectangle at 0x7f8fa8dc55e0>,
      <matplotlib.patches.Rectangle at 0x7f8fa8dc5ac0>,
      <matplotlib.patches.Rectangle at 0x7f8fa8dc5fa0>,
      <matplotlib.patches.Rectangle at 0x7f8fa87304c0>,
      <matplotlib.patches.Rectangle at 0x7f8fa87309a0>,
      <matplotlib.patches.Rectangle at 0x7f8fa8730e80>,
      <matplotlib.patches.Rectangle at 0x7f8fa873d3a0>,
      <matplotlib.patches.Rectangle at 0x7f8fa873d880>,
      <matplotlib.patches.Rectangle at 0x7f8fa873dd60>,
      <matplotlib.patches.Rectangle at 0x7f8fa874d280>,
      <matplotlib.patches.Rectangle at 0x7f8fa874d760>,
      <matplotlib.patches.Rectangle at 0x7f8fa874dc40>,
      <matplotlib.patches.Rectangle at 0x7f8fa875a040>,
      <matplotlib.patches.Rectangle at 0x7f8fa875a640>,
      <matplotlib.patches.Rectangle at 0x7f8fa875ab20>,
      <matplotlib.patches.Rectangle at 0x7f8fa875af40>,
      <matplotlib.patches.Rectangle at 0x7f8fa8767520>,
      <matplotlib.patches.Rectangle at 0x7f8fa8767a00>,
      <matplotlib.patches.Rectangle at 0x7f8fa8767ee0>,
      <matplotlib.patches.Rectangle at 0x7f8fa8e1f400>,
      <matplotlib.patches.Rectangle at 0x7f8fa8e1f8e0>,
      <matplotlib.patches.Rectangle at 0x7f8fa8e1fdc0>,
      <matplotlib.patches.Rectangle at 0x7f8fa8e2c2e0>,
      <matplotlib.patches.Rectangle at 0x7f8fa8e2c7c0>,
      <matplotlib.patches.Rectangle at 0x7f8fa8e2cca0>,
      <matplotlib.patches.Rectangle at 0x7f8fa8e371c0>,
      <matplotlib.patches.Rectangle at 0x7f8fa8e376a0>,
      <matplotlib.patches.Rectangle at 0x7f8fa8e37b80>,
      <matplotlib.patches.Rectangle at 0x7f8fa8e37f40>,
      <matplotlib.patches.Rectangle at 0x7f8fa8e45580>,
      <matplotlib.patches.Rectangle at 0x7f8fa8e45a60>,
      <matplotlib.patches.Rectangle at 0x7f8fa8e45f40>,
      <matplotlib.patches.Rectangle at 0x7f8f8a1ca460>,
      <matplotlib.patches.Rectangle at 0x7f8f8a1ca940>,
      <matplotlib.patches.Rectangle at 0x7f8f8a1cae20>,
      <matplotlib.patches.Rectangle at 0x7f8f8a1d9340>,
      <matplotlib.patches.Rectangle at 0x7f8f8a1d9820>,
      <matplotlib.patches.Rectangle at 0x7f8f8a1d9d00>,
      <matplotlib.patches.Rectangle at 0x7f8f8a1e8220>,
      <matplotlib.patches.Rectangle at 0x7f8f8a1e8700>,
      <matplotlib.patches.Rectangle at 0x7f8f8a1e8be0>,
      <matplotlib.patches.Rectangle at 0x7f8f8a1e8fa0>,
      <matplotlib.patches.Rectangle at 0x7f8f8a1f45e0>,
      <matplotlib.patches.Rectangle at 0x7f8f8a1f4ac0>,
      <matplotlib.patches.Rectangle at 0x7f8f8a1f4fa0>,
      <matplotlib.patches.Rectangle at 0x7f8f8a2024c0>,
      <matplotlib.patches.Rectangle at 0x7f8f8a2029a0>,
      <matplotlib.patches.Rectangle at 0x7f8f8a202e80>,
      <matplotlib.patches.Rectangle at 0x7f8f8a20f3a0>,
      <matplotlib.patches.Rectangle at 0x7f8f8a20f880>,
      <matplotlib.patches.Rectangle at 0x7f8f8a20fd60>,
      <matplotlib.patches.Rectangle at 0x7f8f8a21d280>,
      <matplotlib.patches.Rectangle at 0x7f8f8a21d760>,
      <matplotlib.patches.Rectangle at 0x7f8f8a21dc40>,
      <matplotlib.patches.Rectangle at 0x7f8f8a22d040>,
      <matplotlib.patches.Rectangle at 0x7f8f8a22d640>,
      <matplotlib.patches.Rectangle at 0x7f8f8a22db20>,
      <matplotlib.patches.Rectangle at 0x7f8f8a22df40>,
      <matplotlib.patches.Rectangle at 0x7f8f8a239520>,
      <matplotlib.patches.Rectangle at 0x7f8f8a239a00>,
      <matplotlib.patches.Rectangle at 0x7f8f8a239ee0>,
      <matplotlib.patches.Rectangle at 0x7f8f8a246400>,
      <matplotlib.patches.Rectangle at 0x7f8f8a2468e0>,
      <matplotlib.patches.Rectangle at 0x7f8f8a246dc0>,
      <matplotlib.patches.Rectangle at 0x7f8f8a2542e0>,
      <matplotlib.patches.Rectangle at 0x7f8f8a2547c0>,
      <matplotlib.patches.Rectangle at 0x7f8f8a254ca0>,
      <matplotlib.patches.Rectangle at 0x7f8f8a2621c0>,
      <matplotlib.patches.Rectangle at 0x7f8f8a2626a0>,
      <matplotlib.patches.Rectangle at 0x7f8f8a262b80>,
      <matplotlib.patches.Rectangle at 0x7f8f8a262f40>,
      <matplotlib.patches.Rectangle at 0x7f8f8a26f580>,
      <matplotlib.patches.Rectangle at 0x7f8f8a26fa60>,
      <matplotlib.patches.Rectangle at 0x7f8f8a26ff40>,
      <matplotlib.patches.Rectangle at 0x7f8f8a27e460>,
      <matplotlib.patches.Rectangle at 0x7f8f8a27e940>,
      <matplotlib.patches.Rectangle at 0x7f8f8a27ee20>,
      <matplotlib.patches.Rectangle at 0x7f8f8a28c340>,
      <matplotlib.patches.Rectangle at 0x7f8f8a28c820>,
      <matplotlib.patches.Rectangle at 0x7f8f8a28cd00>,
      <matplotlib.patches.Rectangle at 0x7f8f8a297220>,
      <matplotlib.patches.Rectangle at 0x7f8f8a297700>,
      <matplotlib.patches.Rectangle at 0x7f8f8a297be0>,
      <matplotlib.patches.Rectangle at 0x7f8f8a297fa0>,
      <matplotlib.patches.Rectangle at 0x7f8f8a2a65e0>,
      <matplotlib.patches.Rectangle at 0x7f8f8a2a6ac0>,
      <matplotlib.patches.Rectangle at 0x7f8f8a2a6fa0>,
      <matplotlib.patches.Rectangle at 0x7f8f8a2b34c0>,
      <matplotlib.patches.Rectangle at 0x7f8f8a2b39a0>,
      <matplotlib.patches.Rectangle at 0x7f8f8a2b3e80>,
      <matplotlib.patches.Rectangle at 0x7f8f8a2c13a0>,
      <matplotlib.patches.Rectangle at 0x7f8f8a2c1880>,
      <matplotlib.patches.Rectangle at 0x7f8f8a2c1d60>,
      <matplotlib.patches.Rectangle at 0x7f8f8a2cf280>,
      <matplotlib.patches.Rectangle at 0x7f8f8a2cf760>,
      <matplotlib.patches.Rectangle at 0x7f8f8a2cfc40>,
      <matplotlib.patches.Rectangle at 0x7f8f8a2de040>,
      <matplotlib.patches.Rectangle at 0x7f8f8a2de640>,
      <matplotlib.patches.Rectangle at 0x7f8f8a2deb20>,
      <matplotlib.patches.Rectangle at 0x7f8f8a2def40>,
      <matplotlib.patches.Rectangle at 0x7f8f8a2e9520>,
      <matplotlib.patches.Rectangle at 0x7f8f8a2e9a00>,
      <matplotlib.patches.Rectangle at 0x7f8f8a2e9ee0>,
      <matplotlib.patches.Rectangle at 0x7f8f8a2f7400>,
      <matplotlib.patches.Rectangle at 0x7f8f8a2f78e0>,
      <matplotlib.patches.Rectangle at 0x7f8f8a2f7dc0>,
      <matplotlib.patches.Rectangle at 0x7f8f8a3052e0>,
      <matplotlib.patches.Rectangle at 0x7f8f8a3057c0>,
      <matplotlib.patches.Rectangle at 0x7f8f8a305ca0>,
      <matplotlib.patches.Rectangle at 0x7f8f8a3141c0>,
      <matplotlib.patches.Rectangle at 0x7f8f8a3146a0>,
      <matplotlib.patches.Rectangle at 0x7f8f8a314b80>,
      <matplotlib.patches.Rectangle at 0x7f8f8a314f40>,
      <matplotlib.patches.Rectangle at 0x7f8f8a322580>,
      <matplotlib.patches.Rectangle at 0x7f8f8a322a60>,
      <matplotlib.patches.Rectangle at 0x7f8f8a322f40>,
      <matplotlib.patches.Rectangle at 0x7f8f8a32c460>,
      <matplotlib.patches.Rectangle at 0x7f8f8a32c940>,
      <matplotlib.patches.Rectangle at 0x7f8f8a32ce20>,
      <matplotlib.patches.Rectangle at 0x7f8f8a33c340>,
      <matplotlib.patches.Rectangle at 0x7f8f8a33c820>,
      <matplotlib.patches.Rectangle at 0x7f8f8a33cd00>,
      <matplotlib.patches.Rectangle at 0x7f8f8a34a220>,
      <matplotlib.patches.Rectangle at 0x7f8f8a34a700>,
      <matplotlib.patches.Rectangle at 0x7f8f8a34abe0>,
      <matplotlib.patches.Rectangle at 0x7f8f8a34afa0>,
      <matplotlib.patches.Rectangle at 0x7f8f8a3575e0>,
      <matplotlib.patches.Rectangle at 0x7f8f8a357ac0>,
      <matplotlib.patches.Rectangle at 0x7f8f8a357fa0>,
      <matplotlib.patches.Rectangle at 0x7f8f8a3654c0>,
      <matplotlib.patches.Rectangle at 0x7f8f8a3659a0>,
      <matplotlib.patches.Rectangle at 0x7f8f8a365e80>,
      <matplotlib.patches.Rectangle at 0x7f8f8a3713a0>,
      <matplotlib.patches.Rectangle at 0x7f8f8a371880>,
      <matplotlib.patches.Rectangle at 0x7f8f8a371d60>,
      <matplotlib.patches.Rectangle at 0x7f8f8a382280>,
      <matplotlib.patches.Rectangle at 0x7f8f8a382760>,
      <matplotlib.patches.Rectangle at 0x7f8f8a382c40>,
      <matplotlib.patches.Rectangle at 0x7f8f8a38e040>,
      <matplotlib.patches.Rectangle at 0x7f8f8a38e640>,
      <matplotlib.patches.Rectangle at 0x7f8f8a38eb20>,
      <matplotlib.patches.Rectangle at 0x7f8f8a38ef40>,
      <matplotlib.patches.Rectangle at 0x7f8f8a39b520>,
      <matplotlib.patches.Rectangle at 0x7f8f8a39ba00>,
      <matplotlib.patches.Rectangle at 0x7f8f8a39bee0>,
      <matplotlib.patches.Rectangle at 0x7f8f8a3aa400>])




    
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
plt.title('Tesla Returns')

```




    Text(0.5, 1.0, 'Tesla Returns')




    
![png](output_23_1.png)
    



```python
#Return hist
Amazon['returns'].hist(bins=50)
plt.title('Amazon Returns')

```




    Text(0.5, 1.0, 'Amazon Returns')




    
![png](output_24_1.png)
    



```python
#Overlaying returns
Amazon['returns'].hist(bins=50, alpha = .2, figsize = (13,6), label = 'Bitcoin')
Tesla['returns'].hist(bins=50, alpha = .2, label = 'Tesla')
plt.title('Overlay Returns')
plt.legend()
```




    <matplotlib.legend.Legend at 0x7f8f9a2de1f0>




    
![png](output_25_1.png)
    



```python
#Amazon day over day return
Amazon['returns'].plot( label = 'Bitcoin')
plt.title('Amazon Returns')

```




    Text(0.5, 1.0, 'Amazon Returns')




    
![png](output_26_1.png)
    



```python
#Day over day return 
Amazon['returns'].plot( label = 'Bitcoin',alpha = .5)
Tesla['returns'].plot(label = 'Tesla',alpha = .5)
plt.title('Amazon Returns')
plt.legend()

```




    <matplotlib.legend.Legend at 0x7f8f9a322be0>




    
![png](output_27_1.png)
    



```python
#Centeral tendency
box_df = pd.concat([Tesla['returns'], Amazon['returns']], axis = 1)
box_df.columns=['Tesla Returns', 'Amazon Returns']
box_df.plot(kind= "box", figsize = (16,6))
plt.title('Returns')

```




    Text(0.5, 1.0, 'Returns')




    
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
plt.title('Cumulative Returns')
plt.legend()
```




    <matplotlib.legend.Legend at 0x7f8f9964c340>




    
![png](output_31_1.png)
    



```python

```
