
def RSI(df,t,W):
          if t > W:
               
               prices = df['Adj Close'].iloc[t-W-1:t-1]
               delta = prices.diff()
               gain = delta.where(delta > 0, 0)
               loss = -delta.where(delta < 0, 0)
               avg_gain = gain.mean()
               avg_loss = loss.mean()
               if avg_loss == 0:
                  return 100.0
               else:
                  RS = avg_gain / avg_loss
                  RSI = 100.0 - (100.0)/(1 + RS)
                  return RSI
          else :
              return 50.0
          
def SMA(df,t,W):
        if t>=W :
             SMA = df['Adj Close'].iloc[t-W:t].mean()
             return SMA
        else :
            SMA = df['Adj Close'].iloc[:t].mean()
            return SMA

def EMA(df,t,W):
    global ema
    ema_alpha = 2/(1 + W)
    if t == 1:
        ema = df.iloc[0]["Adj Close"]
        return ema
    elif t <= W:
        ema = SMA(df,t,W)
        return ema
    else:
        current_close = df.iloc[t-1]["Adj Close"]
        ema = (ema_alpha * current_close) + ((1 - ema_alpha) * ema)
        return ema
    
def stochastic_oscillator(df,t,W):
    if t <= W :
         return 50.0
    
    high = df.iloc[t-W-1:t-1]['High'].max()
    low = df.iloc[t-W-1:t-1]['Low'].min()
    close = df.iloc[t-1]['Adj Close']
    if high - low == 0.0 :
         return 0.0
    K = 100 * (close - low) / (high - low)

    return K


def MACD(df,t):
   # Calculate 12-day EMA
    ema_12 = df['Adj Close'].ewm(span=12, adjust=True).mean()
    
    # Calculate 26-day EMA
    ema_26 = df['Adj Close'].ewm(span=26, adjust=True).mean()
    
    # Calculate MACD line
    macd = ema_12 - ema_26
    
    return macd[t-1]

def AD(df,t):
    global ad_value 
    multiplier = ((df['Adj Close'][t-1] - df['Low'][t-1]) - (df['High'][t-1] - df['Adj Close'][t-1]))/(df['High'][t-1] - df['Adj Close'][t-1])
    if t == 1 :
     ad_value = multiplier*df['Volume'][0]
     return ad_value
    else :
     ad_value = ad_value + multiplier*df['Volume'][t-1]
     return ad_value

def OBV(df,t):
    global obv_value
    
    if t == 1:
        obv_value = 0
        return obv_value
    else :
        curr_volume = df['Volume'][t-1]
        money_flow_volume = np.sign(df['Adj Close'][t-1] - df['Adj Close'][t-2])*curr_volume
        obv_value = obv_value + money_flow_volume
        return obv_value
    
def PROC(df,t,W):
    if t <= W :
        proc_value = 0
        return proc_value
    else :
        curr_close = df['Adj Close'][t-1]
        prev_wind_close = df['Adj Close'][t-W-1]
        proc_value = ((curr_close - prev_wind_close)/(prev_wind_close))*100.0
        return proc_value

def William_R(df,t,W):

    if t < W :
        william_R_percent = -50.0
        return william_R_percent
    else :
        High_W = df['High'][t-W:t].max()
        Low_W =  df['Low'][t-W:t].min()
        william_R_percent = ((High_W - df['Adj Close'][t-1])/(High_W - Low_W))*-100
        return william_R_percent

def Disparity_index(df,t,W):

    ema_value = df['Adj Close'].ewm(span = W,adjust = True).mean()

    disparity_index = (df['Adj Close'][t-1] - ema_value[t-1])*100/(ema_value[t-1])

    return disparity_index



      