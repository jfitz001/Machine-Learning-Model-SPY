import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
import pandas as pd
import numpy as np
from numpy import array
import keras
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Activation
from keras.layers import TimeDistributed  
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import ta
import xgboost as xgb

def add_indicators(df):
    df['RSI'] = ta.rsi(df["Close"])
    df['MFI'] = ta.money_flow_index(
        df["High"], df["Low"], df["Close"], df["Volume"])
    df['TSI'] = ta.tsi(df["Close"])
    df['UO'] = ta.uo(df["High"], df["Low"], df["Close"])
    df['AO'] = ta.ao(df["High"], df["Low"])

    df['MACD_diff'] = ta.macd_diff(df["Close"])
    df['Vortex_pos'] = ta.vortex_indicator_pos(
        df["High"], df["Low"], df["Close"])
    df['Vortex_neg'] = ta.vortex_indicator_neg(
        df["High"], df["Low"], df["Close"])
    df['Vortex_diff'] = abs(
        df['Vortex_pos'] -
        df['Vortex_neg'])
    df['Trix'] = ta.trix(df["Close"])
    df['Mass_index'] = ta.mass_index(df["High"], df["Low"])
    df['CCI'] = ta.cci(df["High"], df["Low"], df["Close"])
    df['DPO'] = ta.dpo(df["Close"])
    df['KST'] = ta.kst(df["Close"])
    df['KST_sig'] = ta.kst_sig(df["Close"])
    df['KST_diff'] = (
        df['KST'] -
        df['KST_sig'])
    df['Aroon_up'] = ta.aroon_up(df["Close"])
    df['Aroon_down'] = ta.aroon_down(df["Close"])
    df['Aroon_ind'] = (
        df['Aroon_up'] -
        df['Aroon_down']
    )

    df['BBH'] = ta.bollinger_hband(df["Close"])
    df['BBL'] = ta.bollinger_lband(df["Close"])
    df['BBM'] = ta.bollinger_mavg(df["Close"])
    df['BBHI'] = ta.bollinger_hband_indicator(
        df["Close"])
    df['BBLI'] = ta.bollinger_lband_indicator(
        df["Close"])
    df['KCHI'] = ta.keltner_channel_hband_indicator(df["High"],
                                                    df["Low"],
                                                    df["Close"])
    df['KCLI'] = ta.keltner_channel_lband_indicator(df["High"],
                                                    df["Low"],
                                                    df["Close"])
    df['DCHI'] = ta.donchian_channel_hband_indicator(df["Close"])
    df['DCLI'] = ta.donchian_channel_lband_indicator(df["Close"])

    df['ADI'] = ta.acc_dist_index(df["High"],
                                  df["Low"],
                                  df["Close"],
                                  df["Volume"])
    df['OBV'] = ta.on_balance_volume(df["Close"],
                                     df["Volume"])
    df['CMF'] = ta.chaikin_money_flow(df["High"],
                                      df["Low"],
                                      df["Close"],
                                      df["Volume"])
    df['FI'] = ta.force_index(df["Close"],
                              df["Volume"])
    df['EM'] = ta.ease_of_movement(df["High"],
                                   df["Low"],
                                   df["Close"],
                                   df["Volume"])
    df['VPT'] = ta.volume_price_trend(df["Close"],
                                      df["Volume"])
    df['NVI'] = ta.negative_volume_index(df["Close"],
                                         df["Volume"])

    df['DR'] = ta.daily_return(df["Close"])
    df['DLR'] = ta.daily_log_return(df["Close"])

    df.fillna(method='bfill', inplace=True)

    return df

keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

df = pd.read_csv("SPY.csv")
df = df.dropna()
df = df[["Open", "High", "Low", "Close", "Volume"]]
add_indicators(df)

df['diffed'] = df['Close'] - df['Close'].shift(1)
df['logged_and_diffed'] = np.log(df['Close']) - np.log(df['Close']).shift(1)


df['Price_Rise'] = np.where(df['logged_and_diffed'] > df['logged_and_diffed'].shift(1), 1, 0)

df = df.dropna()

X = df.iloc[:,:-1]
y = df.iloc[:,-1]


split = int(len(df)*0.8)
X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# reshape input into [samples, timesteps, features]


X_train = np.array(X_train)
X_train = np.reshape(X_train,(1, X_train.shape[0], 1, X_train.shape[1]))



n_out = X_train.shape[0]
# define model
model = Sequential()
model.add(LSTM(128, activation='relu', input_shape=(X_train.shape[0],)))
#model.add(RepeatVector(n_out))
#model.add(LSTM(128, activation='relu', return_sequences=True))
model.add(Dense(1))

#es = EarlyStopping(monitor='val_loss', mode='auto', verbose=0, patience=50)

model.compile(loss="mean_squared_error" , optimizer = "adam", metrics=['accuracy'])
print(model.summary())
history = model.fit(X_train, y_train, batch_size = 256, epochs=512, verbose = 2)
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)



df['y_pred'] = np.NaN
df.iloc[(len(df) - len(y_pred)):,-1:] = y_pred
trade_dataset = df.dropna()

trade_dataset['Tomorrows Returns'] = 0.
trade_dataset['Tomorrows Returns'] = np.log(trade_dataset['Close']/trade_dataset['Close'].shift(1))
trade_dataset['Tomorrows Returns'] = trade_dataset['Tomorrows Returns'].shift(-1)

trade_dataset['Strategy Returns'] = 0.
trade_dataset['Strategy Returns'] = np.where(trade_dataset['y_pred'] == True, trade_dataset['Tomorrows Returns'], -trade_dataset['Tomorrows Returns'])

trade_dataset['Cumulative Market Returns'] = np.cumsum(trade_dataset['Tomorrows Returns'])
trade_dataset['Cumulative Strategy Returns'] = np.cumsum(trade_dataset['Strategy Returns'])

plt.plot(trade_dataset['Cumulative Market Returns'], color='r', label='Market Returns')
plt.plot(trade_dataset['Cumulative Strategy Returns'], color='g', label='Strategy Returns')




plt.style.use("ggplot")
plt.xlabel("Dates")
plt.ylabel("Gain/Loss % (in decimals)")
plt.legend()
plt.show()
print(df['y_pred'])
