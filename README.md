import pandas as pd

import numpy as np

from keras.models import Sequential

from keras.layers import Dense, Dropout

from keras.callbacks import EarlyStopping

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score

# 加载数据

data = pd.read_csv('lottery.csv')

# 去除无效数据

data.dropna(inplace=True)

# 处理日期特征

data['日期'] = pd.to_datetime(data['日期'])

# 处理开奖号码特征

data['开奖号码'] = data['开奖号码'].apply(lambda x: [int(i) for i in x.split(',')])

for i in range(6):

    data[f'num_{i+1}'] = data['开奖号码'].apply(lambda x: x[i])

data.drop('开奖号码', axis=1, inplace=True)

# 处理生肖和波色特征

animals = ['鼠', '牛', '虎', '兔', '龙', '蛇', '马', '羊', '猴', '鸡', '狗', '猪']

colors = ['红', '蓝', '绿']

for i in range(12):

    data[f'animal_{animals[i]}'] = data['生肖'].apply(lambda x: int(x == animals[i]))

for i in range(3):

    data[f'color_{colors[i]}'] = data['波色'].apply(lambda x: int(x == colors[i]))

data.drop(['生肖', '波色'], axis=1, inplace=True)

# 分离特征和目标

X = data.drop(['日期', '期数', 'num_1', 'num_2', 'num_3', 'num_4', 'num_5', 'num_6'], axis=1)

y = data[['num_1', 'num_2', 'num_3', 'num_4', 'num_5', 'num_6']]

# 划分训练集和测试集

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 进行特征缩放

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

# 构建神经网络模型

model = Sequential()

model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(32, activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(16, activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(6))

model.compile(loss='mean_squared_error', optimizer='adam')

# 定义早停法回调

early_stopping = EarlyStopping(monitor='val_loss', patience=10)

# 训练模型

history = model.fit(

    X_train,

    y_train,

    epochs=1000,

    batch_size=32,

    validation_split=0.2,

    callbacks=[early_stopping],

    verbose=2

)

# 评估模型

y_pred = model.predict(X_test)

print(f'R2 score: {r2_score(y_test, y_pred)}')
