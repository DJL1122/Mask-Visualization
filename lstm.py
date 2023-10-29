import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def SEIR_model(parameters, population, duration):
    """
    SEIR模型
    """
    beta, gamma, sigma = parameters

    S = [population - 1]
    E = [1]
    I = [0]
    R = [0]

    for t in range(duration):
        dS = -beta * I[t] * S[t] / population
        dE = beta * I[t] * S[t] / population - sigma * E[t]
        dI = sigma * E[t] - gamma * I[t]
        dR = gamma * I[t]

        S.append(S[t] + dS)
        E.append(E[t] + dE)
        I.append([t] + dI)
        R.append(R[t] + dR)

    return S, E, I, R

population = 1000000
duration = 100
parameters = [0.3, 0.1, 0.2]  # 初始参数值

S, E, I, R = SEIR_model(parameters, population, duration)

# 将数据转换为numpy数组并进行归一化处理
data = np.array([S, E, I, R]).T
data = (data - np.min(data)) / (np.max(data) - np.min(data))

sequence_length = 10  # 每个样本的时间序列长度
output_dimension = 4  #输出维度的数量

train_X = []
train_y = []

for i in range(len(data) - sequence_length):
    train_X.append(data[i:i+sequence_length])
    train_y.append(parameters)  # 用真实参数作为标签

train_X = np.array(train_X)
train_y = np.array(train_y)

model = Sequential()
model.add(LSTM(64, input_shape=(sequence_length, output_dimension)))
model.add(Dense(512, activation='relu'))
model.add(Dense(output_dimension))

model.compile(optimizer='adam', loss='mse')
model.fit(train_X, train_y, epochs=100, batch_size=32)

test_data = data[-sequence_length:]  # 使用最新的数据进行预测
test_data = np.reshape(test_data, (1, sequence_length, output_dimension))

estimated_params = model.predict(test_data)
estimated_params = np.squeeze(estimated_params)

print("Estimated parameters:")
print("Beta:", estimated_params[0])
print("Gamma:", estimated_params[1])
print("Sigma:", estimated_params[2])