import os
import time
import warnings
import numpy as np
import keras
from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import LSTM, GRU, CuDNNGRU, CuDNNLSTM, Input, TimeDistributed, Flatten, Bidirectional, Conv1D, MaxPooling1D
from keras.layers.normalization import BatchNormalization
# from keras.layers.recurrent import GRU
from keras import regularizers
from keras.models import Sequential, Model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Hide messy TensorFlow warnings
warnings.filterwarnings("ignore") #Hide messy Numpy warnings

def load_data(filename, seq_len, normalise_window):
    f = open(filename, 'rb').read()
    data = f.decode().split('\n')

    sequence_length = seq_len + 1
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])

    if normalise_window:
        result = normalise_windows(result)

    result = np.array(result)

    row = round(0.9 * result.shape[0])
    train = result[:int(row), :]
    np.random.shuffle(train)
    x_train = train[:, :-1]
    y_train = train[:, -1]
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    return [x_train, y_train, x_test, y_test]

def normalise_windows(window_data):
    normalised_data = []
    for window in window_data:
        normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
        normalised_data.append(normalised_window)
    return normalised_data

def build_model(en_layers, layers, dropouts, pre_train=None):
    # return
    # layer[1] = seq_len
    # layer[0] = dimensions
    # layer[2], layer[3] = state_neurons
    # layer[4] = output
    #
    # batch normalize before elu layer at axis =-1
    # check wat BN for differernt axis in keras

    # 卷积核的数目：128
    # 卷积核的大小：8
    # 权值初始化：glorot_uniform，Glorot均匀分布初始化方法
    # inputs = Input(shape=(layers[1],1))
    # print(inputs)
    # conv_1 = (Conv1D(128, 6, kernel_initializer='glorot_uniform', activation='elu'))(inputs)
    # conv_1 = BatchNormalization()(conv_1)
    # print('到这里1')
    # print(inputs)
    # print(conv_1)

    # # 1.date_proccessed
    # inputs_dp = Input(shape=(layers[1], layers[0]))
    # conv_1_dp = (Conv1D(128, 3, kernel_initializer='glorot_uniform', activation='elu'))(inputs_dp)
    # conv_1_dp = BatchNormalization()(conv_1_dp)
    # print('到这里1')
    # print(inputs_dp)
    # print(conv_1_dp)
    # # conv_1_pm25 = MaxPooling1D(4)(conv_1_pm25)

    # 2.temprature
    inputs_temp = Input(shape=(layers[1], layers[0]))
    conv_1_temp = (Conv1D(128, 3, kernel_initializer='glorot_uniform', activation='elu'))(inputs_temp)
    conv_1_temp = BatchNormalization()(conv_1_temp)
    print('到这里2')
    print(conv_1_temp)
    # return 0
    # # conv_1_ws = MaxPooling1D(4)(conv_1_ws)

    # 3.pressure
    inputs_pressure = Input(shape=(layers[1], layers[0]))
    conv_1_pressure = (Conv1D(128, 3, kernel_initializer='glorot_uniform', activation='elu'))(inputs_pressure)
    conv_1_pressure = BatchNormalization()(conv_1_pressure)
    print('到这里3')
    print(conv_1_pressure)

    # 4.humidity
    inputs_humid = Input(shape=(layers[1], layers[0]))
    conv_1_humid = (Conv1D(128, 3, kernel_initializer='glorot_uniform', activation='elu'))(inputs_humid)
    conv_1_humid = BatchNormalization()(conv_1_humid)
    print('到这里4')
    print(conv_1_humid)

    # 5.wind_speed
    inputs_ws = Input(shape=(layers[1], layers[0]))
    conv_1_ws = (Conv1D(128, 3, kernel_initializer='glorot_uniform', activation='elu'))(inputs_ws)
    conv_1_ws = BatchNormalization()(conv_1_ws)
    print('到这里5')
    print(conv_1_ws)

    # 6.wind_direction
    inputs_wd = Input(shape=(layers[1], layers[0]))
    conv_1_wd = (Conv1D(128, 3, kernel_initializer='glorot_uniform', activation='elu'))(inputs_wd)
    conv_1_wd = BatchNormalization()(conv_1_wd)
    print('到这里6')
    print(conv_1_wd)

    # 7.rainfall
    inputs_rainfall = Input(shape=(layers[1], layers[0]))
    conv_1_rainfall = (Conv1D(128, 3, kernel_initializer='glorot_uniform', activation='elu'))(inputs_rainfall)
    conv_1_rainfall = BatchNormalization()(conv_1_rainfall)
    print('到这里7')
    print(conv_1_rainfall)
    # return 0
    # # conv_1_rh = MaxPooling1D(4)(conv_1_rh)

    # 8.PM25
    inputs_pm25 = Input(shape=(layers[1], layers[0]))
    conv_1_pm25 = (Conv1D(128, 3, kernel_initializer='glorot_uniform', activation='elu'))(inputs_pm25)
    conv_1_pm25 = BatchNormalization()(conv_1_pm25)
    print('到这里8')
    print(conv_1_pm25)
    # return 0
    # # conv_1_bp = MaxPooling1D(4)(conv_1_bp)

    # 9.PM10
    inputs_pm10 = Input(shape=(layers[1], layers[0]))
    conv_1_pm10 = (Conv1D(128, 3, kernel_initializer='glorot_uniform', activation='elu'))(inputs_pm10)
    conv_1_pm10 = BatchNormalization()(conv_1_pm10)
    print('到这里9')
    print(conv_1_pm10)
    # # conv_1_vws = MaxPooling1D(4)(conv_1_vws)

    # 10.sr
    inputs_no2 = Input(shape=(layers[1], layers[0]))
    conv_1_no2 = (Conv1D(128, 3, kernel_initializer='glorot_uniform', activation='elu'))(inputs_no2)
    conv_1_no2 = BatchNormalization()(conv_1_no2)
    print('到这里10')
    print(conv_1_no2)
    # # conv_1_sr = MaxPooling1D(4)(conv_1_sr)

    # 11.O3
    inputs_o3 = Input(shape=(layers[1], layers[0]))
    conv_1_o3 = (Conv1D(128, 3, kernel_initializer='glorot_uniform', activation='elu'))(inputs_o3)
    conv_1_o3 = BatchNormalization()(conv_1_o3)
    print('到这里11')
    print(conv_1_o3)
    # # conv_1_wd = MaxPooling1D(4)(conv_1_wd)

    # 12.SO2
    inputs_so2 = Input(shape=(layers[1], layers[0]))
    conv_1_so2 = (Conv1D(128, 3, kernel_initializer='glorot_uniform', activation='elu'))(inputs_so2)
    conv_1_so2 = BatchNormalization()(conv_1_so2)
    print('在这里12')
    print(conv_1_so2)

    # 13.CO
    inputs_co = Input(shape=(layers[1], layers[0]))
    conv_1_co = (Conv1D(128, 3, kernel_initializer='glorot_uniform', activation='elu'))(inputs_co)
    conv_1_co = BatchNormalization()(conv_1_co)
    print('在这里13')
    print(conv_1_co)

    # # concatenate
    # output = keras.layers.concatenate([conv_1_dp, conv_1_temp, conv_1_pressure, conv_1_humid, conv_1_ws, conv_1_wd,
    #                                    conv_1_rainfall, conv_1_pm25, conv_1_pm10, conv_1_no2, conv_1_o3, conv_1_so2,
    #                                    conv_1_co])
    output = keras.layers.concatenate([conv_1_temp, conv_1_pressure, conv_1_humid, conv_1_ws, conv_1_wd,conv_1_rainfall,
                                       conv_1_pm25, conv_1_pm10, conv_1_no2, conv_1_o3, conv_1_so2,conv_1_co])
    print('到这里9')
    print(output)

    lstm_11 = (GRU(layers[2],return_sequences=True,kernel_initializer='glorot_uniform',recurrent_initializer='orthogonal',
                            bias_initializer='zeros'))(output)
    # lstm_1 = (CuDNNGRU(layers[2],return_sequences=True, kernel_initializer='glorot_uniform',
    #  recurrent_initializer='orthogonal', bias_initializer='zeros'))(output)
    print('到这里10')
    print(lstm_11)

    lstm_11 = Dropout(dropouts[0])(lstm_11)
    print(lstm_11)

    lstm_22 = (GRU(layers[3],return_sequences=True,kernel_initializer='glorot_uniform',recurrent_initializer='orthogonal',
                   bias_initializer='zeros'))(lstm_11)
    # lstm_2 = (CuDNNGRU(layers[2],return_sequences=True, kernel_initializer='glorot_uniform',
    #  recurrent_initializer='orthogonal', bias_initializer='zeros'))(lstm_1)
    lstm_22 = Dropout(dropouts[1])(lstm_22)
    print(lstm_22)
    # return 0

    # 在每个timestep上都执行独立的Dense操作，这里是将Dense应用到(?,?,512)(GRU的第二维度timesteps上) 512维数据上，输出维度是layer[3]
    output = TimeDistributed(Dense(layers[3], activation='tanh'))(lstm_22)
    print('到这里11')
    print(output)

    output = BatchNormalization()(output)
    output = TimeDistributed(Dense(1, activation='tanh'))(output)
    output = BatchNormalization()(output)
    print(output)
    # return 0

    # 拉平数据，真正的全连接
    output = Flatten()(output)
    print(output)
    output = (Dense(layers[-1], activation='linear'))(output)
    print(output)
    # return 0
    # model = Model(inputs=[inputs_dp, inputs_temp, inputs_pressure, inputs_humid, inputs_ws, inputs_wd, inputs_rainfall,
    #                       inputs_pm25, inputs_pm10, inputs_no2, inputs_o3, inputs_so2, inputs_co], outputs=[output])
    model = Model(inputs=[inputs_temp, inputs_pressure, inputs_humid, inputs_ws, inputs_wd, inputs_rainfall,
                          inputs_pm25, inputs_pm10, inputs_no2, inputs_o3, inputs_so2, inputs_co], outputs=[output])

    print('到这里12')
    print(model)
    # return 0


    if not (pre_train is None):
        model.load_weights(pre_train)
        print('进来了,载入预训练模型参数')
    print(model)

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop")
    print('到这里13')
    print(model)
    # return 0
    print(model.summary())
    # return 0
    print("> Compilation Time : ", time.time() - start)
    # return 0
    return model

def predict_point_by_point_aux(model, data):
    #Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
    predicted = model.predict(data)
    predicted = np.reshape(predicted, (predicted.size,))
    return predicted

def predict_point_by_point(model, data):
    #Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
    predicted = model.predict(data)
    predicted = np.reshape(predicted, (predicted.size,))
    return predicted

def predict_sequence_full(model, data, window_size):
    #Shift the window by 1 new prediction each time, re-run predictions on new window
    curr_frame = data[0]
    predicted = []
    for i in range(len(data)):
        predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
        curr_frame = curr_frame[1:]
        print(curr_frame)
        curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
        print(curr_frame)
    return predicted

def predict_sequences_multiple(model, data, window_size, prediction_len):
    #Predict sequence of 50 steps before shifting prediction run forward by 50 steps
    prediction_seqs = []
    for i in range(int(len(data)/prediction_len)):
        curr_frame = data[i*prediction_len]
        predicted = []
        for j in range(prediction_len):
            predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
        prediction_seqs.append(predicted)
    return prediction_seqs
