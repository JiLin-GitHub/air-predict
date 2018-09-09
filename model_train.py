# coding: utf-8

import lstm
import time
import matplotlib.pyplot as plt
import numpy as np
import os
import math
import sys
import pandas as pd
from pandas import Series, DataFrame, Panel
import keras
from keras.callbacks import ModelCheckpoint
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, Normalizer
from sklearn.model_selection import train_test_split
import re
from datetime import datetime
from sklearn.metrics import mean_squared_error
import sqlalchemy
from scipy.stats import logistic
import interpolation
import connection


class BatchGenerator:
    def __init__(self, file, oraengine, time_steps=48, scaler_type='standard', pca=False, pca_dim=8, normal=False):
        self.file = file
        self.time_steps = time_steps
        self.scaler_type = scaler_type
        self.pca = pca
        self.pca_dim = pca_dim
        self.normal = normal

        # 加载的数据类型DataFrame结构（pandas），19704 * 11
        # isnull判断的结果和data_loaded相同，只是用True或False替换
        # sum的结果是以列为单位统计为空的个数。结果是11*1的Series
        print("Loading data ...")
        data_loaded = pd.read_sql(file,oraengine)
        data_loaded.isnull().sum()
        print('初始值中各字段包含的Nan值\n',data_loaded.isnull().sum())
        print('初始值\n',data_loaded,'初始值')

        # load data
        # 取8种预测因子数据，转换成19704 * 8的矩阵---原始训练数据，每次都相同
        data_loaded_np = data_loaded.ix[:,3:].as_matrix()
        # print(data_loaded_np.shape)
        # print(data_loaded_np[1:100,0])


        # PCA
        if self.pca == True:
            print("PCA transform")
            pca = PCA(n_components=self.pca_dim, svd_solver='full')
            pca = pca.fit(data_loaded_np)
            data_loaded_np = pca.transform(data_loaded_np)

        # Row Normalization
        if self.normal == True:
            print("Normalize transform")
            self.norm_scaler = Normalizer().fit(data_loaded_np)
            data_loaded_np = self.norm_scaler.transform(data_loaded_np)


        self.date_proccessd, self.scaler_date_proccessd = self.generate_batch_data(data_loaded_np[0:, 0], name="data_processed")
        self.temp_pres_humid, self.scaler_tph = self.generate_batch_data(data_loaded_np[0:, 1:4], name="temp_pres_humid")
        self.wind_speed, self.scaler_wind_speed = self.generate_batch_data(data_loaded_np[0:, 4],name="wind_speed")
        self.wind_direction, self.scaler_wind_direction = self.generate_batch_data(data_loaded_np[0:, 5], name="wind_direction")
        self.rain, self.scaler_rain = self.generate_batch_data(data_loaded_np[0:, 6],name="rain")
        self.pm25_predict, self.scaler_pm25_predict = self.generate_batch_data(data_loaded_np[0:, 7:11],name="pm25_predict")
        self.pm25_final, self.scaler_pm25_final = self.generate_batch_data(data_loaded_np[0:, 11],name="pm25_final")


    def return_data(self):
        X_norm = pd.DataFrame()
        X_norm = pd.concat([self.date_proccessd,self.temp_pres_humid, self.rain,self.wind_direction,self.wind_speed,self.pm25_predict],axis=1)
        X_norm.columns = np.arange(X_norm.shape[1])
        Y_norm = self.pm25_final

        return X_norm.ix[:20000,:], Y_norm.ix[:20000], self.scaler_pm25_final

    def one_hot_encoding(self,raw_data):
        enc = preprocessing.OneHotEncoder()
        enc.fit(raw_data[:90000].reshape(-1,1))
        array = enc.transform(raw_data[:90000].reshape(-1,1)).toarray()
        return array

    def shift(self, arr, num, fill_value=np.nan):
        result = np.empty_like(arr)
        if num > 0:
            result[:num] = fill_value
            result[num:] = arr[:-num]
        elif num < 0:
            result[num:] = fill_value
            result[:num] = arr[-num:]
        else:
            result = arr
        return result

    def generate_batch_data(self, raw_data, name,timesteps=24):
        if "data_processed"==name or "wind_direction" == name or "wind_speed" == name:
            raw_data = self.one_hot_encoding(raw_data)

        raw_data = pd.DataFrame(raw_data[:90000])
        # print(values,values.shape,type(values))
        print('feature ------------ ', name.upper())

        if self.scaler_type == 'standard':
            scaler = StandardScaler()
        if self.scaler_type == 'robust':
            scaler = RobustScaler()
        if self.scaler_type == 'min_max':
            scaler = MinMaxScaler(feature_range=(0, 1))

        scaler = scaler.fit(raw_data)
        normalized = scaler.transform(raw_data)
        data = normalized

        print('Max: %f, Min: %f, Format: %d*%d' % (np.amax(data), np.amin(data), data.shape[0],data.shape[1]))
        data = pd.DataFrame(data)
        print(data)
        return data,scaler


def to_array(array):
    arr = array.values
    arr = arr.reshape(-1,array.shape[1],1)
    return arr


if __name__ == '__main__':
    #1、 分别返回当前实时时间：以秒为单位、年月日时分秒（秒精确多位）格式
    global_start_time = time.time()
    startTime = datetime.now()
    epochs = 3
    state_neurons_1 = 256
    state_neurons_2 = 512
    state_neurons_3 = 1024
    output = 1
    seed = 7
    batch_size = 100
    scaler_type = "standard"
    dropouts = [0.15, 0.25, 0.45]

    #2、 初始化，从数据库获得训练数据
    sql_factors = "SELECT POINTID,FACTOR_CODE,CREATE_DATE,DATE_PROCESSED,TEMPERATURE,PRESSURE,HUMIDITY,WIND_SPEED," \
                  "WIND_DIRECTION,RAINFALL,ONE_HOUR_BEFORE,TWO_HOUR_BEFORE,THREE_HOUR_BEFORE,ONE_DAY_BEFORE,FACTOR_VALUE " \
                  "FROM AQF_FORECAST_ONE_DAY_WX_PM25 WHERE factor_value < 0.70 AND one_hour_before < 0.7 AND " \
                  "two_hour_before < 0.7 AND three_hour_before < 0.7 AND one_day_before < 0.7 AND factor_value > 0  " \
                  "AND  pressure>=950 AND pressure<=1050 ORDER BY CREATE_DATE "
    oraengine = sqlalchemy.create_engine('oracle://scott:JL123456@localhost:1521/orcljl')
    batch_generator_obj = BatchGenerator(sql_factors,oraengine = oraengine, time_steps=48, scaler_type=scaler_type,
                                         pca=False, pca_dim=8, normal=False)
    X_norm, Y_norm, scaler = batch_generator_obj.return_data()
    X_norm = to_array(X_norm)
    Y_norm = Y_norm.values.reshape(-1,1)
    sys.exit(1)

    #3、 将训练数据分成训练集、测试集
    # random_state，它的用途是在随机划分训练集和测试集时候，划分的结果并不是那么随机，
    # 也即，确定下来random_state是某个值后，重复调用这个函数，划分结果是确定的。
    X_train_norm, X_val_norm, Y_train_norm, Y_val_norm = train_test_split(X_norm, Y_norm, test_size=0.20, random_state=seed)
    # print('训练集大小：', X_train_norm.shape)
    # print('测试集大小：', X_val_norm.shape)
    print('> Data Loaded. Compiling...')


    #4、 建立训练模型
    file_name = "wt_GRU_Keras_linear_rmsprop_" + scaler_type + "_dp(" + "_".join( str(x) for x in dropouts) + \
                ")_200_" + str(X_norm.shape[1]) + "_" + str(state_neurons_1) + "_" + str(state_neurons_2) + "_" + \
                str(batch_size)
    model = lstm.build_model([4, 7], [7, X_norm.shape[1], state_neurons_1, state_neurons_2, output], dropouts,
                             pre_train=file_name + ".h5")
    # print('建立的模型:',model)
    checkpoint = ModelCheckpoint(file_name+".h5", monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
#
#     #5、 训练模型
    mod_hist = model.fit([X_train_norm],[Y_train_norm], validation_data=([X_val_norm], [Y_val_norm]), epochs=epochs,
                         batch_size=batch_size, callbacks=callbacks_list, shuffle=True)
    print(mod_hist)
    print(mod_hist.history['loss'])
    print('---------------------')
    print(mod_hist.history['val_loss'])
    print('Training duration (s) : ', time.time() - global_start_time)
    print('Training duration (Hr) : ', datetime.now() - startTime)

    file = "Air_Predict_GRU_Keras_linear_rmsprop_" + scaler_type + "_dp(" + "_".join(str(x) for x in dropouts) + ")_200_" \
           + str(X_norm.shape[1]) + "_" + str(state_neurons_1) + "_" + str(state_neurons_2) + "_" + str(batch_size)
    print("Saving model : " + file + ".h5")
    model.save(file + ".h5")
    print("Model saved...")



#6、 利用目前多的模型进行预测
# 基于以上训练利用训练数据集、测试集、整体进行预测
if __name__=='__main__':
    #predict sequence
    _y_norm = lstm.predict_point_by_point_aux(model, [X_norm])
    _y_train = lstm.predict_point_by_point_aux(model, [X_train_norm])
    _y_val = lstm.predict_point_by_point_aux(model, [X_val_norm])


# 将以上测测得的结果，你变换成真实值、改变数据结构
if __name__=='__main__':
    _y_norm = _y_norm.reshape(-1,1)
    _y_train = _y_train.reshape(-1,1)
    _y_val = _y_val.reshape(-1,1)

    y_norm = Y_norm.reshape(-1,1)
    y_train = Y_train_norm.reshape(-1,1)
    y_val = Y_val_norm.reshape(-1,1)

    print("Predicted Output shape: ", np.shape(_y_norm))
    print("Original Output shape:  ", np.shape(y_norm))

    #inverse the predictions to its actual value
    print("Predicted Output sample: ")
    _y = scaler.inverse_transform(_y_norm)
    _y_tr = scaler.inverse_transform(_y_train)
    _y_va = scaler.inverse_transform(_y_val)
    for i in range(5):
        print(_y_va[i])

    #inverse the outputs to its actual value
    print("Original Output sample: ")
    y = scaler.inverse_transform(y_norm)
    y_tr = scaler.inverse_transform(y_train)
    y_va = scaler.inverse_transform(y_val)
    for i in range(5):
        print(y_va[i])

    y = np.exp(y)
    y_tr = np.exp(y_tr)
    y_va = np.exp(y_va)

    _y = np.exp(_y)
    _y_va = np.exp(_y_va)
    _y_tr = np.exp(_y_tr)

    #predicted
    # np.ravel(_y)：将_y转换成1维序列，并替换原始数据
    _Y = pd.Series(np.ravel(_y))
    _Y_TR = pd.Series(np.ravel(_y_tr))
    _Y_VA = pd.Series(np.ravel(_y_va))
#     _Y_TE = pd.Series(np.ravel(_y_te))

    #original
    Y =  pd.Series(np.ravel(y))
    Y_TR = pd.Series(np.ravel(y_tr))
    Y_VA = pd.Series(np.ravel(y_va))
#     Y_TE = pd.Series(np.ravel(y_te))



#7、 预测并图形化展示预测效果
# 计算各数据集预测值与真实值之间的偏差——均方差，并图形化展示
if __name__=='__main__':
    print('Total RMSE :', math.sqrt(mean_squared_error(_Y, Y)))
    print('Training RMSE :', math.sqrt(mean_squared_error(_Y_TR, Y_TR)))
    print('Validation RMSE :', math.sqrt(mean_squared_error(_Y_VA, Y_VA)))
#     print('Test RMSE :', math.sqrt(mean_squared_error(_Y_TE, Y_TE)))

    plt.figure(num=1,figsize=[14,7])
    # fig1_1 = plt.add_subplot(111)
    plot_predicted, = plt.plot(_Y_VA[100:500], label='predicted')
    plot_train, = plt.plot(Y_VA[100:500], label='actual')
    plt.legend(handles=[plot_predicted, plot_train])



# # summarize history for loss
# print(mod_hist.history['loss'])
# print(mod_hist.history['val_loss'])
# plt.plot(mod_hist.history['loss'])
# plt.plot(mod_hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
print('--------------------到非主函数这里\n')

# 载入训练好数据模型参数，重新初始化LSTM模型，只用测试集进行验证；图形化展示
if __name__=='__main__':
    #recreate model
    file_name = "wt_GRU_Keras_linear_rmsprop_" + scaler_type + "_dp(" + "_".join( str(x) for x in dropouts) + \
                ")_200_" + str(X_norm.shape[1]) + "_" + str(state_neurons_1) + "_" + str(state_neurons_2) + "_" + \
                str(batch_size)
    model = lstm.build_model([4, 7], [7, X_norm.shape[1], state_neurons_1, state_neurons_2, output], dropouts, pre_train=file_name+".h5")
    print("Created model and loaded weights from "+file_name+".h5")

    # estimate accuracy on whole dataset using loaded weights
    _y_val = lstm.predict_point_by_point_aux(model, [X_val_norm])
    _y_val = _y_val.reshape(-1,1)
    y_val = Y_val_norm.reshape(-1,1)

    #inverse the predictions to its actual value
    print("Predicted Output sample: ")
    _y_va = scaler.inverse_transform(_y_val)
    for i in range(5):
        print(_y_va[i])

    #inverse the outputs to its actual value
    print("Original Output sample: ")
    y_va = scaler.inverse_transform(y_val)
    for i in range(5):
        print(y_va[i])

    # y_va = np.exp(y_va)
    # _y_va = np.exp(_y_va)

    #predicted
    _Y_VA = pd.Series(np.ravel(_y_va))
    #original
    Y_VA = pd.Series(np.ravel(y_va))

    print('Validation RMSE :', math.sqrt(mean_squared_error(_Y_VA, Y_VA)))

    plt.figure(num=2,figsize=[14,7])
    # fig2_2 = fig2.add_subplot(111)

    plot_predicted, = plt.plot(_Y_VA[:5500], label='predicted')
    plot_train, = plt.plot(Y_VA[:5500], label='actual')
    plt.legend(handles=[plot_predicted, plot_train],loc='upper left')

plt.show()




def squash(x):
    return float(1.0 * x/(x+10))

