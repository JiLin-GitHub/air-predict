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
from keras.callbacks import ModelCheckpoint,EarlyStopping,TensorBoard
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, Normalizer
from sklearn.model_selection import train_test_split
import re
from datetime import datetime
from sklearn.metrics import mean_squared_error,mean_absolute_error
import sqlalchemy
from scipy.stats import logistic
import interpolation
import connection

STAIONNUM = 11


class BatchGenerator:
    def __init__(self, file, oraengine, timesteps=48, output=24,scaler_type='standard', pca=False, pca_dim=8, normal=False,factor_num=''):
        self.file = file
        self.timesteps = timesteps
        self.output = output
        self.scaler_type = scaler_type
        self.pca = pca
        self.pca_dim = pca_dim
        self.normal = normal
        self.factor_num = factor_num

        # 加载的数据类型DataFrame结构（pandas），19704 * 11
        # isnull判断的结果和data_loaded相同，只是用True或False替换
        # sum的结果是以列为单位统计为空的个数。结果是11*1的Series
        print("Loading data ...")
        data_loaded = pd.read_sql(file,oraengine)
        data_loaded.ix[:,'date_processed'] = data_loaded.ix[:,'date_processed']/25  #25:该字段数字范围是1-24，目的是将其归一化
        data_loaded.ix[:,'wind_speed'] = data_loaded.ix[:,'wind_speed']/10          #10:理由同上
        data_loaded.ix[:, 'wind_direction'] = data_loaded.ix[:, 'wind_direction'] / 17  #17:理由同上
        print('初始值中各字段包含的Nan值\n', data_loaded.isnull().sum())
        print(data_loaded)

        data_target = data_loaded[data_loaded['pointid'] == '8']
        print(data_target)
        # load data
        # 取8种预测因子数据，转换成19704 * 8的矩阵---原始训练数据，每次都相同
        data_loaded_np = data_loaded.ix[:,2:].as_matrix()
        data_target = data_target.as_matrix()
        print(data_loaded_np.shape)
        print(data_loaded_np[:20,10:12])
        print(data_target[:,9])

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


        self.date_proccessd, self.scaler_date_proccessd = self.generate_batch_data(data_loaded_np[0:, 0], name="date_processed", timesteps=self.timesteps)

        self.temp, self.scaler_temp = self.generate_batch_data(data_loaded_np[0:, 1], name="temp", timesteps=self.timesteps)

        self.pressure, self.scaler_pressure = self.generate_batch_data(data_loaded_np[0:, 2], name="pressure", timesteps=self.timesteps)

        self.humid, self.scaler_humid = self.generate_batch_data(data_loaded_np[0:, 3], name="humid", timesteps=self.timesteps)

        self.wind_speed, self.scaler_wind_speed = self.generate_batch_data(data_loaded_np[0:, 4],name="wind_speed", timesteps=self.timesteps)

        self.wind_direction, self.scaler_wind_direction = self.generate_batch_data(data_loaded_np[0:, 5], name="wind_direction", timesteps=self.timesteps)

        self.rain, self.scaler_rain = self.generate_batch_data(data_loaded_np[0:, 6],name="rain", timesteps=self.timesteps)

        self.pm25, self.scaler_pm25 = self.generate_batch_data(data_loaded_np[0:, 7],name="pm25", timesteps=self.timesteps)

        self.pm10, self.scaler_pm10 = self.generate_batch_data(data_loaded_np[0:, 8],name="pm10", timesteps=self.timesteps)

        self.no2, self.scaler_no2 = self.generate_batch_data(data_loaded_np[0:, 9], name="no2", timesteps=self.timesteps)

        self.o3, self.scaler_o3 = self.generate_batch_data(data_loaded_np[0:, 10], name="o3", timesteps=self.timesteps)

        self.so2, self.scaler_so2 = self.generate_batch_data(data_loaded_np[0:, 11], name="so2", timesteps=self.timesteps)

        self.co, self.scaler_co = self.generate_batch_data(data_loaded_np[0:, 12], name="co", timesteps=self.timesteps)

        self.target, self.scaler_target = self.generate_batch_data(data_target[:,9], name="target", timesteps=output)

        if not (self.scaler_type is None):
            if self.factor_num == '1':
                filename = "np_" + self.scaler_type + "_process_comp_" + str(self.timesteps) + "_" + str(
                    self.pca) + "_" + str(self.normal) + ".npz"
            elif self.factor_num == '2':
                filename = "val_np_" + self.scaler_type + "_process_comp_" + str(self.timesteps) + "_" + str(
                    self.pca) + "_" + str(self.normal) + ".npz"
            else:
                pass
        else:
            if self.factor_num == '1':
                filename = "np_process_comp_" + str(self.timesteps) + "_" + str(self.pca) + "_" + str(self.normal) + ".npz"
            elif self.factor_num == '2':
                filename = "val_np_process_comp_" + str(self.timesteps) + "_" + str(self.pca) + "_" + str(self.normal) + ".npz"
            else:
                pass

        if os.path.isfile("data_log/" + filename):
            print("Found existing file :", "data_log/" + filename)
            print("Loading ...")
            npzfile = np.load("data_log/" + filename)
            # print(npzfile['arr_0'])
            self.date_proccessd = npzfile['arr_0']
            self.temp = npzfile['arr_1']
            self.pressure = npzfile['arr_2']
            self.humid = npzfile['arr_3']
            self.wind_speed = npzfile['arr_4']
            self.wind_direction = npzfile['arr_5']
            self.rain = npzfile['arr_6']
            self.pm25 = npzfile['arr_7']
            self.pm10 = npzfile['arr_8']
            self.no2 = npzfile['arr_9']
            self.o3 = npzfile['arr_10']
            self.so2 = npzfile['arr_11']
            self.co = npzfile['arr_12']
            self.target = npzfile['arr_13']
            # print('真实的训练数据集', data_loaded_np[0:, 1].max(),data_loaded_np[0:, 1].min())
            print("Complete.")
        else:
            # self.Y = self.y_norm_pm
            print("Input shape pm25:", np.shape(self.date_proccessd))
            print("Input shape ws:", np.shape(self.temp))
            print("Input shape rh:", np.shape(self.pressure))
            print("Input shape bp:", np.shape(self.humid))
            print("Input shape vws:", np.shape(self.wind_speed))
            print("Input shape sr:", np.shape(self.wind_direction))
            print("Input shape wd:", np.shape(self.rain))
            print("Input shape temp:", np.shape(self.pm25))
            print("Input shape temp:", np.shape(self.pm10))
            print("Input shape temp:", np.shape(self.no2))
            print("Input shape temp:", np.shape(self.o3))
            print("Input shape temp:", np.shape(self.so2))
            print("Input shape temp:", np.shape(self.co))
            print("Output shape :", np.shape(self.target))
            print("Saving file ...")
            np.savez("data_log/" + filename, self.date_proccessd, self.temp, self.pressure, self.humid,
                     self.wind_speed, self.wind_direction, self.rain, self.pm25, self.pm10, self.no2, self.o3,
                     self.so2, self.co, self.target)
            print("Saved file to :", filename)
            # print('真实的训练数据集',data_loaded_np[0:,0])
            print("Complete.")


    def return_data(self):
        return self.date_proccessd, self.temp, self.pressure, self.humid, self.wind_speed, self.wind_direction, \
               self.rain, self.pm25, self.pm10, self.no2, self.o3, self.so2, self.co, self.target, self.scaler_target

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

    def generate_batch_data(self, raw_data, name, timesteps=24):
        # if "data_processed"==name or "wind_direction" == name or "wind_speed" == name:
        #     raw_data = self.one_hot_encoding(raw_data)
        raw_data = pd.DataFrame(raw_data)
        value = raw_data.values
        print('feature ------------ ', name.upper())

        if self.scaler_type == 'standard':
            scaler = StandardScaler()
        if self.scaler_type == 'robust':
            scaler = RobustScaler()
        if self.scaler_type == 'min_max':
            scaler = MinMaxScaler(feature_range=(0, 1))

        scaler = scaler.fit(value)
        normalized = scaler.transform(value)
        data = normalized

        print('Max: %f, Min: %f, Format: %d*%d' % (np.amax(data), np.amin(data), data.shape[0],data.shape[1]))


        if name != 'target':
            # 训练数据-48有关
            input_serise = data[:(len(data)-24*11)]
            x_batches = np.array([])
        else:
            # 目标数据-24有关
            target_serise = self.shift(data,-(timesteps)).astype(np.float32)
            y_batches = np.array([])


        # check if file exists
        if (self.scaler_type is None):
            if self.factor_num == '1':
                seq_file_name = "np_processed_" + name + "_" + str(timesteps) + "_" + str(self.pca) + "_" + str(
                    self.normal) + ".npz"
            elif self.factor_num == '2':
                seq_file_name = "val_np_processed_" + name + "_" + str(timesteps) + "_" + str(self.pca) + "_" + str(
                    self.normal) + ".npz"
            else:
                pass
        else:
            if self.factor_num == '1':
                seq_file_name = "np_" + self.scaler_type + "_processed_" + name + "_" + str(timesteps) + "_" + str(
                    self.pca) + "_" + str(self.normal) + ".npz"
            elif self.factor_num == '2':
                seq_file_name = "val_np_" + self.scaler_type + "_processed_" + name + "_" + str(timesteps) + "_" + str(
                    self.pca) + "_" + str(self.normal) + ".npz"
            else:
                pass

        if os.path.isfile("data_log/" + seq_file_name):
            npzfile = np.load("data_log/" + seq_file_name)
            if name != 'target':
                input_batches = npzfile['arr_0']
                ret = input_batches
            else:
                target_batches = npzfile['arr_0']
                ret = target_batches
            return ret, scaler
        else:
            if self.factor_num == '1':
                for i in range(5415):
                    try:
                        if name != 'target':
                            x_batches = np.append(x_batches, input_serise[i*11:(i+timesteps)*11].reshape(-1, timesteps, 11))
                        else:
                            y_batches = np.append(y_batches, target_serise[i:i+timesteps].reshape(-1,timesteps))
                    except ValueError:
                        break
            elif self.factor_num == '2':
                for i in range(783):
                    try:
                        if name != 'target':
                            x_batches = np.append(x_batches, input_serise[i*11:(i+timesteps)*11].reshape(-1, timesteps, 11))
                        else:
                            y_batches = np.append(y_batches, target_serise[i:i+timesteps].reshape(-1,timesteps))
                    except ValueError:
                        break
            else:
                sys.exit(1001)


            if name != 'target':
                x_batches = x_batches.reshape(-1, timesteps, 11)
                np.savez("data_log/" + seq_file_name, x_batches)
                return x_batches, scaler
            else:
                y_batches = y_batches.reshape(-1, timesteps)
                np.savez("data_log/" + seq_file_name, y_batches)
                return y_batches, scaler


def to_array(array):
    arr = array.values
    arr = arr.reshape(-1,array.shape[1],1)
    return arr


if __name__ == '__main__':
    #1、分别返回当前实时时间：以秒为单位、年月日时分秒（秒精确多位）格式
    global_start_time = time.time()
    startTime = datetime.now()
    epochs = 50
    state_neurons_1 = 256
    state_neurons_2 = 512
    state_neurons_3 = 1024
    output = 24
    seed = 7
    batch_size = 100
    scaler_type = "standard"
    dropouts = [0.15, 0.25, 0.45]
    seq_len = 48

    #2、 初始化，从数据库获得训练数据
    sql_factors = "SELECT * FROM TRAININGDATA ORDER BY CREATE_DATE,TO_NUMBER(POINTID) "
    sql_factors_val = "SELECT * FROM VALDATA ORDER BY CREATE_DATE,TO_NUMBER(POINTID) "
    oraengine = sqlalchemy.create_engine('oracle://scott:JL123456@localhost:1521/orcljl')
    batch_generator_obj1 = BatchGenerator(sql_factors,oraengine = oraengine, timesteps=seq_len, output=output ,
                                          scaler_type=scaler_type,pca=False, pca_dim=8, normal=False,factor_num='1')
    batch_generator_obj2 = BatchGenerator(sql_factors_val, oraengine=oraengine, timesteps=seq_len, output=output,
                                          scaler_type=scaler_type, pca=False, pca_dim=8, normal=False, factor_num='2')

    X_date_proccessd, X_temp, X_pressure, X_humid, X_wind_speed, X_wind_direction, X_rain, X_pm25, X_pm10, X_no2, X_o3,\
    X_so2, X_co, Y_target, scaler= batch_generator_obj1.return_data()

    X_date_proccessd_val, X_temp_val, X_pressure_val, X_humid_val, X_wind_speed_val, X_wind_direction_val,\
    X_rain_val, X_pm25_val, X_pm10_val, X_no2_val, X_o3_val, X_so2_val, X_co_val, Y_target_val, scaler_val = batch_generator_obj2.return_data()

    # 合并生成两个训练集
    X_date_proccessd = np.concatenate((X_date_proccessd,X_date_proccessd_val),axis=0)
    X_temp = np.concatenate((X_temp,X_temp_val),axis=0)
    X_pressure = np.concatenate((X_pressure,X_pressure_val) ,axis=0)
    X_humid = np.concatenate((X_humid,X_humid_val), axis=0)
    X_wind_speed = np.concatenate((X_wind_speed,X_wind_speed_val), axis=0)
    X_wind_direction = np.concatenate((X_wind_direction, X_wind_direction_val), axis=0)
    X_rain = np.concatenate((X_rain, X_rain_val), axis=0)
    X_pm25 = np.concatenate((X_pm25, X_pm25_val), axis=0)
    X_pm10 = np.concatenate((X_pm10, X_pm10_val), axis=0)
    X_no2 = np.concatenate((X_no2,X_no2_val), axis=0)
    X_o3 = np.concatenate((X_o3,X_o3_val), axis=0)
    X_so2 = np.concatenate((X_so2, X_so2_val), axis=0)
    X_co = np.concatenate((X_co, X_co_val), axis=0)
    Y_target = np.concatenate((Y_target,Y_target_val), axis=0)

    print(type(X_date_proccessd),X_date_proccessd.shape)


    #3、 将训练数据分成训练集、测试集
    # random_state，它的用途是在随机划分训练集和测试集时候，划分的结果并不是那么随机，
    # 也即，确定下来random_state是某个值后，重复调用这个函数，划分结果是确定的。
    X_train_dp,X_val_dp, X_train_temp, X_val_temp, X_train_pressure, X_val_pressure, X_train_humid, X_val_humid, \
    X_train_ws, X_val_ws, X_train_wd, X_val_wd, X_train_rain, X_val_rain, X_train_pm25, X_val_pm25, X_train_pm10, X_val_pm10,\
    X_train_no2, X_val_no2,X_train_o3, X_val_o3,X_train_so2, X_val_so2, X_train_co, X_val_co, Y_train_target, Y_val_target = \
        train_test_split(X_date_proccessd, X_temp, X_pressure, X_humid, X_wind_speed, X_wind_direction, X_rain, X_pm25, X_pm10,
                         X_no2, X_o3, X_so2, X_co, Y_target, test_size=0.20, random_state=seed)

    # # 手动分离训练集、验证集
    # X_train_dp = X_date_proccessd[:4332]
    # X_val_dp = X_date_proccessd[4332:]
    # X_train_temp = X_temp[:4332]
    # X_val_temp = X_temp[4332:]
    # X_train_pressure = X_pressure[:4332]
    # X_val_pressure = X_pressure[4332:]
    # X_train_humid = X_humid[:4332]
    # X_val_humid = X_humid[4332:]
    # X_train_ws = X_wind_speed[:4332]
    # X_val_ws = X_wind_speed[4332:]
    # X_train_wd = X_wind_direction[:4332]
    # X_val_wd = X_wind_direction[4332:]
    # X_train_rain = X_rain[:4332]
    # X_val_rain = X_rain[4332:]
    # X_train_pm25 = X_pm25[:4332]
    # X_val_pm25 = X_pm25[4332:]
    # X_train_pm10 = X_pm10[:4332]
    # X_val_pm10 = X_pm10[4332:]
    # X_train_no2 = X_no2[:4332]
    # X_val_no2 = X_no2[4332:]
    # X_train_o3 = X_o3[:4332]
    # X_val_o3 = X_o3[4332:]
    # X_train_so2 = X_so2[:4332]
    # X_val_so2 = X_so2[4332:]
    # X_train_co = X_co[:4332]
    # X_val_co = X_co[4332:]
    # Y_train_target = Y_target[:4332]
    # Y_val_target = Y_target[4332:]


    print('训练集大小：', X_train_dp.shape)
    print('测试集大小：', X_val_dp.shape)
    print('目标集大小：', Y_train_target.shape)
    print('目标集大小：', Y_val_target.shape)
    print('> Data Loaded. Compiling...')


    # print(type(X_train_humid))
    # print(np.isnan(X_train_dp).sum(),np.isnan(X_val_dp).sum(),'\n', np.isnan(X_train_temp).sum(), np.isnan(X_val_temp).sum(),'\n')
    # sys.exit(100)


    #4、 建立训练模型
    file_name = "wt_GRU_Keras_linear_rmsprop_" + scaler_type + "_dp(" + "_".join( str(x) for x in dropouts) + \
                ")_200_" + str(seq_len) + "_" + str(state_neurons_1) + "_" + str(state_neurons_2) + "_" + \
                str(batch_size)
    model = lstm.build_model([13, 11], [X_date_proccessd.shape[2], seq_len, state_neurons_1, state_neurons_2, output], dropouts,pre_train=file_name + ".h5")

    # print('建立的模型:',model)
    checkpoint = ModelCheckpoint(file_name+".h5", monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    earlystop = EarlyStopping(monitor='val_loss', patience=3, mode='min',min_delta=0.005)
    callbacks_list = [checkpoint,earlystop]
#
#     #5、 训练模型
    mod_hist = model.fit(
        [X_train_temp, X_train_pressure, X_train_humid, X_train_ws, X_train_wd, X_train_rain, X_train_pm25,
         X_train_pm10, X_train_no2, X_train_o3, X_train_so2, X_train_co], [Y_train_target],
        validation_data=([X_val_temp, X_val_pressure, X_val_humid, X_val_ws, X_val_wd, X_val_rain, X_val_pm25,
                          X_val_pm10, X_val_no2, X_val_o3, X_val_so2, X_val_co], [Y_val_target]),
        epochs=epochs,batch_size=batch_size, callbacks=callbacks_list, shuffle=True)

    print(mod_hist)
    print(mod_hist.history['loss'])
    print('---------------------')
    print(mod_hist.history['val_loss'])
    print('Training duration (s) : ', time.time() - global_start_time)
    print('Training duration (Hr) : ', datetime.now() - startTime)

    file = "Air_Predict_GRU_Keras_linear_rmsprop_" + scaler_type + "_dp(" + "_".join(str(x) for x in dropouts) + ")_200_" \
           + str(seq_len) + "_" + str(state_neurons_1) + "_" + str(state_neurons_2) + "_" + str(batch_size)
    print("Saving model : " + file + ".h5")
    model.save(file + ".h5")
    print("Model saved...")
    # sys.exit(1001)



#6、 利用目前多的模型进行预测
# 基于以上训练利用训练数据集、测试集、整体进行预测
if __name__=='__main__':
    #predict sequence
    # 去掉了date_proccessed
    _y_norm = lstm.predict_point_by_point_aux(model, [ X_temp, X_pressure, X_humid, X_wind_speed,
                                                      X_wind_direction, X_rain, X_pm25, X_pm10, X_no2, X_o3, X_so2, X_co])
    _y_train = lstm.predict_point_by_point_aux(model, [X_train_temp, X_train_pressure,X_train_humid,X_train_ws,
                                                       X_train_wd,X_train_rain, X_train_pm25,X_train_pm10,X_train_no2,
                                                       X_train_o3,X_train_so2,X_train_co])
    _y_val = lstm.predict_point_by_point_aux(model, [X_val_temp,X_val_pressure,X_val_humid,X_val_ws,X_val_wd,
                                                     X_val_rain,X_val_pm25,X_val_pm10,X_val_no2,X_val_o3,X_val_so2,X_val_co])
    print(_y_norm,'\n',_y_norm.shape)
    print(_y_train, '\n', _y_train.shape)
    print(_y_val, '\n', _y_val.shape)

# 将以上测测得的结果，你变换成真实值、改变数据结构
if __name__=='__main__':
    _y_norm = _y_norm.reshape(-1,1)
    _y_train = _y_train.reshape(-1,1)
    _y_val = _y_val.reshape(-1,1)

    y_norm = Y_target.reshape(-1,1)
    y_train = Y_train_target.reshape(-1,1)
    y_val = Y_val_target.reshape(-1,1)

    print("Predicted Output shape: ", np.shape(_y_norm))
    print("Original Output shape:  ", np.shape(y_norm))

    #inverse the predictions to its actual value
    print("Predicted Output sample: ")
    _y = scaler.inverse_transform(_y_norm)
    _y_tr = scaler.inverse_transform(_y_train)
    _y_va = scaler.inverse_transform(_y_val)
    for i in range(24):
        print(_y_va[i])

    #inverse the outputs to its actual value
    print("Original Output sample: ")
    y = scaler.inverse_transform(y_norm)
    y_tr = scaler.inverse_transform(y_train)
    y_va = scaler.inverse_transform(y_val)
    for i in range(24):
        print(y_va[i])

    # y = np.exp(y)
    # y_tr = np.exp(y_tr)
    # y_va = np.exp(y_va)
    #
    # _y = np.exp(_y)
    # _y_va = np.exp(_y_va)
    # _y_tr = np.exp(_y_tr)

    #predicted
    # np.ravel(_y)：将_y转换成1维序列，并替换原始数据
    _Y = pd.Series(np.ravel(_y))
    _Y_TR = pd.Series(np.ravel(_y_tr))
    _Y_VA = pd.Series(np.ravel(_y_va))


    #original
    Y =  pd.Series(np.ravel(y))
    Y_TR = pd.Series(np.ravel(y_tr))
    Y_VA = pd.Series(np.ravel(y_va))




#7、 预测并图形化展示预测效果
# 计算各数据集预测值与真实值之间的偏差——均方差，并图形化展示
if __name__=='__main__':
    print('Total RMSE :', math.sqrt(mean_squared_error(_Y, Y)))
    print('Training RMSE :', math.sqrt(mean_squared_error(_Y_TR, Y_TR)))
    print('Validation RMSE :', math.sqrt(mean_squared_error(_Y_VA, Y_VA)))
    print('Validation MAE : ',mean_absolute_error(_Y_VA,Y_VA))
    print('Validation MAEP : ',100 - np.mean(np.fabs(_Y_VA-Y_VA) / Y_VA)*100)
#     print('Test RMSE :', math.sqrt(mean_squared_error(_Y_TE, Y_TE)))

    # plt.figure(num=1,figsize=[14,7])
    # # fig1_1 = plt.add_subplot(111)
    # plot_predicted, = plt.plot(_Y_VA[100:500], label='predicted')
    # plot_train, = plt.plot(Y_VA[100:500], label='actual')
    # plt.legend(handles=[plot_predicted, plot_train])
    # plt.show()



# # summarize history for loss
print(mod_hist.history['loss'])
print(mod_hist.history['val_loss'])
# plt.plot(mod_hist.history['loss'],lable='loss')
# plt.plot(mod_hist.history['val_loss'],lable='val_loss')
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
# print('--------------------到非主函数这里\n')

# # 载入训练好数据模型参数，重新初始化LSTM模型，只用测试集进行验证；图形化展示
if __name__=='__main__':
    #recreate model
    file_name = "wt_GRU_Keras_linear_rmsprop_" + scaler_type + "_dp(" + "_".join(str(x) for x in dropouts) + \
                ")_200_" + str(seq_len) + "_" + str(state_neurons_1) + "_" + str(state_neurons_2) + "_" + \
                str(batch_size)
    model = lstm.build_model([13, 11], [X_date_proccessd.shape[2], seq_len, state_neurons_1, state_neurons_2, output], dropouts, pre_train=file_name+".h5")
    print("Created model and loaded weights from "+file_name+".h5")

    # estimate accuracy on whole dataset using loaded weights
    _y_val = lstm.predict_point_by_point_aux(model, [X_val_temp,X_val_pressure,X_val_humid,X_val_ws,X_val_wd,
                                                     X_val_rain,X_val_pm25,X_val_pm10,X_val_no2,X_val_o3,X_val_so2,X_val_co])
    _y_val = _y_val.reshape(-1,1)
    y_val = Y_val_target.reshape(-1,1)

    #inverse the predictions to its actual value
    print("Predicted Output sample: ")
    _y_va = scaler.inverse_transform(_y_val)
    for i in range(24):
        print(_y_va[i])

    #inverse the outputs to its actual value
    print("Original Output sample: ")
    y_va = scaler.inverse_transform(y_val)
    for i in range(24):
        print(y_va[i])

    # y_va = np.exp(y_va)
    # _y_va = np.exp(_y_va)

    #predicted
    _Y_VA = pd.Series(np.ravel(_y_va))
    #original
    Y_VA = pd.Series(np.ravel(y_va))

    print('Validation RMSE :', math.sqrt(mean_squared_error(_Y_VA, Y_VA)))
    print('Validation MAE : ',mean_absolute_error(_Y_VA,Y_VA))
    print('Validation MAEP : ',100 - np.mean(np.fabs(_Y_VA-Y_VA) / Y_VA)*100)


    # plt.figure(num=2,figsize=[14,7])
    # # fig2_2 = fig2.add_subplot(111)
    #
    # plot_predicted, = plt.plot(_Y_VA[:], label='predicted')
    # plot_train, = plt.plot(Y_VA[:], label='actual')
    # plt.legend(handles=[plot_predicted, plot_train],loc='upper left')
    # plt.show()




def squash(x):
    return float(1.0 * x/(x+10))

