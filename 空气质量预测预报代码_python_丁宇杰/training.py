import numpy as np
import pandas as pd
import sqlalchemy
import time
from sklearn import preprocessing

from aqf.connection import DBConnection
from aqf.process import Aqf


class Process(object):
    def process(self):
        pass

    @staticmethod
    def dataFormat(data, data_range):
        range_min = data_range["factor_min_value"]
        range_max = data_range["factor_max_value"]
        return (data - range_min.T) / (range_max - range_min).T
        #'.T'的作用是使上面的range_min行标签和data列标签相同位置上的元素相加；后面的是range_min
        # 和range_max相同列标签位置上的元素相减后，使用'.T'来转置，能和前面的相同标签名的位置元素做除法

    @staticmethod
    def dataFormatOne(data, data_range):
        data_range.index = ['0']
        range_min = data_range.ix[0, "factor_min_value"]
        range_max = data_range.ix[0, "factor_max_value"]
        return (data - range_min) / (range_max - range_min)

    @staticmethod
    def dataFormatBySklearn(data):
        min_max_scaler = preprocessing.MinMaxScaler()
        weather_minmax = min_max_scaler.fit_transform(data)
        return weather_minmax

    @staticmethod
    def getMaxMin(data):
        data_max = data.max(axis=0)
        data_min = data.min(axis=0)
        return pd.concat([data_max, data_min], axis=1)

    @staticmethod
    def MaxMinAntiNormalization(value, data_range):
        data_range.index = ['0']
        range_min = data_range.ix[0, "factor_min_value"]
        range_max = data_range.ix[0, "factor_max_value"]
        return value * (range_max - range_min) + range_min

    @staticmethod
    def process1(weights, weather, offset):
        return 1 / (1 + np.exp(-(weights * weather) - offset))

    @staticmethod
    def process2(weights, result, offset):
        sum1 = weights * result
        sum2 = sum1.sum(axis=1)
        return 1 / (1 + np.exp(-sum2 - offset / 13))


class Learning(object):
    @staticmethod
    def getCorrectionK(expect, r2, r1):
        return ((expect['factor_value'] - r2) * r2 * (1 - r1).T).T

    @staticmethod
    def getCorrectionJ(correction, weights, result):
        correction_weight = correction * weights
        correction_weight_sum = correction_weight.sum(axis=1)
        print("求和：\n", correction_weight_sum)
        print("------------------------------------------")
        return ((result * (1 - result)).T * correction_weight_sum).T

    @staticmethod
    def getWeightK(weights, correction, result):
        return weights + 0.25 * correction * result

    @staticmethod
    def getThresholdVectorK(correction):
        return 0.5 + 0.002 * correction

    @staticmethod
    def getWeightJ(weights, correction, weather):
        return weights + 0.25 * correction * weather

    @staticmethod
    def getThresholdVectorJ(correction):
        return 0.5 + 0.002 * correction

    @staticmethod
    def weight_col_format(weight):
        weight.columns = ['WEIGHT_DATE_PROCESSED', 'WEIGHT_TEMPERATURE', 'WEIGHT_PRESSURE', 'WEIGHT_HUMIDITY',
                          'WEIGHT_WIND_SPEED', 'WEIGHT_WIND_DIRECTION', 'WEIGHT_RAINFALL', 'WEIGHT_CLOUDAGE',
                          'WEIGHT_SOLAR_RADIATION', 'WEIGHT_ONE_HOUR_BEFORE', 'WEIGHT_TWO_HOUR_BEFORE',
                          'WEIGHT_THREE_HOUR_BEFORE', 'WEIGHT_ONE_DAY_BEFORE']

    @staticmethod
    def thresholdVector_col_format(thresholdVector):
        thresholdVector.columns = ['OFFSET_DATE_PROCESSED', 'OFFSET_TEMPERATURE', 'OFFSET_PRESSURE', 'OFFSET_HUMIDITY',
                                   'OFFSET_WIND_SPEED', 'OFFSET_WIND_DIRECTION', 'OFFSET_RAINFALL', 'OFFSET_CLOUDAGE',
                                   'OFFSET_SOLAR_RADIATION', 'OFFSET_ONE_HOUR_BEFORE', 'OFFSET_TWO_HOUR_BEFORE',
                                   'OFFSET_THREE_HOUR_BEFORE', 'OFFSET_ONE_DAY_BEFORE']

    @staticmethod
    def correction_col_format(correction):
        correction.columns = ['SLOPE_DATE_PROCESSED', 'SLOPE_TEMPERATURE', 'SLOPE_PRESSURE', 'SLOPE_HUMIDITY',
                              'SLOPE_WIND_SPEED', 'SLOPE_WIND_DIRECTION', 'SLOPE_RAINFALL', 'SLOPE_CLOUDAGE',
                              'SLOPE_SOLAR_RADIATION', 'SLOPE_ONE_HOUR_BEFORE', 'SLOPE_TWO_HOUR_BEFORE',
                              'SLOPE_THREE_HOUR_BEFORE', 'SLOPE_ONE_DAY_BEFORE']


class Training(object):
    @staticmethod
    def main(times):
        # oraengine = sqlalchemy.create_engine('oracle://itp:12369@172.30.202.209:1521/WXHBZCPT1')
        oraengine = sqlalchemy.create_engine('oracle://wsn1:wsn123456@10.146.51.223:1521/orcl')

        sql_factors = "SELECT POINTID,FACTOR_CODE,CREATE_DATE,DATE_PROCESSED,TEMPERATURE,PRESSURE,HUMIDITY,WIND_SPEED,WIND_DIRECTION,RAINFALL,CLOUDAGE,SOLAR_RADIATION_INTENSITY,ONE_HOUR_BEFORE,TWO_HOUR_BEFORE,THREE_HOUR_BEFORE,ONE_DAY_BEFORE,FACTOR_VALUE FROM V_AQF_FORMAT_DATA WHERE CREATE_DATE >= TO_DATE('2017-03-08','YYYY-MM-DD') AND CREATE_DATE < TO_DATE('2017-03-10','YYYY-MM-DD') order by create_date"
        result_factors = pd.read_sql(sql_factors, oraengine)
        # print(result_factors)

        sql_training_1 = "SELECT * FROM aqf_training_data a WHERE a.calculation_level = '1' AND a.training_id = (SELECT MAX(training_id) FROM aqf_training_data) AND a.create_date >= to_date('2017-03-08', 'YYYY-MM-DD') AND a.create_date < to_date('2017-03-10', 'YYYY-MM-DD') ORDER BY a.create_date"
        result_training_1 = pd.read_sql(sql_training_1, oraengine)

        sql_training_2 = "SELECT * FROM aqf_training_data a WHERE a.calculation_level = '2' AND a.training_id = (SELECT MAX(training_id) FROM aqf_training_data) AND a.create_date >= to_date('2017-03-08', 'YYYY-MM-DD') AND a.create_date < to_date('2017-03-10', 'YYYY-MM-DD') ORDER BY a.create_date"
        result_training_2 = pd.read_sql(sql_training_2, oraengine)

        sql_range = 'SELECT FACTOR_NAME, FACTOR_MIN_VALUE, FACTOR_MAX_VALUE FROM AQF_FACTORS_RANGE'
        result_range = pd.read_sql(sql_range, oraengine)
        result_range_weather = result_range.ix[:8, 1:]
        result_range_monitor = result_range.ix[9:, :]

        factors_name = ["DATE_PROCESSED", "TEMPERATURE", "PRESSURE", "HUMIDITY", "WIND_SPEED", "WIND_DIRECTION",
                        "RAINFALL", "CLOUDAGE", "SOLAR_RADIATION_INTENSITY", "ONE_HOUR_BEFORE", "TWO_HOUR_BEFORE",
                        "THREE_HOUR_BEFORE", "ONE_DAY_BEFORE"]


        # aqi_factors = ['a34004', 'a34002', 'a21026', 'a21004', 'a05024', 'a21005']
        aqi_factors = ['a34004']

        for factor in aqi_factors:
            data_PM25 = result_factors[result_factors["factor_code"] == factor]
            data_info = data_PM25.ix[:, :3]
            data_factors = data_PM25.ix[:, 3:16]
            expected_value = data_PM25.ix[:, 16:]

            data_training_1 = result_training_1[result_training_1["factor_code"] == factor]
            data_training_1.index = data_PM25.index
            data_weights_1 = data_training_1.ix[:, 6:19]
            data_offset_1 = data_training_1.ix[:, 19:32]

            data_training_2 = result_training_2[result_training_2["factor_code"] == factor]
            data_training_2.index = data_PM25.index
            data_weights_2 = data_training_2.ix[:, 6:19]
            data_offset_2 = data_training_2.ix[:, 19:32]

            # 处理缺失数据，若值为NAN，全部补0
            data_factors = data_factors.fillna(0)
            expected_value = expected_value.fillna(0)

            print("初始数据：\n", data_factors)
            print("------------------------------------------")

            monitor_range = result_range_monitor[result_range_monitor["factor_name"] == factor].ix[:, 1:]
            data_range = pd.concat([result_range_weather, monitor_range, monitor_range, monitor_range, monitor_range])
            data_range.index = factors_name

            print("最大值和最小值：\n", data_range)
            print("------------------------------------------")

            data_weights_1.columns = factors_name
            print("权重第一层：\n", data_weights_1)
            print("------------------------------------------")

            data_weights_2.columns = factors_name
            print("权重第二层：\n", data_weights_2)
            print("------------------------------------------")

            data_factors.columns = factors_name
            weather_standard = pd.DataFrame(Process.dataFormat(data_factors, data_range))
            print("归一化后的数据：\n", weather_standard)
            print("------------------------------------------")

            # data_weights_1.index = weather_standard.index
            # data_offset_1.index = weather_standard.index
            data_offset_1.columns = factors_name
            result1 = Process.process1(data_weights_1, weather_standard, data_offset_1)
            print("第一次计算后的数据：\n", result1)
            print("------------------------------------------")

            # data_weights_2.index = weather_standard.index
            # data_offset_2.index = weather_standard.index
            # data_offset_2.columns = factors_name
            data_offset_2_sum = data_offset_2.sum(axis=1)
            predict = Process.process2(data_weights_2, result1, data_offset_2_sum)
            print("第二次计算后的数据：\n", predict)
            print("------------------------------------------")

            predict_AntiNormalization = Process.MaxMinAntiNormalization(predict, monitor_range)
            print("实际预测数据：\n", predict_AntiNormalization)
            print("------------------------------------------")

            accuracy = 1 - abs((predict_AntiNormalization - expected_value.T).T) / expected_value
            accuracy.rename(columns={'factor_value': 'ACCURACY'}, inplace=True)
            print("预测结果准确率：\n", accuracy)
            print("------------------------------------------")

            pre = pd.concat([data_info, expected_value, predict_AntiNormalization, accuracy], axis=1)
            pre.insert(0, "RESULT_ID", times)
            pre.rename(columns={0: 'CAL_VALUE'}, inplace=True)
            print("存入数据库的数据：\n", pre)
            print("------------------------------------------")

            # 计算结果插入数据库
            pre.to_sql("aqf_result", oraengine, if_exists='append', index=False)

            print("期望值：\n", expected_value)
            print("------------------------------------------")
            expectedValue_scaler = Process.dataFormatOne(expected_value, monitor_range)
            print("期望值归一化后：\n", expectedValue_scaler)
            print("------------------------------------------")

            correctionK = Learning.getCorrectionK(expectedValue_scaler, predict, result1)
            print("输出层和隐藏层之间的连接权值修正量：\n", correctionK)
            print("------------------------------------------")

            correctionJ = Learning.getCorrectionJ(correctionK, data_weights_2, result1)
            print("隐藏层和输入层之间的连接权值修正量：\n", correctionJ)
            print("------------------------------------------")

            print("输出层和隐藏层之间的连接权值矩阵和阈值向量：\n")
            weightK = Learning.getWeightK(data_weights_2, correctionK, result1)
            print("输出层和隐藏层之间的连接权值：\n", weightK, "\n")
            thresholdVectorK = Learning.getThresholdVectorK(correctionK)
            print("输出层和隐藏层之间的阈值向量：\n", thresholdVectorK)
            print("------------------------------------------")

            print("隐藏层和输入层之间的连接权值矩阵和阈值向量：\n")
            weightJ = Learning.getWeightJ(data_weights_1, correctionJ, weather_standard)
            print("隐藏层和输入层之间的连接权值：\n", weightJ, "\n")
            thresholdVectorJ = Learning.getThresholdVectorJ(correctionJ)
            print("隐藏层和输入层之间的阈值向量：\n", thresholdVectorJ)
            print("------------------------------------------")

            # 修改列名 a.rename(columns={'A': 'a', 'B': 'b', 'C': 'c'}, inplace = True)
            data_info.rename(columns={0: 'pointid', 1: 'factor_code', 2: 'create_date'}, inplace=True)
            Learning.weight_col_format(weightK)
            Learning.thresholdVector_col_format(thresholdVectorK)
            Learning.correction_col_format(correctionK)
            paramsK = pd.concat([data_info, weightK, thresholdVectorK, correctionK], axis=1)
            paramsK.insert(0, "TRAINING_ID", times)
            paramsK.insert(1, "TRAINING_TIME", str(time.strftime('%Y-%m-%d %H:%M:%S')))
            paramsK.insert(4, "CALCULATION_LEVEL", 2)

            # 调整后的权重、偏移量、斜率插入数据库
            paramsK.to_sql("aqf_training_data", oraengine, if_exists='append', index=False)


            Learning.weight_col_format(weightJ)
            Learning.thresholdVector_col_format(thresholdVectorJ)
            Learning.correction_col_format(correctionJ)
            paramsJ = pd.concat([data_info, weightJ, thresholdVectorJ, correctionJ], axis=1)
            paramsJ.insert(0, "TRAINING_ID", times)
            paramsJ.insert(1, "TRAINING_TIME", str(time.strftime('%Y-%m-%d %H:%M:%S')))
            paramsJ.insert(5, "CALCULATION_LEVEL", 1)

            # 调整后的权重、偏移量、斜率插入数据库
            paramsJ.to_sql("aqf_training_data", oraengine, if_exists='append', index=False)

if __name__ == '__main__':
    # for times in np.arange(3):
    #     if times == 0:
    #         Aqf.main(times+1)
    #     else:
    #         Training.main(times+1)
    Training.main(5)
