import numpy as np
import pandas as pd
import sqlalchemy
import time
from sklearn import preprocessing

from aqf.connection import DBConnection


class Process(object):
    def process(self):
        pass

    @staticmethod
    def dataFormat(data, data_range):
        range_min = data_range["FACTOR_MIN_VALUE"]
        range_max = data_range["FACTOR_MAX_VALUE"]
        return (data - range_min.T) / (range_max - range_min).T

    @staticmethod
    def dataFormatOne(data, data_range):
        data_range.index = ['0']
        range_min = data_range.ix[0, "FACTOR_MIN_VALUE"]
        range_max = data_range.ix[0, "FACTOR_MAX_VALUE"]
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
        range_min = data_range.ix[0, "FACTOR_MIN_VALUE"]
        range_max = data_range.ix[0, "FACTOR_MAX_VALUE"]
        return value * (range_max - range_min) + range_min

    @staticmethod
    def process1(weights, weather):
        return 1 / (1 + np.exp(-(weights[2] * weather) - 0.5))

    @staticmethod
    def process2(weights, result):
        sum1 = weights[2] * result
        sum2 = sum1.sum(axis=1)
        return 1 / (1 + np.exp(-sum2 - 0.5))


class Learning(object):
    @staticmethod
    def getCorrectionK(expect, r2, r1):
        return ((expect['FACTOR_VALUE'] - r2) * r2 * (1 - r1).T).T

    @staticmethod
    def getCorrectionJ(correction, weights, result):
        correction_weight = correction * weights[2]
        correction_weight_sum = correction_weight.sum(axis=1)
        print("求和：\n", correction_weight_sum)
        print("------------------------------------------")
        return ((result * (1 - result)).T * correction_weight_sum).T

    @staticmethod
    def getWeightK(weights, correction, result):
        return weights[2] + 0.25 * correction * result

    @staticmethod
    def getThresholdVectorK(correction):
        return 0.5 + 0.002 * correction

    @staticmethod
    def getWeightJ(weights, correction, weather):
        return weights[2] + 0.25 * correction * weather

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


class Aqf(object):
    @staticmethod
    def main(times):
        # oraengine = sqlalchemy.create_engine('oracle://itp:12369@172.30.202.209:1521/WXHBZCPT1')
        # dbconnection = DBConnection("172.30.202.209", "1521", "WXHBZCPT1", "itp", "12369")
        oraengine = sqlalchemy.create_engine('oracle://wsn1:wsn123456@10.146.51.223:1521/orcl')
        dbconnection = DBConnection("10.146.51.223", "1521", "orcl", "wsn1", "wsn123456")
        conn = dbconnection.getConnection()
        dbconnection.connection = conn
        sql_factors = "SELECT pointid,factor_code,create_date,date_processed,temperature,pressure,humidity,wind_speed,wind_direction,rainfall,cloudage,solar_radiation_intensity,one_hour_before,two_hour_before,three_hour_before,one_day_before,factor_value FROM V_AQF_FORMAT_DATA where create_date >= to_date('2017-03-08','yyyy-mm-dd') and create_date < to_date('2017-03-10','yyyy-mm-dd') order by create_date"
        # result_factors = dbconnection.query(sql_factors)
        result_factors = pd.read_sql(sql_factors, conn)

        sql_weights_1 = 'select "factor_name", "factor_chinese_name","factor_weight_original_value" from AQF_METEOROLOGICAL_FACTORS_1'
        result_weights_1 = dbconnection.query(sql_weights_1)

        sql_weights_2 = 'select "factor_name", "factor_chinese_name","factor_weight_original_value" from AQF_METEOROLOGICAL_FACTORS_2'
        result_weights_2 = dbconnection.query(sql_weights_2)

        sql_range = 'SELECT factor_name, factor_min_value, factor_max_value FROM aqf_factors_range'
        # result_range = dbconnection.query(sql_range)
        # min_max_range = pd.DataFrame(result_range, columns=['factor_name', 'factor_min_value', 'factor_max_value'])
        result_range = pd.read_sql(sql_range, conn)
        result_range_weather = result_range.ix[:8, 1:]
        result_range_monitor = result_range.ix[9:, :]

        factors_name = ["DATE_PROCESSED", "TEMPERATURE", "PRESSURE", "HUMIDITY", "WIND_SPEED", "WIND_DIRECTION",
                        "RAINFALL", "CLOUDAGE", "SOLAR_RADIATION_INTENSITY", "ONE_HOUR_BEFORE", "TWO_HOUR_BEFORE",
                        "THREE_HOUR_BEFORE", "ONE_DAY_BEFORE"]

        dbconnection.closeConnection()

        # result_pd = pd.DataFrame(result_factors)
        # data_info = result_pd.ix[:, :2]
        # data_factors = result_pd.ix[:, 3:15]
        # expected_value = result_pd.ix[:, 16:]

        aqi_factors = ['a34004', 'a34002', 'a21026', 'a21004', 'a05024', 'a21005']
        # aqi_factors = ['a34004']

        for factor in aqi_factors:
            data_PM25 = result_factors[result_factors["FACTOR_CODE"] == factor]
            data_info = data_PM25.ix[:, :3]
            data_factors = data_PM25.ix[:, 3:16]
            expected_value = data_PM25.ix[:, 16:]

            # 处理缺失数据，若值为NAN，全部补0
            data_factors = data_factors.fillna(0)
            expected_value = expected_value.fillna(0)

            print("初始数据：\n", data_factors)
            print("------------------------------------------")

            monitor_range = result_range_monitor[result_range_monitor["FACTOR_NAME"] == factor].ix[:, 1:]
            data_range = pd.concat([result_range_weather, monitor_range, monitor_range, monitor_range, monitor_range])
            data_range.index = factors_name

            print("最大值和最小值：\n", data_range)
            print("------------------------------------------")

            weights_1 = pd.DataFrame(result_weights_1)
            weights_1.index = factors_name
            print("初始权重第一层：\n", weights_1)
            print("------------------------------------------")

            weights_2 = pd.DataFrame(result_weights_2)
            weights_2.index = factors_name
            print("初始权重第二层：\n", weights_2)
            print("------------------------------------------")

            weather_standard = pd.DataFrame(Process.dataFormat(data_factors, data_range))
            print("归一化后的数据：\n", weather_standard)
            print("------------------------------------------")

            result1 = Process.process1(weights_1, weather_standard)
            print("第一次计算后的数据：\n", result1)
            print("------------------------------------------")

            predict = Process.process2(weights_2, result1)
            print("第二次计算后的数据：\n", predict)
            print("------------------------------------------")

            predict_AntiNormalization = Process.MaxMinAntiNormalization(predict, monitor_range)
            print("实际预测数据：\n", predict_AntiNormalization)
            print("------------------------------------------")

            # accuracy = 1 - abs(predict_AntiNormalization - expected_value).T / expected_value
            accuracy = 1 - abs((predict_AntiNormalization - expected_value.T).T) / expected_value
            accuracy.rename(columns={'FACTOR_VALUE': 'ACCURACY'}, inplace=True)
            print("预测结果准确率：\n", accuracy)
            print("------------------------------------------")

            pre = pd.concat([data_info, expected_value, predict_AntiNormalization, accuracy], axis=1)
            pre.insert(0, "RESULT_ID", times)
            pre.rename(columns={0: 'CAL_VALUE'}, inplace=True)
            print("存入数据库的数据：\n", pre)
            print("------------------------------------------")

            pre.to_sql("aqf_result", oraengine, if_exists='append', index=False)

            # 计算结果插入数据库
            # cursor = dbconnection.getCorsor()
            # dbconnection.cursor = cursor
            # insert_values = [(i + 1, predict_AntiNormalization[i]) for i in range(len(predict_AntiNormalization))]
            # sql = "insert into AQF_RESULT(result_id, result) VALUES (:1,:2)"
            # result_weights_2 = dbconnection.insert(sql)
            # conn.commit()

            print("期望值：\n", expected_value)
            print("------------------------------------------")
            expectedValue_scaler = Process.dataFormatOne(expected_value, monitor_range)
            print("期望值归一化后：\n", expectedValue_scaler)
            print("------------------------------------------")

            correctionK = Learning.getCorrectionK(expectedValue_scaler, predict, result1)
            print("输出层和隐藏层之间的连接权值修正量：\n", correctionK)
            print("------------------------------------------")

            correctionJ = Learning.getCorrectionJ(correctionK, weights_2, result1)
            print("隐藏层和输入层之间的连接权值修正量：\n", correctionJ)
            print("------------------------------------------")

            print("输出层和隐藏层之间的连接权值矩阵和阈值向量：\n")
            weightK = Learning.getWeightK(weights_2, correctionK, result1)
            print("输出层和隐藏层之间的连接权值：\n", weightK, "\n")
            thresholdVectorK = Learning.getThresholdVectorK(correctionK)
            print("输出层和隐藏层之间的阈值向量：\n", thresholdVectorK)
            print("------------------------------------------")

            print("隐藏层和输入层之间的连接权值矩阵和阈值向量：\n")
            weightJ = Learning.getWeightJ(weights_1, correctionJ, weather_standard)
            print("隐藏层和输入层之间的连接权值：\n", weightJ, "\n")
            thresholdVectorJ = Learning.getThresholdVectorJ(correctionJ)
            print("隐藏层和输入层之间的阈值向量：\n", thresholdVectorJ)
            print("------------------------------------------")

            # 修改列名 a.rename(columns={'A': 'a', 'B': 'b', 'C': 'c'}, inplace = True)
            data_info.rename(columns={0: 'pointid', 1: 'factor_code', 2: 'create_date'}, inplace=True)
            # weightK.columns = ['WEIGHT_DATE_PROCESSED', 'WEIGHT_TEMPERATURE', 'WEIGHT_PRESSURE', 'WEIGHT_HUMIDITY',
            #                    'WEIGHT_WIND_SPEED', 'WEIGHT_WIND_DIRECTION', 'WEIGHT_RAINFALL', 'WEIGHT_CLOUDAGE',
            #                    'WEIGHT_SOLAR_RADIATION', 'WEIGHT_ONE_HOUR_BEFORE', 'WEIGHT_TWO_HOUR_BEFORE',
            #                    'WEIGHT_THREE_HOUR_BEFORE', 'WEIGHT_ONE_DAY_BEFORE']
            # thresholdVectorK.columns = ['OFFSET_DATE_PROCESSED', 'OFFSET_TEMPERATURE', 'OFFSET_PRESSURE', 'OFFSET_HUMIDITY',
            #                             'OFFSET_WIND_SPEED', 'OFFSET_WIND_DIRECTION', 'OFFSET_RAINFALL', 'OFFSET_CLOUDAGE',
            #                             'OFFSET_SOLAR_RADIATION', 'OFFSET_ONE_HOUR_BEFORE', 'OFFSET_TWO_HOUR_BEFORE',
            #                             'OFFSET_THREE_HOUR_BEFORE', 'OFFSET_ONE_DAY_BEFORE']
            # correctionK.columns = ['SLOPE_DATE_PROCESSED', 'SLOPE_TEMPERATURE', 'SLOPE_PRESSURE', 'SLOPE_HUMIDITY',
            #                        'SLOPE_WIND_SPEED', 'SLOPE_WIND_DIRECTION', 'SLOPE_RAINFALL', 'SLOPE_CLOUDAGE',
            #                        'SLOPE_SOLAR_RADIATION', 'SLOPE_ONE_HOUR_BEFORE', 'SLOPE_TWO_HOUR_BEFORE',
            #                        'SLOPE_THREE_HOUR_BEFORE', 'SLOPE_ONE_DAY_BEFORE']
            Learning.weight_col_format(weightK)
            Learning.thresholdVector_col_format(thresholdVectorK)
            Learning.correction_col_format(correctionK)
            paramsK = pd.concat([data_info, weightK, thresholdVectorK, correctionK], axis=1)
            paramsK.insert(0, "TRAINING_ID", times)
            paramsK.insert(1, "TRAINING_TIME", str(time.strftime('%Y-%m-%d %H:%M:%S')))
            paramsK.insert(5, "CALCULATION_LEVEL", 2)
            # print(paramsK)
            # print('**********\n', paramsK.iloc[0])
            # print('**********\n', paramsK.to_dict(orient='record'))

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


            # training_sql = "insert into AQF_TRAINING_DATA(TRAINING_ID, TRAINING_TIME, POINTID, FACTOR_CODE, CALCULATION_LEVEL, " \
            #                "WEIGHT_DATE_PROCESSED, WEIGHT_TEMPERATURE, WEIGHT_PRESSURE, WEIGHT_HUMIDITY, WEIGHT_WIND_SPEED, " \
            #                "WEIGHT_WIND_DIRECTION,WEIGHT_RAINFALL,WEIGHT_CLOUDAGE,WEIGHT_SOLAR_RADIATION,WEIGHT_ONE_HOUR_BEFORE," \
            #                "WEIGHT_TWO_HOUR_BEFORE,WEIGHT_THREE_HOUR_BEFORE,WEIGHT_ONE_DAY_BEFORE," \
            #                "OFFSET_DATE_PROCESSED,OFFSET_TEMPERATURE,OFFSET_PRESSURE,OFFSET_HUMIDITY,OFFSET_WIND_SPEED," \
            #                "OFFSET_WIND_DIRECTION,OFFSET_RAINFALL,OFFSET_CLOUDAGE,OFFSET_SOLAR_RADIATION,OFFSET_ONE_HOUR_BEFORE," \
            #                "OFFSET_TWO_HOUR_BEFORE,OFFSET_THREE_HOUR_BEFORE,OFFSET_ONE_DAY_BEFORE," \
            #                "SLOPE_DATE_PROCESSED,SLOPE_TEMPERATURE,SLOPE_PRESSURE,SLOPE_HUMIDITY,SLOPE_WIND_SPEED," \
            #                "SLOPE_WIND_DIRECTION,SLOPE_RAINFALL,SLOPE_CLOUDAGE,SLOPE_SOLAR_RADIATION,SLOPE_ONE_HOUR_BEFORE," \
            #                "SLOPE_TWO_HOUR_BEFORE,SLOPE_THREE_HOUR_BEFORE,SLOPE_ONE_DAY_BEFORE) " \
            #                "VALUES (" + times + ",sysdate,:pointid,:factor_code,'1'," \
            #                                     ":weight_0,:weight_1,:weight_2,:weight_3,:weight_4," \
            #                                     ":weight_5,:weight_6,:weight_7,:weight_8,:weight_9,:weight_10,:weight_11,:weight_12," \
            #                                     ":thresholdVector_0,:thresholdVector_1,:thresholdVector_2,:thresholdVector_3," \
            #                                     ":thresholdVector_4,:thresholdVector_5,:thresholdVector_6,:thresholdVector_7," \
            #                                     ":thresholdVector_8,:thresholdVector_9,:thresholdVector_10,:thresholdVector_11," \
            #                                     ":thresholdVector_12," \
            #                                     ":correction_0,:correction_1,:correction_2,:correction_3,:correction_4,:correction_5," \
            #                                     ":correction_6,:correction_7,:correction_8,:correction_9,:correction_10," \
            #                                     ":correction_11,:correction_12)"
            # print(training_sql)
            # for trainingK in paramsK.to_dict(orient='record'):
            #     dbconnection.DDLDB_P(training_sql, trainingK)

            # dbconnection.closeConnection()


if __name__ == '__main__':
    Aqf.main()
