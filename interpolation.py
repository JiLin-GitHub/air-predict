# -*- coding: utf-8 -*-
"""
插值。
"""
from datetime import datetime

import numpy as np
import sqlalchemy
from scipy import interpolate
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import matplotlib as mpl


# x = [120.269, 120.275, 120.354, 120.354, 120.288]
# y = [31.4867, 31.6219, 31.5848, 31.5475, 31.6842]
#
# fvals = [100, 50, 88, 74, 63]
#
# x1 = np.linspace(1, 200, 5)
# print(x1)
# x_new = np.linspace(1, 200, 100)
# print(x_new)
#
# y1 = np.linspace(1, 50, 5)
# y_new = np.linspace(1, 50, 100)
#
# tck = interpolate.splrep(x1, fvals)
# y_bspline = interpolate.splev(x_new, tck)
# print(y_bspline)

class interpolation(object):
    @staticmethod
    def my_interpolation(time, conn):
        sql_interpolation = "SELECT * FROM aqf_result A, AQF_STATION_COORDINATE b WHERE A.POINTID = b.STATION_CODE " \
                            "AND A.CREATE_DATE = TO_DATE('" \
                            + time.strftime("%Y-%m-%d %H:%M:%S") \
                            + "', 'yyyy-mm-dd hh24:mi:ss') " \
                            "and a.FACTOR_CODE = 'a05024' ORDER BY A.CREATE_DATE, A.FACTOR_CODE, TO_NUMBER(A.POINTID)"
        # TO_NUMBER()函数的作用是将字符串类型转换为整数类型，作用和TO_CHAR()函数相反

        # 读取数据库中关于插值部分的数据
        data_origin = pd.read_sql(sql_interpolation, conn)
        print("初始数据：\n", data_origin)
        print("------------------------------------------")

        # 运用numpy.array创建series数组，数组元素是data_origin源数据cal_value标签列series数值，并分别转换为float类型
        # 同上，取了station_lon：站经度
        # 同上，取了station_lat：站纬度
        # 三者的关系是实际测得的现实点位的经纬度和对应经纬度上的值
        cal_value = np.array([float(i) for i in data_origin['cal_value']])
        longitudu = np.array([float(i) for i in data_origin['station_lon']])
        latitude = np.array([float(i) for i in data_origin['station_lat']])

        #根据取得的经纬度、预测值信息，进行插值计算处理
        # 经过插值处理后，变成了一个x_value + y_value + z_value列镖标签分别为x,y,z的dataformat
        # 而每列上同行上的三个元素xi、yi、zi分别代表了经度、纬度、插值计算后该经纬度点对应的值
        data_interpolation = interpolation.deal_interpolation(interpolation, longitudu, latitude, cal_value)


        data_interpolation.columns = ['station_lon', 'station_lat', 'interp']
        data_interpolation.insert(0, "factor_code", data_origin['factor_code'][0])
        data_interpolation.insert(1, "create_date", data_origin['create_date'][0])
        print("插值后的数据：\n", data_interpolation)
        print("------------------------------------------")

        # 处理后的插值dataformat数据结构存到服务器，插值数据库列表名称：aqf_result_interpolation
        data_interpolation.to_sql("aqf_result_interpolation", oraengine, if_exists='append', index=False)


    # 插值处理函数
    # 返回插值处理结束后合成的三合一dataformat数据类型，三列分别代表了插值处理后（10*10）经度、纬度、插值效果值
    def deal_interpolation(self, x, y, z):
        nx, ny = 10, 10
        # x = np.array([120.269, 120.275, 120.354, 120.354, 120.288])
        # y = np.array([31.4867, 31.6219, 31.5848, 31.5475, 31.6842])
        # z = np.array([100, 50, 88, 74, 63])
        # print(x, y, z)

        # 在经度上(x)，最大值和最小值之间插值nx个,返回一维数组
        # 在纬度上(y)，最大值和最小值之间插值ny个
        xi = np.linspace(x.min(), x.max(), nx)
        yi = np.linspace(y.min(), y.max(), ny)

        #xi,yi分别以行扩展len(yi),列扩展len(xi)个；形成两个二维数组
        #[ xi,  模拟了经度，列元素相同    [ [yi[0],yi[0],yi[0],····,yi[0]], 模拟了纬度，横向的行元素相同
        #  xi,                           [yi[1],yi[1],yi[1],····,yi[1]],
        #  ·····                         ·····
        #  xi  共len(yi)行               [yi[leny-1],yi[leny-1],yi[leny-1],····,yi[leny-1]]    共len(xi)列
        # ]                              ]
        # 对于这里，都是10*10的二维数组，结果是这两个二维数组对应位置上的元素分别代表了一个点的横纵坐标
        xi, yi = np.meshgrid(xi, yi)

        # 把经纬度混合数组的内容分别“拉平”，便于根据索引查找某一点上的经纬度
        xi, yi = xi.flatten(), yi.flatten()
        # print(pd.DataFrame(xi))
        # print(pd.DataFrame(yi))

        # 运用Rbf插值算法，依据实际测得的经纬度及对应位置上的值x、y、z，在xi、yi小分辨率格点上进行插值处理
        # 返回处理结束后插值格点数组，10*10
        grid2 = self.scipy_idw(x, y, z, xi, yi)
        # print(pd.DataFrame(grid2))
        # grid2 = grid2.reshape((ny, nx))

        # self.plot(x, y, z, grid2)
        # plt.title("Scipy's Rbf with function=linear")
        # plt.show()

        # 横向三合一，转换为dataformat数据类型
        x_value = pd.DataFrame(xi, columns=['x'])
        y_value = pd.DataFrame(yi, columns=['y'])
        z_value = pd.DataFrame(grid2, columns=['z'])
        return pd.concat([x_value, y_value, z_value], axis=1)

    def scipy_idw(x, y, z, xi, yi):
        interp = interpolate.Rbf(x, y, z, function='regular')
        return interp(xi, yi)

    def plot(x, y, z, grid):
        plt.figure()
        plt.imshow(grid, extent=(x.min(), x.max(), y.max(), y.min()))
        plt.hold(True)
        plt.scatter(x, y, c=z)
        plt.colorbar()


if __name__ == '__main__':
    # oraengine = sqlalchemy.create_engine('oracle://itp:12369@172.30.202.209:1521/WXHBZCPT1')
    oraengine = sqlalchemy.create_engine('oracle://wsn1:wsn123456@10.146.51.223:1521/orcl')

    for i in np.arange(24):
        stime = datetime(2017, 3, 8, i, 0)
        # print(stime)
        interpolation.my_interpolation(stime, oraengine)
