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
        data_origin = pd.read_sql(sql_interpolation, conn)
        print("初始数据：\n", data_origin)
        print("------------------------------------------")

        cal_value = np.array([float(i) for i in data_origin['cal_value']])
        longitudu = np.array([float(i) for i in data_origin['station_lon']])
        latitude = np.array([float(i) for i in data_origin['station_lat']])

        data_interpolation = interpolation.deal_interpolation(interpolation, longitudu, latitude, cal_value)
        data_interpolation.columns = ['station_lon', 'station_lat', 'interp']
        data_interpolation.insert(0, "factor_code", data_origin['factor_code'][0])
        data_interpolation.insert(1, "create_date", data_origin['create_date'][0])
        print("插值后的数据：\n", data_interpolation)
        print("------------------------------------------")

        data_interpolation.to_sql("aqf_result_interpolation", oraengine, if_exists='append', index=False)

    def deal_interpolation(self, x, y, z):
        nx, ny = 10, 10
        # x = np.array([120.269, 120.275, 120.354, 120.354, 120.288])
        # y = np.array([31.4867, 31.6219, 31.5848, 31.5475, 31.6842])
        # z = np.array([100, 50, 88, 74, 63])
        # print(x, y, z)
        xi = np.linspace(x.min(), x.max(), nx)
        yi = np.linspace(y.min(), y.max(), ny)
        xi, yi = np.meshgrid(xi, yi)
        xi, yi = xi.flatten(), yi.flatten()
        # print(pd.DataFrame(xi))
        # print(pd.DataFrame(yi))

        grid2 = self.scipy_idw(x, y, z, xi, yi)
        # print(pd.DataFrame(grid2))
        # grid2 = grid2.reshape((ny, nx))

        # self.plot(x, y, z, grid2)
        # plt.title("Scipy's Rbf with function=linear")
        # plt.show()

        x_value = pd.DataFrame(xi, columns=['x'])
        y_value = pd.DataFrame(yi, columns=['y'])
        z_value = pd.DataFrame(grid2, columns=['z'])
        return pd.concat([x_value, y_value, z_value], axis=1)

    def scipy_idw(x, y, z, xi, yi):
        interp = interpolate.Rbf(x, y, z, function='linear')
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
