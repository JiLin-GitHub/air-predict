# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import time
import random
from datetime import datetime
from matplotlib import pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.dates as mdates
import sqlalchemy
fonts = fm.FontProperties(fname='C:\Windows\Fonts\STXINWEI.TTF',size=16) # 设置字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置-黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

# x = np.arange(1,11,1)
# fig1 = plt.figure(num=1, figsize=[10,6])
# fig2 = plt.figure(num=2, figsize=[10,6])
# ax = fig1.add_subplot(221)
# ax1 = fig1.add_subplot(222)
# ax2 = fig1.add_subplot(223)
# ax.plot(x, x * 2)
# ax1.plot(x,x*4)
# ax2.plot(x,x*6)
# plt.show()

# global_start_time = time.time()
# startTime = datetime.now()
#
# print(global_start_time)
# print(startTime)
# print(random.seed(4))

def main(sql=None, ora=None):
    data = pd.read_sql(sql,ora)
    print(data)
    data8 = data[data['pointid'] == '8']
    data1 = data[data['pointid'] == '1']
    data2 = data[data['pointid'] == '2']
    data3 = data[data['pointid'] == '3']
    data4 = data[data['pointid'] == '4']
    data5 = data[data['pointid'] == '5']
    data6 = data[data['pointid'] == '6']
    data7 = data[data['pointid'] == '7']
    data9 = data[data['pointid'] == '9']
    data12 = data[data['pointid'] == '12']
    data31 = data[data['pointid'] == '31']
    # data = data[data['pointid'] == '8']
    data1.index = range(len(data1))
    data2.index = range(len(data2))
    data3.index = range(len(data3))
    data4.index = range(len(data4))
    data5.index = range(len(data5))
    data6.index = range(len(data6))
    data7.index = range(len(data7))
    data8.index = range(len(data8))
    data9.index = range(len(data9))
    data12.index = range(len(data12))
    data31.index = range(len(data31))


    print(data8.ix[:23,'pm25'])

    dateserise = [datetime.strptime(d.strftime("%Y/%m/%d %H:%M:%S"), "%Y/%m/%d %H:%M:%S") for d in data8.ix[:,'create_date']]

    fig = plt.figure(1,figsize=(15,7))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d %H:%M:%S'), )
    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=24))


    # plt.plot(data.ix[:23,'create_date'],data.ix[:23,'pm25'])
    plt.plot(dateserise[:],data8.ix[:,'pm25']*1000,label='PM25')
    plt.plot(dateserise,data8.ix[:,'humidity'],'r--',label='humidity')
    plt.plot(dateserise,data8.ix[:,'rainfall'],'y--',label='rainfall')
    plt.plot(dateserise,data8.ix[:,'temperature'],'g--',label='temperature')
    # plt.plot(dateserise,data8.ix[:,'so2'],'m--',label='SO2')
    # plt.plot(dateserise,data8.ix[:,'o3'],color='green',label='O3')
    # plt.plot(dateserise[:],data1.ix[:,'pm25'],color='red',label='1号监测站')
    # plt.plot(dateserise[:],data2.ix[:,'pm25'],color='b',label='2号监测站')
    # plt.plot(dateserise[:],data3.ix[:,'pm25'],color='m',label='3号监测站')
    # plt.plot(dateserise[:],data4.ix[:,'pm25'],color='g',label='4号监测站')
    # plt.plot(dateserise[:],data5.ix[:,'pm25'],color='c',label='5号监测站')
    # plt.plot(dateserise[:],data6.ix[:,'pm25'],color='y',label='6号监测站')
    # plt.plot(dateserise[:],data7.ix[:,'pm25'],color='k',label='7号监测站')
    # plt.plot(dateserise[:], data8.ix[:, 'pm25'], label='8号监测站')
    # plt.plot(dateserise[:],data9.ix[:,'pm25'],color='0.8',label='9号监测站')
    # plt.plot(dateserise[:],data12.ix[:,'pm25'],'r:',label='12号监测站')
    # plt.plot(dateserise[:],data31.ix[:,'pm25'],'g:',label='31号监测站')




    plt.xlabel('时间')
    plt.ylabel('数量（降水:mm PM2.5:微克/1000/立方米）')
    plt.title('8号旺庄监测站')
    plt.gcf().autofmt_xdate()
    plt.grid(True,linestyle=':', linewidth=0.5)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    sql = "SELECT * FROM TRAININGDATA WHERE create_date>=to_date('2016-6-1 00','YYYY-MM-DD HH24')" \
          "AND create_date < to_date('2016-7-1 00','YYYY-MM-DD HH24') ORDER BY CREATE_DATE,TO_NUMBER(POINTID)"
    oraengine = sqlalchemy.create_engine('oracle://scott:JL123456@localhost:1521/orcljl')
    main(sql,oraengine)




