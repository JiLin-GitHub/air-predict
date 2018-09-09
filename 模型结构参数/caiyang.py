from sklearn.datasets import make_classification
from collections import Counter
import numpy as np
import pandas as pd

X, y = make_classification(n_samples=5000, n_features=3, n_informative=2,
                           n_redundant=0, n_repeated=0, n_classes=3,
                           n_clusters_per_class=1,
                           weights=[0.01, 0.05, 0.94],
                           class_sep=0.8, random_state=0)

# Out[10]: Counter({0: 64, 1: 262, 2: 4674})

# # 随机采样生成
# # 1、构造一个88*42的二维数组(88就是后面想要扩展到多少条数据，42对应特征数，数组内容随意)
# x1 = np.random.rand(88,42)
# # 2、使用pandas输出数据库,找到class字段值为‘U2R’的11条记录
# x2 = pd.read_csv('...')
# x2 = x2[x2["class"] == 'U2R']
# # 3、将x1、x2合并，获得（88+11）*42维度的二维数组
# X = np.concatenate((x1, x2), axis = 1)  #这里具体怎么合并查一下
# # 4、生成分类标签一维数组.大小为（88+11）*1;前88个类标为1，后11个2（其他也可以，只要同类相同，不同类不同即可）
# y = [1,1,1,1,'...',1,1,2,2,2,2,2,2,2,2,2,2,2]
# # 5、用X、y进行下面过采样操作

from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=None)
X_resampled, y_resampled = ros.fit_sample(X, y)
# print(X_resampled[y_resampled==2],type(X_resampled[y_resampled==2]),X_resampled[y_resampled==2].shape)
# 6、取出采样结果
final = X_resampled[y_resampled==2]
# 7、结果导入表（导入之前删除原来被采样的记录，final中已包含）
# pd.to_csv(final，...)
