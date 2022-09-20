from cProfile import label
import json
from unicodedata import category
import math
import matplotlib.pyplot as plt
import pandas as pd

def getList(path, datas):
    # 获取各类别目标的面积列表
    dict_13 = {"0" : [], "1" : [],"2" : [],"3" : [],"4" : [],"5" : [],"6" : [],"7" : [],"8" : [],"9" : [],"10" : [],"11" : [],"12" : []}

    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    annotations = data['annotations']

    for i in range(len(annotations)):
        dict_temp = annotations[i]
        # 得到一个各类面积的字典
        dict_13[str(dict_temp["category_id"])].append(math.sqrt(dict_temp["area"]))
        # dict_13[str(dict_temp["category_id"])].append(dict_temp["area"])

    
    for i in range(len(dict_13)):
        for j in range(len(dict_13[str(i)])):
            datas[i].append(dict_13[str(i)][j])
    return datas
        
path1 = 'C:/Users/zyr/Desktop/hrrsd_train_m-fld_4352_3084.json'
path2 = 'C:/Users/zyr/Desktop/hrrsd_test_m-fld_13057_3084.json'
path3 = 'C:/Users/zyr/Desktop/hrrsd_val_m-fld_4352_3084.json'
path4 = 'C:/Users/zyr/Desktop/hrrsd_tiny_100_3084.json'
datas = [[], [], [], [], [], [], [], [], [], [], [], [], []]

datas = getList(path1, datas)
datas = getList(path2, datas)
datas = getList(path3, datas)
datas = getList(path4, datas)









labels = ["Bridge", "Vehicle", "Baseball Diamond", "Ship", "Storage Tank", "Airplane", "Parking Lot", "Tennis Court", "Crossroad", "Ground Track Field", "T Junction", "Basketball Court", "Harbor"]

# import numpy as np
# Data = np.random.normal((3, 5, 4), (1.25, 1.00, 1.25), (100, 3))
# fig = plt.figure()
# view = plt.boxplot(Data)
# plt.show()

# bplot = plt.boxplot(datas, labels=labels, showfliers=False)

# plt.show()




# import pandas as pd
# num =[1,2,3,4,5,6,7,8]
# df = pd.DataFrame(num)
# boxplot = df.boxplot()
# print(df.describe())
# plt.show()

df = pd.DataFrame(datas)
dataf = df.T

Qs = []
for i in range(13):
    q1_loc = math.ceil(len(dataf[i]) * 0.25 + 1)
    q1 = dataf[i][q1_loc]
    # mid = np.median(dataf[i])
    q3_loc = math.floor(len(dataf[i]) * 0.75 + 1)
    q3 = dataf[i][q3_loc]
    low_whisker = q1 - 1.5 * (q3 - q1)
    up_whisker = q3 + 1.5 * (q3 - q1)
    qs = []
    qs.append(low_whisker)
    qs.append(up_whisker)
    # qs.append(mid)
    Qs.append(qs)

# print(Qs)
# dataf2 = []
# for i in range(13):
#     y = [x / 100 for x in dataf[i]]
#     dataf2.append(y)

import seaborn as sns
from operator import itemgetter
import numpy as np
# %matplotlib inline

def outliers_proc(data, col_name, scale=3):
    """
    用于清洗异常值，默认用 box_plot（scale=3）进行清洗
    :param data: 接收 pandas 数据格式
    :param col_name: pandas 列名
    :param scale: 尺度
    :return:
    """

    def box_plot_outliers(data_ser, box_scale):
        """
        利用箱线图去除异常值
        :param data_ser: 接收 pandas.Series 数据格式
        :param box_scale: 箱线图尺度，
        :return:
        """
        iqr = box_scale * (data_ser.quantile(0.75) - data_ser.quantile(0.25))
        val_low = data_ser.quantile(0.25) - 1.5 * iqr
        val_up = data_ser.quantile(0.75) + 1.5 * iqr
        rule_low = (data_ser < val_low)
        rule_up = (data_ser > val_up)
        return (rule_low, rule_up), (val_low, val_up)

    data_n = data.copy()
    data_series = data_n[col_name]
    rule, value = box_plot_outliers(data_series, box_scale=scale)
    index = np.arange(data_series.shape[0])[rule[0] | rule[1]]
    print("Delete number is: {}".format(len(index)))
    data_n = data_n.drop(index)
    data_n.reset_index(drop=True, inplace=True)
    print("Now column number is: {}".format(data_n.shape[0]))
    index_low = np.arange(data_series.shape[0])[rule[0]]
    outliers = data_series.iloc[index_low]
    print("Description of data less than the lower bound is:")
    print(pd.Series(outliers).describe())
    index_up = np.arange(data_series.shape[0])[rule[1]]
    outliers = data_series.iloc[index_up]
    print("Description of data larger than the upper bound is:")
    print(pd.Series(outliers).describe())
    
    # fig, ax = plt.subplots(1, 2, figsize=(10, 7))
    # sns.boxplot(y=data[col_name], data=data, palette="Set1", ax=ax[0])
    # sns.boxplot(y=data_n[col_name], data=data_n, palette="Set1", ax=ax[1])
    # plt.show()
    return data_n

for i in range(13):
    Train_data = outliers_proc(dataf, i, scale=3)
    dataf = Train_data

# for i in range(13):
#     for j in range(len(dataf[i])):
#         if dataf[i][j] > Qs[i][1]:
#             # dataf[i][j] = Qs[i][2]
#             dataf[i][j] = 0



print(dataf.describe())
# print(dataf.describe())
# print(Train_data.shape)
# print(Train_data.describe())
# array = np.array(dataf)
# list = array.tolist()
# bplot = plt.boxplot(dataf, labels=labels, showfliers=False)
boxplot = dataf.boxplot()
plt.show()
print(dataf)
array = np.array(dataf.T)
print(array.shape)
list = array.tolist()
print("len", len(list[0]))
for i in range(13):
    for j in range(len(list[i])):
        # print(j)
        if math.isnan(list[i][j]):
            print('j', j)
            temp = list[i][:j]
            list[i] = temp
            break

# print(list)
bplot = plt.boxplot(list, showfliers=False)
plt.show()

# dataf2 = dataf.copy()
# for i in range(13):
#     Train_data = outliers_proc(dataf2, i, scale=3)
#     dataf2 = Train_data

# print(dataf2.describe())
# boxplot = dataf2.boxplot()
# plt.show()

# df = pd.read_excel("C:/Users/zyr/Desktop/demo.xlsx")
# dataf = df.T
# print(dataf.describe())
# print(df.describe())

# datas.counts[(datas.counts > up_whisker) | (datas.counts < low_whisker)]
# plot = dataf.boxplot()
# plt.show()


# print(df.shape)
# df.to_excel("C:/Users/zyr/Desktop/demo.xlsx", index=False)