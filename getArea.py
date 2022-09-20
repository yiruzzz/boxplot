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



df = pd.DataFrame(datas)
dataf = df.T



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



print(dataf.describe())

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

bplot = plt.boxplot(list, labels=labels, showfliers=False, widths=0.4, capprops={'color':'g', 'linewidth':2}, medianprops={'color':'DarkBlue'})
plt.grid(axis='y')
plt.show()

# dataf.to_excel("C:/Users/zyr/Desktop/demo.xlsx", index=False)