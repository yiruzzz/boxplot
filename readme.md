### 箱形图
一、四分位数：Q1（下四分位数）、Q2（中位数）、Q3（上四分位数），处于所有数字长度的1/4、1/2、3/4处的值。

计算四分位数：

```
    q1_loc = math.ceil(len(dataf[i]) * 0.25 + 1)
    q1 = dataf[i][q1_loc]
    # mid = np.median(dataf[i])
    q3_loc = math.floor(len(dataf[i]) * 0.75 + 1)
    q3 = dataf[i][q3_loc]
    low_whisker = q1 - 1.5 * (q3 - q1)
    up_whisker = q3 + 1.5 * (q3 - q1)
```
或dataframe类型的数据，直接用`data_ser.quantile(0.25)`可以得到下四分位数。

二、上下限：

箱形图两端的两条线，在这两条线的范围之外的点是异常点。

计算方法：

```
    low_whisker = q1 - 1.5 * (q3 - q1)
    up_whisker = q3 + 1.5 * (q3 - q1)
```

三、去除异常点：

```
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
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 7))
    sns.boxplot(y=data[col_name], data=data, palette="Set1", ax=ax[0])
    sns.boxplot(y=data_n[col_name], data=data_n, palette="Set1", ax=ax[1])
    plt.show()
    return data_n
```


### 画箱形图的三种方法：
一、python库matplotlib （可以设置不显示异常点，无法调整坐标文字大小）

```
bplot = plt.boxplot(list, labels=labels, showfliers=False, widths=0.4, capprops={'color':'g', 'linewidth':2}, medianprops={'color':'DarkBlue'})
plt.grid(axis='y')
plt.show()
```

二、pandas （无法设置不显示异常点，无法调整坐标文字大小）

参考 https://www.jb51.net/article/173621.htm

三、matlab （可以设置不显示异常点，可以调整坐标文字大小并旋转一定角度）

```
boxplot(aspect, 'Labels', {'Bridge', 'Vehicle', 'Baseball Diamond', 'Ship', 'Storage Tank', 'Airplane', 'Parking Lot', 'Tennis Court', 'Crossroad', 'Ground Track Field', 'T Junction', 'Basketball Court', 'Harbor'}, 'symbol', '')
text_h = findobj(gca, 'Type', 'text');
set(text_h, 'FontSize', 22);
% xlabel('Class');
% ylabel('Aspect Ratio');
```
其中`'symbol', ''`可以使得异常点不显示

参考：

[python绘制箱线图 - The-Chosen-One - 博客园](https://www.cnblogs.com/yanjy-onlyone/p/13435821.html)

[Python数据可视化：箱线图多种库画法\_pytho](https://www.jb51.net/article/173621.htm)

[Box plot (箱线图) 解读以及Python实现\_boxplot python](https://blog.csdn.net/shulixu/article/details/86551482)

[Python-matplotlib统计图之箱线图漫谈 - 简书](https://www.jianshu.com/p/b2f70f867a4a)

[箱型图Python绘制 - 知乎](https://zhuanlan.zhihu.com/p/321910805)

[ Matplotlib - 箱线图、箱型图 boxplot () 所有用法详解](https://blog.csdn.net/weixin_40683253/article/details/87857194)

[设置箱体颜色 python matplotlib filled boxplots - Stack Overflow](https://stackoverflow.com/questions/20289091/python-matplotlib-filled-boxplots)

[matlab横坐标轴设置,Matlab箱线图Boxplot横坐标x轴设置 - CodeAntenna](https://codeantenna.com/a/27PYLrUn1W)

[matlab画盒图（箱线图）如何不显示异常值](https://blog.csdn.net/summer15407901/article/details/107715646)

[matlab boxplot 字体大小,MATLAB boxplot 字体位置调整以及图片保存问题-爱代码爱编程](https://icode.best/i/44394140275322)
