
from sklearn import preprocessing
import numpy as np
import pickle
import random
import warnings 

from sklearn.ensemble import RandomForestRegressor
import sklearn.metrics as sm

from numpy import set_printoptions
set_printoptions(threshold=np.inf)

#忽略一些版本不兼容等警告
warnings.filterwarnings("ignore")

path = 'srcdata10_5'

f1 = open(path, 'rb')
data1 = pickle.load(f1)
train_list = data1['train']
test_list = data1['test']

BATCH_START = 0     # 建立 batch data 时候的 index
TIME_STEPS = 5     # backpropagation through time 的 time_steps
BATCH_SIZE = 10
INPUT_SIZE = 14     # sin 数据输入 size
OUTPUT_SIZE = 1     # cos 数据输出 size

def get_data_batch():
    x_part1 = []
    y_part1 = []
    for item in train_list:
        x_part1.append(item['x'].flatten())
        y_part1.append(item['y'])
    seq =np.array(x_part1)
    res =np.array(y_part1)


    return seq , res


def get_data_batch_test():
    x_part1 = []
    y_part1 = []
    for item in test_list:
        x_part1.append(item['x'].flatten())
        y_part1.append(item['y'])
    seq =np.array(x_part1)
    res =np.array(y_part1)


    return seq , res



def train():

    train_data_x, train_data_y = get_data_batch()
    
    test_data_x, test_data_y = get_data_batch_test()

    rf=RandomForestRegressor(n_estimators=200)
    predict_y =rf.fit(train_data_x,train_data_y).predict(test_data_x)

    print('RandomForestRegressor的r2_score得分：', sm.r2_score(test_data_y, predict_y))
    print('RandomForestRegressor的MSE得分：', sm.mean_squared_error(test_data_y, predict_y))
    print('RandomForestRegressor的MAE得分：', sm.mean_absolute_error(test_data_y, predict_y))

if __name__ == "__main__":
    train()