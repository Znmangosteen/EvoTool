# from PyQt5.QtCore import pyqtSignal, QThread
import pickle
import random

import yaml

from algorithms import ALGO_DICT
from lightgbm_encap import lightgbm_model
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import time
import shutil
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from PyQt5.Qt import *
from load_tools import *
import os

from mertric import eval_model
from random_forest_encap import random_forest_model


class prediction_model_infer(QThread):
    process_signal = pyqtSignal(int)

    def __init__(self, **kwargs):
        super().__init__()
        self.dataset_config = {}
        self.dataset = pd.DataFrame()

        # self.set_dataset(**load_dataset('./dataset/emission.yaml'))
        # self.chosen_algo = self.ALGO_DICT['random_forest']
        self.model = None

        self.set_model(load_model('./test_model/model.eam'))
        self.set_dataset(**load_infer_dataset('./dataset/emission_infer_2.yaml'))
        # self.set_model_config(load_config('./model_config/rf_config.yaml'))

        self.save_path = ''

    def set_algo(self, algo):
        self.chosen_algo = ALGO_DICT[algo]

    def set_model(self, model):
        self.model = model

    def set_dataset(self, dataset_config, dataset):
        self.dataset_config = dataset_config
        self.dataset = dataset

    def run(self):
        # 用当前时间作为储存结果文件夹的名字
        folder_name = time.asctime(time.localtime(time.time())).replace(':', '-')
        folder_name = './run/infer/{}/'.format(folder_name)
        self.save_path = folder_name
        os.makedirs(folder_name)

        x = self.dataset['x']
        y = self.dataset['y']
        self.process_signal.emit(5)

        result = self.model.predict(x)

        with open(folder_name + 'result.txt', 'w')as f:
            f.write(result.__str__())

        self.process_signal.emit(100)

        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
