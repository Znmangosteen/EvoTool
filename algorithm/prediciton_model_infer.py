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


class prediction_model_train(QThread):
    process_signal = pyqtSignal(int)

    def __init__(self, **kwargs):
        super().__init__()
        self.dataset_config = {}
        self.dataset = pd.DataFrame()

        self.set_dataset(**load_dataset('./dataset/emission.yaml'))
        self.chosen_algo = self.ALGO_DICT['random_forest']
        self.model_config = {}
        self.set_model_config(load_config('./model_config/rf_config.yaml'))

        self.save_path = ''

        self.enable_feature_select = True

    def set_algo(self, algo):
        self.chosen_algo = ALGO_DICT[algo]

    def load_model(self, path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def set_dataset(self, dataset_config, dataset):
        self.dataset_config = dataset_config
        self.dataset = dataset

    def infer(self):
        data = self.dataset
        self.model.predict(data)
