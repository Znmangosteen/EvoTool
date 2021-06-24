import pickle

import yaml
from PyQt5.QtWidgets import *
from PyQt5 import QtWidgets
from sklearn.model_selection import train_test_split

from ui import *
import os
import pandas as pd


def load_dataset(dataset_config_name):
    try:
        with open(dataset_config_name, encoding='utf-8') as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)

        train_dataset_name = config['train_dataset']
        if train_dataset_name.endswith('xlsx'):
            train_dataset = pd.read_excel(train_dataset_name)
        elif train_dataset_name.endswith('csv'):
            train_dataset = pd.read_csv(train_dataset_name)
        else:
            print('不支持的数据类型')
            assert 0

        if config['split_dataset']:

            train_dataset, val_dataset = train_test_split(train_dataset, test_size=config['test_size'],
                                                          random_state=config['random_state'])
        else:
            val_dataset_name = config['val_dataset']
            if val_dataset_name.endswith('xlsx'):
                val_dataset = pd.read_excel(val_dataset_name)
            elif train_dataset_name.endswith('csv'):
                val_dataset = pd.read_csv(val_dataset_name)
            else:
                print('不支持的数据类型')
                assert 0
        val_dataset = {'x': val_dataset[config['feature_columns']], 'y': val_dataset[config['output_column']]}
        train_dataset = {'x': train_dataset[config['feature_columns']], 'y': train_dataset[config['output_column']]}
        return {'dataset_config': config, 'train_dataset': train_dataset, 'val_dataset': val_dataset}
    except Exception as e:
        print(e)
        print('数据加载失败')


def load_config(config_name):
    try:
        with open(config_name, encoding='utf-8') as f:
            params = yaml.load(f, Loader=yaml.SafeLoader)
        return params
    except Exception as e:
        print(e)
        print('模型配置加载失败')

def load_model(model_name):
    try:
        with open(model_name, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(e)
        print('模型加载失败')

# def open_file(self):
#     # print('123')
#     fileName, fileType = QtWidgets.QFileDialog.getOpenFileName(self, "选取文件", os.getcwd(),
#                                                                "All Files(*);;Text Files(*.txt)")
#     print(fileName)
#     print(fileType)
