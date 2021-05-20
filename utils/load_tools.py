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
            dataset_config = yaml.load(f, Loader=yaml.SafeLoader)

        train_dataset_name = dataset_config['train_dataset']
        if train_dataset_name.endswith('xlsx'):
            train_dataset = pd.read_excel(train_dataset_name)
        elif train_dataset_name.endswith('csv'):
            train_dataset = pd.read_csv(train_dataset_name)
        else:
            print('不支持的数据类型')
            assert 0

        if dataset_config['split_dataset']:

            train_dataset, val_dataset = train_test_split(train_dataset, test_size=dataset_config['test_size'],
                                                          random_state=dataset_config['random_state'])
        else:
            val_dataset_name = dataset_config['val_dataset']
            if val_dataset_name.endswith('xlsx'):
                val_dataset = pd.read_excel(val_dataset_name)
            elif train_dataset_name.endswith('csv'):
                val_dataset = pd.read_csv(val_dataset_name)
            else:
                print('不支持的数据类型')
                assert 0

        return {'dataset_config': dataset_config, 'train_dataset': train_dataset, 'val_dataset': val_dataset}
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

# def open_file(self):
#     # print('123')
#     fileName, fileType = QtWidgets.QFileDialog.getOpenFileName(self, "选取文件", os.getcwd(),
#                                                                "All Files(*);;Text Files(*.txt)")
#     print(fileName)
#     print(fileType)
