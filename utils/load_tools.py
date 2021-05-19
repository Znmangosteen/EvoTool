import yaml
from PyQt5.QtWidgets import *
from PyQt5 import QtWidgets
from ui import *
import os
import pandas as pd


def load_dataset(dataset_name):
    try:
        if dataset_name.endswith('xlsx'):
            data = pd.read_excel(dataset_name)
        elif dataset_name.endswith('csv'):
            data = pd.read_csv(dataset_name)
        else:
            print('不支持的数据类型')
            assert 0
        return data
    except Exception as e:
        print(e)
        print('数据加载失败')


def load_config(config_name):

    with open(config_name) as f:
        params = yaml.load(f, Loader=yaml.SafeLoader)
    return params
# def open_file(self):
#     # print('123')
#     fileName, fileType = QtWidgets.QFileDialog.getOpenFileName(self, "选取文件", os.getcwd(),
#                                                                "All Files(*);;Text Files(*.txt)")
#     print(fileName)
#     print(fileType)
