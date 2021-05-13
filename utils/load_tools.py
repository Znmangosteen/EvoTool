from PyQt5.QtWidgets import *
from PyQt5 import QtWidgets
from ui import *
import os
import pandas as pd


def load_dataset(dataset_name):
    try:
        if dataset_name.endwith('xlsx'):
            data = pd.read_excel(dataset_name)
        elif dataset_name.endwith('csv'):
            data = pd.read_csv(dataset_name)
        else:
            print('不支持的数据类型')
            assert 0
        s = data
    except:
        print('数据加载失败')

def open_file(self):
    # print('123')
    fileName, fileType = QtWidgets.QFileDialog.getOpenFileName(self, "选取文件", os.getcwd(),
                                                               "All Files(*);;Text Files(*.txt)")
    print(fileName)
    print(fileType)
