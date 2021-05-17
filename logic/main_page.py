from PyQt5.QtWidgets import *
from PyQt5 import QtWidgets

from prediction_model_train import prediction_model_train
from ui import *
import os

from ui.main_page import Ui_MainWindow


class MainForm(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainForm, self).__init__()

        self._prediction_model_train = prediction_model_train()

        self._prediction_model_train.process_signal.connect(self.update_process)
        self.setupUi(self)
        self.open_dataset_btn.clicked.connect(self.open_dataset)
        self.open_model_btn.clicked.connect(self.open_model)
        self.train_btn.clicked.connect(self.train_model)

        # self.open_file()

    def open_dataset(self):
        dataset_name, fileType = QtWidgets.QFileDialog.getOpenFileName(self, "选取数据集", os.getcwd(),
                                                                       "dataset(*.xlsx *.csv)")

        self._prediction_model_train.set_train_data(dataset_name)

    def open_model_config(self):
        config_name, fileType = QtWidgets.QFileDialog.getOpenFileName(self, "选取配置文件", os.getcwd(), "config(*.yaml)")
        self._prediction_model_train.set_model_config(config_name)

    def open_model(self):
        fileName, fileType = QtWidgets.QFileDialog.getOpenFileName(self, "选取文件", os.getcwd(),
                                                                   "model(*.eam)")

    def update_process(self, process):
        print(process)
        self.progressBar.setValue(process)

    def train_model(self):
        self._prediction_model_train.start()
