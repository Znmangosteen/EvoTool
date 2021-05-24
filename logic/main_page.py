from PyQt5.QtGui import QStandardItemModel, QStandardItem
from PyQt5.QtWidgets import *
from PyQt5 import QtWidgets

from load_tools import load_dataset, load_config
from prediction_model_train import prediction_model_train
from ui import *
import os

from ui.main_page import Ui_MainWindow
from pathlib import Path


class MainForm(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainForm, self).__init__()

        self._prediction_model_train = prediction_model_train()

        self._prediction_model_train.process_signal.connect(self.update_process)
        self.setupUi(self)
        self.open_dataset_btn.clicked.connect(self.open_dataset)
        self.open_model_btn.clicked.connect(self.open_model)
        self.train_btn.clicked.connect(self.train_model)
        self.prediction_config_btn.clicked.connect(self.open_model_config)

        self.conf_table_model = QStandardItemModel()
        self.conf_table_model.setHorizontalHeaderLabels(['设定', '值'])
        self.config_table.setModel(self.conf_table_model)

        self.dataset_table_model = QStandardItemModel()
        self.dataset_table_model.setHorizontalHeaderLabels(['设定', '值'])
        self.dataset_table.setModel(self.dataset_table_model)

        self.open_result_btn.clicked.connect(self.open_result_folder)

        # self.algo_select.activated[str].connect(self.change_algo)

    def open_dataset(self):
        # dataset_name, fileType = QtWidgets.QFileDialog.getOpenFileName(self, "选取数据集", os.getcwd(),
        #                                                                "dataset(*.xlsx *.csv)")
        dataset_name, fileType = QtWidgets.QFileDialog.getOpenFileName(self, "选取数据集配置文件", os.getcwd(),
                                                                       "dataset(*.yaml)")
        if dataset_name:
            self._prediction_model_train.set_dataset(**load_dataset(dataset_name))

        self.dataset_table_model.clear()
        self.dataset_table_model.setHorizontalHeaderLabels(['设定', '值'])
        for k, v in self._prediction_model_train.dataset_config.items():
            self.dataset_table_model.appendRow([QStandardItem(str(k)), QStandardItem(str(v))])

    def open_model_config(self):
        config_name, fileType = QtWidgets.QFileDialog.getOpenFileName(self, "选取配置文件", os.getcwd(), "config(*.yaml)")

        config = load_config(config_name)

        self.conf_table_model.clear()
        self.conf_table_model.setHorizontalHeaderLabels(['设定', '值'])
        for k, v in config.items():
            self.conf_table_model.appendRow([QStandardItem(str(k)), QStandardItem(str(v))])

        if config_name:
            self._prediction_model_train.set_model_config(config)

    def open_model(self):
        fileName, fileType = QtWidgets.QFileDialog.getOpenFileName(self, "选取文件", os.getcwd(),
                                                                   "model(*.eam)")

    def open_result_folder(self):
        os.startfile(Path(self._prediction_model_train.save_path))

    def update_process(self, process):
        print(process)
        self.progressBar.setValue(process)

    def train_model(self):
        self._prediction_model_train.start()

    # def change_algo(self,selected_algo):
    #     self._prediction_model_train.set_algo(selected_algo)
    #     print(selected_algo)
