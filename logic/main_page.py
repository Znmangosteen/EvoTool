from PyQt5.QtWidgets import *
from PyQt5 import QtWidgets
from ui import *
import os

from ui.main_page import Ui_MainWindow


class MainForm(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainForm, self).__init__()
        self.setupUi(self)
        self.open_dataset_btn.clicked.connect(self.open_dataset)
        self.open_model_btn.clicked.connect(self.open_model)

        # self.open_file()

    def open_dataset(self):
        fileName, fileType = QtWidgets.QFileDialog.getOpenFileName(self, "选取文件", os.getcwd(),
                                                                   "dataset(*.xlsx *.csv)")

    def open_model(self):
        fileName, fileType = QtWidgets.QFileDialog.getOpenFileName(self, "选取文件", os.getcwd(),
                                                                   "model(*.eam)")