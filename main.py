from PyQt5.QtCore import QCoreApplication
from PyQt5 import QtWidgets
import sys
from logic.main_page import MainForm
from prediction_model_train import prediction_model_train


def algorithm_init():
    # prediction_model_train()
    pass


if __name__ == "__main__":
    algorithm_init()
    
    app = QtWidgets.QApplication(sys.argv)
    win = MainForm()
    win.show()
    sys.exit(app.exec_())