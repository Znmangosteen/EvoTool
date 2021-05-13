from PyQt5.QtCore import QCoreApplication
from PyQt5 import QtWidgets
import sys
from logic.main_page import MainForm

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = MainForm()
    win.show()
    sys.exit(app.exec_())