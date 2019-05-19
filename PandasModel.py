import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QAction, QTableWidget, QTableWidgetItem, QVBoxLayout

import pandas as pd #Kütüphaneyi projeye dahil ediyoruz.

class PandasModel(QWidget):
    def __init__(self,DataFrame,title,satir,sutun,kullanici_satir):
        super().__init__()
        self.title =title
        self.left = 20
        self.top = 20
        self.width = 800
        self.height = 600
        self.initUI(satir,sutun,DataFrame,kullanici_satir)
        self.sayi=150
    def initUI(self,satir,sutun,DataFrame,kullanici_satir):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.createTable(satir,sutun,DataFrame,kullanici_satir)

        # Add box layout, add table to box layout and add box layout to widget
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.tableWidget)
        self.setLayout(self.layout)

        # Show widget
        self.show()

    def createTable(self,satir,sutun,Dataframe,kullanici_satir):
        # Create table
        self.tableWidget = QTableWidget()
        self.tableWidget.setRowCount(kullanici_satir+1) # sutun bilgileri ekleneceği için bir fazlasını aldım
        self.tableWidget.setColumnCount(sutun)
        #print(Dataframe.values.tolist())
        sayac=0
        for column in Dataframe:
            self.tableWidget.setItem(0,sayac, QTableWidgetItem(str(column)))
            sayac+=1
        for i in range(0,kullanici_satir):
            for j in range(0,sutun):
                #print(i," ",j)
                self.tableWidget.setItem(i+1, j, QTableWidgetItem(str(Dataframe.values[i][j])))

        #self.tableWidget.move(0, 0)

        # table selection change

    # @pyqtSlot()
    # def on_click(self):
    #     print("\n")
    #     for currentQTableWidgetItem in self.tableWidget.selectedItems():
    #         print(currentQTableWidgetItem.row(), currentQTableWidgetItem.column(), currentQTableWidgetItem.text())


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = PandasModel()
    #sys.exit(app.exec_())