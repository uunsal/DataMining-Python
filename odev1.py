# UFUK REMZİ ÜNSAL
# NO = 161213125 - Birinci Öğretim
# Knn Algoritması
import pandas as pd  # Kütüphaneyi projeye dahil ediyoruz.
from PandasModel import PandasModel
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QAction, QTableWidget, QTableWidgetItem, QVBoxLayout
import sys
from math import sqrt
from sklearn.datasets import load_breast_cancer
from id3 import Id3Estimator
from id3 import export_graphviz
import math
import numpy as np
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
import random

path = "Iris.csv"
dosya = open(path, 'rb')
app = QApplication(sys.argv)


# kuyruk = []
# class TreeNode:
#     def __init__(self,data):
#         self.data=data
#         self.children = []
#     def get_data(self):
#         return self.data
#     def get_children(self):
#         return self.children
#
def table_show(veriler, title, Row_Count, Column_Count, Row_Count_İnput):
    DataFrame = pd.DataFrame(veriler)
    # print(DataFrame)
    row_count = Row_Count
    column_Count = Column_Count
    ex = PandasModel(DataFrame, title, Row_Count, Column_Count, Row_Count_İnput)
    sys.exit(app.exec_())


#
#
# def recursive_build_tree(dugum,veriler,tumveriler,sutunno):
#     clas = pd.Series(tumveriler[list(tumveriler.columns)[int(sutunno) - 1]])
#     for i in dugum.get_children():
#         eklenecek = i
#         i=i.get_data()
#         category={}
#         category[i] =[]
#         Gainler=[]
#         degerler = {}
#         dal = TreeNode(0)
#         temp = []
#         print(clas.value_counts())
#         for t in clas.value_counts().keys():
#             degerler[t]=0
#         Rec_Entropy = 0
#         for indis in range(0,len(tumveriler[dugum.get_data()])):
#             if(tumveriler[dugum.get_data()][indis]==i):
#                 #category[i].append(indis)
#                 temp.append(indis)
#
#         category[i].append(degerler)
#         if(dugum.data in veriler.columns):
#             veriler = veriler.drop(dugum.get_data(), axis=1)
#         print(temp)
#         print(category)
#         for s in temp:
#             category[i][0][tumveriler.values[s][sutunno-1]]+=1
#         for y in category[i][0]:
#             deger1,deger2 =category[i][0][y],len(temp)
#             if(deger1!=0):
#                 Rec_Entropy+= -((deger1/deger2)*math.log(deger1/deger2,2))
#         print(Rec_Entropy)
#         if(Rec_Entropy ==0):
#             print(dugum.get_data())
#
#         for column in veriler.columns:
#             print(column)
#             gain = 0
#             new_veriler= {}
#             new_veriler[column] = {}
#             if(column!=tumveriler.columns[0] and column!=tumveriler.columns[sutunno-1]):
#                 for cc in veriler[column].value_counts().keys():
#                     dizi = []
#                     for ka in temp:
#                         dizi.append(veriler.values[ka][list(veriler.columns).index(column)])
#                     alt_kategori = pd.Series(dizi)
#                     for cc2 in alt_kategori.value_counts().keys():
#                         new_veriler[column][cc2] = [alt_kategori.value_counts()[cc2], {}]
#                         for l in clas:
#                             new_veriler[column][cc2][1][l]=0
#
#                 for veri in temp:
#                     new_veriler[column][veriler.values[veri][list(veriler.columns).index(column)]][1][tumveriler.values[veri][sutunno-1]]+=1
#                 for gain in new_veriler:
#                     Gain = 0
#                     for gain2 in new_veriler[gain]:
#                         bolen  =new_veriler[gain][gain2][0]
#                         Entopy=0
#                         for gain3 in new_veriler[gain][gain2][1]:
#                             if(new_veriler[gain][gain2][1][gain3]!=0):
#                                 deger=new_veriler[gain][gain2][1][gain3]
#                                 Entopy += (deger/bolen)*math.log(deger/bolen,2)
#                         if(Entopy!=0):
#                             Entopy = -Entopy
#                         Gain+= (bolen/len(temp))*Entopy
#                     Gainler.append(Gain)
#                 dal = TreeNode(veriler.columns[Gainler.index(pd.Series(Gainler).min())+1])
#
#         if(pd.Series(Gainler).min()==0):
#             sutun = veriler.columns[Gainler.index(pd.Series(Gainler).min())+1]
#             cls = veriler[sutun].value_counts()
#             sonlar=set()
#             for son in temp:
#                 for son1 in cls.keys():
#                     if(son1==veriler.values[son][list(veriler.columns).index(sutun)]):
#                         sonlar.add(tumveriler.values[son][sutunno-1])
#                         break
#
#             for zero_max_gain in sonlar:
#                 new_n=TreeNode(zero_max_gain)
#                 dal.get_children().append(new_n)
#
#             eklenecek.get_children().append(dal)
#             #return
#             if(Rec_Entropy==0):
#                 dal.get_children().append(veriler[tumveriler.columns[sutunno-1]].value_counts().keys()[0])
#                 return
#         for p in veriler[dal.get_data()].value_counts().keys():
#             dal.get_children().append(TreeNode(p))
#         eklenecek.get_children().append(dal)
#         degerlendir = []
#         print(dal.get_data())
#         for bit in veriler[tumveriler.columns[sutunno-1]].value_counts().keys():
#
#             degerlendir.append(bit)
#         for t in dal.get_children():
#             if not(t.get_data() in degerlendir):
#                 recursive_build_tree(t,veriler,tumveriler,sutunno)
#
#
#
#
#
#
#
#
#
#
#
#
# def DecisionTree(sutunno,veriler):
#     degerler = {}
#     AnaEntropy = 0
#     classes = pd.Series(veriler[list(veriler.columns)[int(sutunno)-1]])
#     for i in classes.value_counts().keys():
#         AnaEntropy +=-(classes.value_counts()[i]/len(veriler.values))*math.log(classes.value_counts()[i]/len(veriler.values),2)
#     print(AnaEntropy)
#     for i in veriler.columns:
#         if(i!=veriler.columns[0] and i!=veriler.columns[int(sutunno)-1]):
#             degerler[i]={}
#             for j in veriler[i].value_counts().keys():
#
#                 degerler[i][j]=[veriler[i].value_counts()[j],{}]
#                 for p in classes.value_counts().keys():
#                     degerler[i][j][1][p] =0
#     for j in degerler.keys():
#         for a in degerler [j]:
#             for b in veriler.values:
#                 if(b[list(veriler.columns).index(j)]==a):
#                     #print(a,b[int(sutunno)-1])
#                     #print(a,b)
#                     degerler[j][a][1][b[int(sutunno)-1]]+=1
#     Gainler = []
#     Entropi = 0
#     Gain=0
#     Entropiler=[]
#     for key in degerler.keys():
#         for j in degerler[key]:
#             Entropi = 0
#             for t in degerler[key][j][1]:
#                 if (degerler[key][j][1][t]/degerler[key][j][0])!=0:
#                     deger1,deger2=degerler[key][j][1][t],degerler[key][j][0]
#                     #print(deger1,deger2)
#                     Entropi+=-(deger1/deger2)*math.log(deger1/deger2,2)
#
#             Gain +=( deger2/len(veriler.values))*Entropi
#         Gainler.append(Gain)
#         Gain = 0
#     min_gain = pd.Series(Gainler).min()
#     root = TreeNode(veriler.columns[Gainler.index(min_gain)+1])
#     for i in veriler[root.data].value_counts().keys():
#         new_node = TreeNode(i)
#         root.get_children().append(new_node)
#     recursive_build_tree(root,veriler,veriler,int(sutunno))
#     return root
#
#
# def TreeTraversal(dugum):
#     print(dugum.get_data(),dugum.get_children())
#     for i in dugum.get_children():
#         kuyruk.append(i)
#     if(len(kuyruk)>0):
#         TreeTraversal(kuyruk.pop(0))
#
def Categorical_Data(veriler, sutunno):
    DataF = {}
    le = LabelEncoder()

    for i in veriler.columns:
        if veriler[i].dtype == "object" and i != veriler.columns[sutunno]:
            DataF[i] = le.fit_transform(veriler[i])
        else:
            DataF[i] = veriler[i]
    return pd.DataFrame(DataF)


def decision_tree(veriler, sutunno):
    print("\n\n\n", "-" * 50, "KARAR AĞACI", "-" * 50)
    # clf = Id3Estimator()
    # veriler = veriler.drop("Id",axis=1)
    # clf.fit(np.array(veriler.columns),np.array(veriler.values), check_input=True)
    classifier = DecisionTreeClassifier()
    veriler = Categorical_Data(veriler, sutunno)
    X = veriler.drop(veriler.columns[sutunno], axis=1)
    Y = veriler[veriler.columns[sutunno]]
    # if "object" in veriler.dtypes.values: #kategorik veri varsa
    #     le = preprocessing.LabelEncoder()
    #     X = X.apply(le.fit_transform)
    #     print(X.head())
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30)
    classifier = DecisionTreeClassifier(criterion='entropy')
    classifier.fit(X_train, y_train)
    # print(X_test,"\n",X_train)
    y_pred = classifier.predict(X_test)
    Dogrular_yanlışlar = {}
    Dogrular_yanlışlar_oranlar = {}
    for dy in veriler[veriler.columns[sutunno]].value_counts().keys():
        Dogrular_yanlışlar[dy] = {"Doğru": 0, "Yanlış": 0}
        Dogrular_yanlışlar_oranlar[dy] = {"Doğru %": 0, "Yanlış %": 0}
    print("\nÖrnek Sayısı:", len(veriler.values), " ", "Nitelik Sayısı:", len(X.columns), "\n")
    print("Eğitim Verisi Örnek Sayısı:", len(X_train), "Test Verisi Örnek Sayısı:", len(X_test), "\n")
    sayac = 0
    for dy_bul in y_test:
        if dy_bul == list(y_pred)[sayac]:
            Dogrular_yanlışlar[dy_bul]["Doğru"] += 1
        else:
            Dogrular_yanlışlar[dy_bul]["Yanlış"] += 1
        sayac += 1

    print("Doğru Ve Yanlış Sayıları", Dogrular_yanlışlar, "\n")
    for j in Dogrular_yanlışlar:
        dogru, yanlış = Dogrular_yanlışlar[j]["Doğru"], Dogrular_yanlışlar[j]["Yanlış"]
        if (dogru + yanlış) != 0:
            Dogrular_yanlışlar_oranlar[j]["Doğru %"] = (dogru / (dogru + yanlış)) * 100
            Dogrular_yanlışlar_oranlar[j]["Yanlış %"] = (yanlış / (dogru + yanlış)) * 100
    print("Doğru Ve Yanlış Oranları (Yüzde)", Dogrular_yanlışlar_oranlar, "\n")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    df = pd.DataFrame({'Gerçek Veri': y_test, 'Tahmin': y_pred})
    print(df)


def Euclidean_Distance(veriler, new, sutunno, K):
    uzaklıklar = []
    uzaklıklarT = []
    sayac = 0
    for v in veriler.values:
        hesap = 0
        for j in range(0, len(new)):
            hesap += (float(v[j]) - float(new[veriler.columns[j]])) ** 2
        uzaklıklar.append(sqrt(hesap))
        uzaklıklarT.append(sqrt(hesap))
        sayac += 1

    enYakinlar = []
    for k in range(0, K):
        enYakinlar.append(uzaklıklar.index(min(uzaklıklarT)))
        uzaklıklarT.pop(uzaklıklarT.index(min(uzaklıklarT)))

    return enYakinlar, uzaklıklar


def minmaxNormalization(veriler, sutunno):  # 0 - 1 normalizasyonu yapar
    normalizeVeriler = {}
    for c in veriler.columns:
        normalizeVeriler[c] = []
    for v in veriler.columns:
        if (v != veriler.columns[sutunno]):
            for d in veriler[v].tolist():
                max = veriler[v].max()
                min = veriler[v].min()
                normalizeVeriler[v].append((int(d) - min) / (max - min))  # 0-1 normalizasyonu
        if (v == veriler.columns[sutunno]):
            for d in veriler[v].tolist():
                normalizeVeriler[v].append(d)
    return pd.DataFrame(normalizeVeriler)


def KnnVeriHazırla(veriler, yuzde):  # Test verisi
    testCount = int(veriler.shape[0] * yuzde)
    otherCount = int(veriler.shape[0] - testCount)
    rowCount = veriler.shape[0]
    testIndex = []
    verilerDataframe = {}
    sayac = 0
    for j in veriler.columns:
        verilerDataframe[j] = []
    while sayac < testCount:
        sayi = (random.randint(0, rowCount - 1))
        if not (sayi in testIndex):
            testIndex.append(sayi)
            sayac += 1
    # print(testIndex,"\n",len(testIndex))
    for v in veriler.columns:
        for d in range(0, len(veriler[v].tolist())):
            if not d in testIndex:
                verilerDataframe[v].append(veriler[v].tolist()[d])
    return testIndex, (pd.DataFrame(verilerDataframe))


def KnnHesapla(veriler, new, newGercek, sutunno, K, yontem, InputK):
    enYakinlar, uzaklıklar = Euclidean_Distance(veriler, new, sutunno, K)
    sınıfEtiketleri = {}
    agırlıklar = {}
    if (int(yontem) == 1):
        for v in veriler[veriler.columns[sutunno]]:
            sınıfEtiketleri[v] = 0
        for s in enYakinlar:
            s = int(s)
            sınıfEtiketleri[veriler.values[s][sutunno]] += 1
        maxEtiket = max(list(sınıfEtiketleri.values()))
        if (InputK == True):
            print("\n", sınıfEtiketleri)
            print("\n\n", newGercek, " noktasının sınıfı ",
                  list(sınıfEtiketleri.keys())[list(sınıfEtiketleri.values()).index(maxEtiket)], " olarak belirlenir.")
        else:
            return list(sınıfEtiketleri.keys())[list(sınıfEtiketleri.values()).index(maxEtiket)]

    elif (int(yontem) == 2):
        for v in veriler[veriler.columns[sutunno]]:
            sınıfEtiketleri[v] = []
            agırlıklar[v] = 0
        for s in enYakinlar:
            s = int(s)
            sınıfEtiketleri[veriler.values[s][sutunno]].append(1 / (uzaklıklar[s] ** 2))
        #print(sınıfEtiketleri)
        for e in sınıfEtiketleri:
            for l in sınıfEtiketleri[e]:
                agırlıklar[e] += l
        if (InputK == True):
            print("\n", agırlıklar)
            print("\n\n", newGercek, " noktasının sınıfı ",
                  list(agırlıklar.keys())[(list(agırlıklar.values()).index(max(agırlıklar.values())))]
                  , " olarak belirlenir.")
        else:
            return list(agırlıklar.keys())[(list(agırlıklar.values()).index(max(agırlıklar.values())))]
    else:
        print("Yanlış giriş yaptınız lütfen tekrar deneyiniz..")


def Knn(veriler, gercekVeriler):
    print("\n", "-" * 40, "KNN ALGORITMASI", "-" * 40, "\n\n")
    K = int(input("K parametresini giriniz : "))
    yontem = int(input("Yontem seçiniz \nEn çok tekrarlanan Sınıfın seçilmesi -> 1\nAğırlıklı oylama ile " +
                       "seçilmesi -> 2 \nSeçiminizi giriniz: "))
    sutunno = int(input("Class sütun index giriniz : "))
    girdiSecim = (input("Yeni örnek hazır mı alınsın kullanıcı tarafından mı alınsın ? (h,k) : "))
    new = {}
    newGercek = {}
    if (girdiSecim == "k"):
        for i in veriler.columns:
            if (i != veriler.columns.tolist()[sutunno]):
                maxC = gercekVeriler[i].max()
                minC = gercekVeriler[i].min()
                deger = (input("Yeni gözlem için " + i + " parametresini giriniz: "))
                new[i] = ((float(deger) - minC) / (maxC - minC))
                newGercek[i] = deger
        veriler = minmaxNormalization(veriler, sutunno)
        KnnHesapla(veriler, new, newGercek, sutunno, K, yontem, True)
    if girdiSecim == "h":
        dogruYanlıslar = {}
        oranlar = {}
        for key in veriler[veriler.columns[sutunno]]:
            dogruYanlıslar[key]=[0,0]
            oranlar[key]=[0,0]
        testIndex, veriler = KnnVeriHazırla(veriler, 0.3) #Hazır örnek için kullanıldı
        veriler = minmaxNormalization(veriler, sutunno)
        print("\nTahmin            Gerçek")
        for i in testIndex:
            new = {}
            newGercek = {}
            newClass = ""
            for c in veriler.columns:
                if(c!=veriler.columns[sutunno]):
                    maxC = gercekVeriler[c].max()
                    minC = gercekVeriler[c].min()
                    new[c] = (float(gercekVeriler.values[i][list(veriler.columns).index(c)])-minC) / (maxC - minC)
                    newGercek[c] = gercekVeriler.values[i][list(veriler.columns).index(c)]
                else:
                    newClass = gercekVeriler.values[i][list(veriler.columns).index(c)] # gerçekteki sınıf etiketi
            tahminSınıfı = KnnHesapla(veriler, new, newGercek, sutunno, K, yontem, False) #False hazır veriler alındığında kullanıldı

            print(tahminSınıfı,"  ",newClass)
            if(tahminSınıfı==newClass):
                dogruYanlıslar[newClass][0]+=1
            else:
                dogruYanlıslar[newClass][1]+=1
        print("\nDoğru Yanlış Sayıları",dogruYanlıslar)
        for do in dogruYanlıslar:
            oranlar[do][0] = (dogruYanlıslar[do][0]/(dogruYanlıslar[do][0]+dogruYanlıslar[do][1]))*100
            oranlar[do][1] = (dogruYanlıslar[do][1]/(dogruYanlıslar[do][0]+dogruYanlıslar[do][1]))*100
        print("Doğru Yanlış Oranları % = ",oranlar)


def main():
    veriler = pd.read_csv(path)
    Row_Count = veriler.shape[0]
    Column_Count = veriler.shape[1]
    # print(veriler.describe())
    # print(veriler["SepalWidthCm"].head(70))
    # print("\nTITLE:",veriler)
    # print(veriler.info())

    print("\nTITLE : ", dosya.name.split(".")[0] + " Dataset")
    print("\nColumn Names:", veriler.columns.tolist(), "\n")
    print("\n5 VALUES SUMMERIZATION")
    sapanveri = False
    # name = [x for x in globals() if globals()[x] is pd.DataFrame(veriler)][0]

    for i in veriler.columns.tolist():
        sapanveri = False

        # if (veriler[i].dtype in ["object"]):
        #     check = True
        #     veriler_obj = []
        #     for j in range(0, veriler[i].count()):
        #         try:
        #             data = float(veriler[i][j])
        #             veriler_obj.append(data)
        #         except ValueError as er:
        #             er.args
        #     if (len(veriler_obj) > 0):
        #         veriler_new = pd.Series(veriler[i],veriler_obj).describe()
        #         print(veriler_new.values)
        #         print(veriler_new.name)
        #         print("Min:", veriler_new.min(), "\nMax:", veriler_new.max(), "\nMedian:", veriler_new.median(), "\nQ1:",
        #               ceyreklikler[0], "\nQ3:", ceyreklikler[1])
        #         print("QUTLIER INDEXS")
        #         ıqr = ceyreklikler[1] - ceyreklikler[0]
        #         print(ıqr)

        if (veriler[i].dtype in ["int64", "float64"]):
            ceyreklikler = veriler[i].quantile([0.25, 0.75]).tolist()
            print(i)
            print("Min:", veriler[i].min(), "\nMax:", veriler[i].max(), "\nMedian:", veriler[i].median(), "\nQ1:",
                  ceyreklikler[0], "\nQ3:", ceyreklikler[1])
            print("QUTLIER INDEXS")
            ıqr = ceyreklikler[1] - ceyreklikler[0]
            # print(ıqr)
            for j in range(0, veriler[i].count()):
                veri = veriler[i][j]
                if (veri < (ceyreklikler[0] - (1.5 * ıqr)) or veri > (ceyreklikler[1] + (ıqr * 1.5))):
                    sapanveri = True
                    print(j)
            if (sapanveri == False):
                print("Outlier sample not found")
            print("-" * 100)

    print("CLASS LABELS AND COUNTS")
    sutunno = input("Sütun index no giriniz(Çıkış ç):")
    while (sutunno != "ç"):
        if not (veriler[veriler.columns.tolist()[int(sutunno)]].dtype in ["int64", "float64"]):
            guruplama = veriler.groupby(veriler.columns.tolist()[int(sutunno)])
            for name, groups in guruplama:
                print(name, ":", dict(veriler[veriler.columns.tolist()[int(sutunno)]].value_counts())[name])
        else:
            print("Girdiğiniz veriler sayısal veridir lütfen başka sütun indexi giriniz.")
        sutunno = input("Sütun index no giriniz(Çıkış ç):")

    # sutunno = input("Sınıflandırma yapılacak index no giriniz:")
    # index_sutunno = input("İndex Sütün numarası giriniz(çıkmak için -1):")
    # sınıflandırılan_veriler = veriler
    # if(int(index_sutunno)!=-1):
    #     sınıflandırılan_veriler = veriler.drop(veriler.columns[int(index_sutunno)],axis=1)
    #     sutunno=int(sutunno)-1
    # yuzdeYetmis = int((Row_Count*70)/100) # eğitim verisi için
    # yuzdeOtuz = Row_Count-yuzdeYetmis # test verisi için
    # root = DecisionTree(sutunno,veriler)
    # TreeTraversal(root)
    # decision_tree(sınıflandırılan_veriler,int(sutunno))
    print()
    Knn(veriler, veriler)
    print()
    # row_count_input=input("Tablonun kaç satırını görmek istersiniz:");
    # if(int(row_count_input)<=Row_Count):
    #     table_show(veriler, dosya.name.split(".")[0] + " Dataset", Row_Count, Column_Count,int(row_count_input))
    # else:
    #     print("Girdiğiniz değer csv dosyasının satır sayısından büyük satır sayısı=",Row_Count)


main()
