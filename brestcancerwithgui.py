import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from tkinter import *
from tkinter import filedialog
base = Tk()
# Create a canvas
base.geometry('400x400')
base.title("Detector")
base.configure(background='pink')
def machine(a):
    data=pd.read_csv(str(a))
    data.drop(['id'],axis=1,inplace=True)
    data['diagnosis']=(data['diagnosis']=='M').astype(int)
    Processed_Data=data.drop(['diagnosis'],axis=1)
    Parameters=['radius_mean','perimeter_mean', 'compactness_mean', 'concave points_mean', 'radius_worst','perimeter_worst', 'texture_worst','perimeter_se','radius_se','compactness_se','concave points_se','compactness_worst','concave points_worst', 'area_worst', 'concavity_mean']
    Processed_Data=Processed_Data.drop(Parameters,axis=1)
    
    from sklearn.impute import KNNImputer
    imputer=KNNImputer(n_neighbors=4)
    palette_color = ['black','blue']
    cmap = ['grey','black']
    fig, axs = plt.subplots(nrows=4, ncols=3 ,figsize=(14,18))
    sns.scatterplot(data=data, x='radius_mean', y='texture_mean', hue='diagnosis',palette=palette_color, ax=axs[0][0])
    sns.scatterplot(data=data, x='radius_mean', y='perimeter_mean', hue='diagnosis',palette=palette_color,  ax=axs[0][1])
    sns.scatterplot(data=data, x='radius_mean', y='radius_se', hue='diagnosis',palette=palette_color,  ax=axs[0][2])
    sns.scatterplot(data=data, x='radius_mean', y='concave points_mean', hue='diagnosis',palette=palette_color,  ax=axs[1][0])
    sns.scatterplot(data=data, x='radius_mean', y='smoothness_worst', hue='diagnosis',palette=palette_color,  ax=axs[1][1])
    sns.scatterplot(data=data, x='radius_mean', y='area_mean', hue='diagnosis',palette=palette_color,  ax=axs[1][2])
    sns.scatterplot(data=data, x='radius_mean', y='area_se', hue='diagnosis',palette=palette_color, ax=axs[2][0])
    sns.scatterplot(data=data, x='radius_mean', y='smoothness_se', hue='diagnosis',palette=palette_color,  ax=axs[2][1])
    sns.scatterplot(data=data, x='radius_mean', y='radius_worst', hue='diagnosis',palette=palette_color,  ax=axs[2][2])
    sns.scatterplot(data=data, x='radius_mean', y='concavity_se', hue='diagnosis',palette=palette_color,  ax=axs[3][0])
    sns.scatterplot(data=data, x='radius_mean', y='concave points_se', hue='diagnosis',palette=palette_color,  ax=axs[3][1])
    sns.scatterplot(data=data, x='radius_mean', y='symmetry_worst', hue='diagnosis',palette=palette_color,  ax=axs[3][2])
    #plt.show()

    
    Y=data['diagnosis']
    X=Processed_Data
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test=train_test_split(X,Y,train_size=0.7,test_size=0.3,random_state=0)
    from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,precision_score,recall_score 
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
    classifier.fit(x_train, y_train)
    yPredict=classifier.predict(x_test)
    print('Accuracy: {}'.format(accuracy_score(y_test, yPredict)))
    print('Recall: {}'.format(recall_score(y_test, yPredict)))
    def CM(confusion):
        groups=['TN','FP','FN','TP']
        counts=['{0:0.0f}'.format(value)for value in confusion.flatten()]
        labels=np.asarray([f'{v1}\n{v2}' for v1, v2 in zip(groups,counts)]).reshape(2, 2)
        plt.figure(figsize=(10,10))
        sns.heatmap(confusion,annot=labels,cmap='Reds',cbar=False, fmt='')
        plt.show()
    CM(confusion_matrix(y_test, yPredict))
def file_opener():
   input = filedialog.askopenfile(initialdir="/")
   a=input.name
   machine(a)
   '''for i in input:
      print(i)'''
# Button label
x = Button(base, text ='Upload the .csv file', command = lambda:file_opener())
x.pack()
mainloop()

