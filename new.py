import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
data = pd.read_csv('C:\machinelearning\data\data.csv')
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

data = pd.read_csv('C:\machinelearning\data\data.csv')
data.drop(['id'], axis=1, inplace=True)
#data.drop(['Unnamed: 32'], axis=1 , inplace=True)
data['diagnosis'] = (data['diagnosis'] == 'M').astype(int)
dataProcessed = data.drop(['diagnosis'], axis=1)
dropList = ['radius_mean','perimeter_mean', 'compactness_mean', 'concave points_mean', 'radius_worst','perimeter_worst', 'texture_worst','perimeter_se','radius_se','compactness_se','concave points_se','compactness_worst','concave points_worst', 'area_worst', 'concavity_mean']
dataProcessed = dataProcessed.drop(dropList, axis=1)
def outlierLimit(column):
    q1, q3 = np.nanpercentile(column, [25, 75])
    iqr = q3 - q1
    upLimit = q3 + 1.5 * iqr
    loLimit = q1 - 1.5 * iqr
    return upLimit, loLimit
    for column in dataProcessed.columns:
        if dataProcessed[column].dtype != 'object':
            upLimit, loLimit = outlierLimit(dataProcessed[column])
            dataProcessed[column] = np.where((dataProcessed[column] > upLimit) | (dataProcessed[column] < loLimit), np.nan, dataProcessed[column])
    iterative_imputer
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=4)
dataProcessed.iloc[:, :] = imputer.fit_transform(dataProcessed)

Y = data['diagnosis']
X = dataProcessed
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.7, test_size=0.3, random_state=50)
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score 
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state=50)
lr.fit(x_train, y_train)
yPredict = lr.predict(x_test)
print('Accuracy: {}'.format(accuracy_score(y_test, yPredict)))
print('Recall: {}'.format(recall_score(y_test, yPredict)))

def drawConfusionMatrix(confusion):
    groups = ['TN','FP','FN','TP']
    counts = ['{0:0.0f}'.format(value) for value in confusion.flatten()]
    labels = np.asarray([f'{v1}\n{v2}' for v1, v2 in zip(groups, counts)]).reshape(2, 2)
    plt.figure(figsize=(10,10))
    sns.heatmap(confusion, annot=labels, cmap='Blues', cbar=False, fmt='')
    plt.show()
drawConfusionMatrix(confusion_matrix(y_test, yPredict))




