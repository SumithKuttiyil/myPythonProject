#Predict the age of Abalone using K-Nearest Neighbors Classification Algorithm

import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors
npArray=np.loadtxt('abalone.data')
train_data_row=int(round(np.shape(npArray)[0]*0.7))
test_data_row=np.shape(npArray)[0]-int(round(np.shape(npArray)[0]*0.7))
X,y_train=npArray[:train_data_row, :8],npArray[:train_data_row,8]
nbr=neighbors.KNeighborsClassifier(15, weights='uniform')
nbr.fit(X,y_train)
y_predict=nbr.predict(npArray[test_data_row:,:8])
y_actual=npArray[test_data_row:,8]
plt.scatter(y_actual, y_predict)
plt.plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], 'k--')
plt.xlabel('Measured')
plt.ylabel('Predicted')
plt.title('K-Nearest Neighbors Classification')
plt.show()

