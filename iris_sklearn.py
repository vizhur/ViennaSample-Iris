# Please make sure scikit-learn is included the conda_dependencies.yml file.
import pickle
import sys
import os

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import numpy as np
from sklearn.metrics import confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle

from azureml.sdk import data_collector
from azureml.dataprep.package import run

# initialize the logger
run_logger = data_collector.current_run() 

# create the outputs folder
os.makedirs('./outputs', exist_ok=True)

print('Python version: {}'.format(sys.version))
print()

# load Iris dataset
iris = run('iris.dprep', dataflow_idx=0)
print ('Iris dataset shape: {}'.format(iris.shape))


# load features and labels
X, Y = iris[['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']].values, iris['Species'].values
X, Y = shuffle(X, Y)
X_train, Y_train = X[:-30,:], Y[:-30]
X_test, Y_test = X[-30:,:], Y[-30:]

# change regularization rate and you will likely get a different accuracy.
reg = 0.01
# load regularization rate from argument if present
if len(sys.argv) > 1:
    reg = float(sys.argv[1])

print("Regularization rate is {}".format(reg))

# log the regularization rate
run_logger.log("Regularization Rate", reg)

# train a logistic regression model
clf1 = LogisticRegression(C=1/reg).fit(X, Y)
print (clf1)

accuracy = clf1.score(X, Y)
print ("Accuracy is {}".format(accuracy))

# log accuracy
run_logger.log("Accuracy", accuracy)

print("")
print("==========================================")
print("Serialize and deserialize using the outputs folder.")
print("")

# serialize the model on disk in the special 'outputs' folder
print ("Export the model to model.pkl")
f = open('./outputs/model.pkl', 'wb')
pickle.dump(clf1, f)
f.close()

# load the model back from the 'outputs' folder into memory
print("Import the model from model.pkl")
f2 = open('./outputs/model.pkl', 'rb')
clf2 = pickle.load(f2)

# predict a new sample
X_new = [[3.0, 3.6, 1.3, 0.25]]
print ('New sample: {}'.format(X_new))
pred = clf2.predict(X_new)
print('Predicted class is {}'.format(pred))


Y_hat = clf1.predict(X_test)
labels = ['Iris-virginica', 'Iris-versicolor', 'Iris-setosa']
cm = confusion_matrix(Y_test, Y_hat, labels)

fig = plt.figure(figsize=(6,4), dpi=75)
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Reds)
plt.colorbar()
tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, rotation=45)
plt.yticks(tick_marks, labels)
plt.xlabel("Predicted Species")
plt.ylabel("True Species")
fig.savefig('./outputs/cm.png', bbox_inches='tight')
