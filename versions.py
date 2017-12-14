from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'supervised_class']
dataset = pandas.read_csv(url, names=names)

#print(dataset.shape)
#print(dataset.head(10))
#print(dataset.groupby('supervised_class').size())


# Split-out validation dataset
seed = 10
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
val_partition = 0.20
X_train, X_val, Y_train, Y_val = model_selection.train_test_split(X, Y, test_size=val_partition, random_state=seed)


# Spot Check Algorithms
engines = []
engines.append(('KNN', KNeighborsClassifier()))
engines.append(('NB', GaussianNB()))
engines.append(('RF', RandomForestClassifier()))
#engines.append(('SVM', SVC()))
engines.append(('LR', LogisticRegression()))
#engines.append(('NN', MLPClassifier()))
# evaluate each model in turn
results = []
names = []
no_of_splits=10
scoring = 'accuracy'
for name, model in engines:
	kfold_cross_val = model_selection.KFold(no_of_splits, random_state=seed)
	cross_val_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold_cross_val, scoring=scoring)
	results.append(cross_val_results)
	names.append(name)
	msg = "%s: %f" % (name, 100.0*cross_val_results.mean())
	print(msg + " %")

# Compare Algorithms
#fig = plt.figure()
#fig.suptitle('Algorithm Comparison')
#ax = fig.add_subplot(111)
#plt.boxplot(results)
#ax.set_xticklabels(names)
#plt.show()



# Make predictions on validation dataset

knn= KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_val)
print(100*(accuracy_score(Y_val, predictions)))

