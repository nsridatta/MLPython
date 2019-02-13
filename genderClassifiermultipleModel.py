from sklearn import tree
from sklearn import neighbors
from sklearn import svm
from sklearn import gaussian_process
from sklearn import naive_bayes

#[height, weight, shoe size]
X = [[181, 80, 44],[177,78,43],[175,77,44],[165,65,38],[166,65,39],
	[179,80,46],[182,85,45],[155,50,35],[160,65,38],[166,65,40],[188,89,45]]
Y = ['male','male','male','female','female','male','male','female','female','male','male']

#Decision Tree Classifier model
Dclf = tree.DecisionTreeClassifier()

#KNeighborsClassifier model
kclf = neighbors.KNeighborsClassifier()

#SVC model
svcclf = svm.SVC(gamma = 'auto')

#GaussianProcessClassifier model
gclf = gaussian_process.GaussianProcessClassifier()

#GaussianNB model
gnbclf = naive_bayes.GaussianNB()

#Fit the model and train it 
Dclf = Dclf.fit(X,Y)
kclf = kclf.fit(X,Y)
svcclf = svcclf.fit(X,Y)
gclf = gclf.fit(X,Y)
gnbclf = gnbclf.fit(X,Y)

#Test or predict output for a random sample of data set
prediction = Dclf.predict([[183,70,40]])
print('DecisionTreeClassifier ',prediction)

#Test or predict output for a random sample of data set
kprediction = kclf.predict([[167,60,41]])
print('KNeighborsClassifier ', kprediction)

#Test or predict output for a random sample of data set
svcprediction = svcclf.predict([[167,60,41]])
print('SVC ', svcprediction)

#Test or predict output for a random sample of data set
gprediction = gclf.predict([[167,60,41]])
print('GaussianProcessClassifier ',gprediction)

#Test or predict output for a random sample of data set
gnprediction = gclf.predict([[167,60,41]])
print('GaussianNB ',gnprediction)