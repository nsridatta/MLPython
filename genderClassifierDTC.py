from sklearn import tree

#[height, weight, shoe size]
X = [[181, 80, 44],[177,78,43],[175,77,44],[165,65,38],[166,65,39],
	[179,80,46],[182,85,45],[155,50,35],[160,65,38],[166,65,40],[188,89,45]]
Y = ['male','male','male','female','female','male','male','female','female','male','male']

#Classficication Model
clf = tree.DecisionTreeClassifier()
#Fitting data to the model
clf = clf.fit(X,Y)
#Predict the result using test set values
prediction = clf.predict([[167,60,41]])
#display the output
print(prediction)
