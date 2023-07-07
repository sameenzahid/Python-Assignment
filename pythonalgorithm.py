# Logistic Regressio
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# import matplotlib.pyplot as plt
# import seaborn as sns
# import math
# dataset = pd.read_csv("car.csv")
# print(dataset.head())
# x = dataset['enginesize']
# y = dataset['price']
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_test.shape)
# model = LinearRegression()
# model.fit(x_train.values.reshape(-1,1), y_train)
# print(model.coef_)
# print(model.intercept_)
# y_pred= model.predict(x_test.values.reshape(-1,1))
# mse= mean_squared_error(y_test,y_pred)
# print(mse)
# rmse= math.sqrt(mse)
# print(rmse)
# mae= mean_absolute_error(y_test, y_pred)
# print(mae)
# r2= r2_score(y_test, y_pred)
# print(r2)
# plt.scatter(y_test, y_pred)
# plt.xlabel('Actual')
# plt.ylabel('Predicted')
# plt.show()
# sns.regplot(x=x, y=y, ci=None, color='blue')

# Decisio Tree
# from sklearn.datasets import load_iris
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import train_test_split
# X,y = load_iris(return_X_y=True)
# X_train, X_test, y_train, y_test = train_test_split(X , y , test_size=0.4, random_state=52)
# clr=DecisionTreeClassifier()
# clr.fit(X_train, y_train)
# pred = clr.predict(X_test)
# print(pred)
# accu = clr.score(X_test, y_test)
# print("Accuracy" , accu*100)

# Logistic Regression
# from sklearn.datasets import load_iris
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# X,y = load_iris(return_X_y=True)
# X_train, X_test, y_train, y_test = train_test_split(X , y , test_size=0.4, random_state=52)
# clr=LogisticRegression()
# clr.fit(X_train, y_train)
# pred = clr.predict(X_test)
# print(pred)
# accu = clr.score(X_test, y_test)
# print("Accuracy" , accu*100)

# Random Forest
# from sklearn.datasets import load_iris
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# X,y = load_iris(return_X_y=True)
# X_train, X_test, y_train, y_test = train_test_split(X , y , test_size=0.4, random_state=52)
# clr=RandomForestClassifier()
# clr.fit(X_train, y_train)
# pred = clr.predict(X_test)
# print(pred)
# accu = clr.score(X_test, y_test)
# print("Accuracy" , accu*100)

# Naive Bayes
# from sklearn.datasets import load_iris
# from sklearn.naive_bayes import GaussianNB
# from sklearn.model_selection import train_test_split
# X,y = load_iris(return_X_y=True)
# X_train, X_test, y_train, y_test = train_test_split(X , y , test_size=0.4, random_state=52)
# clr=GaussianNB()
# clr.fit(X_train, y_train)
# pred = clr.predict(X_test)
# print(pred)
# accu = clr.score(X_test, y_test)
# print("Accuracy" , accu*100)

# KNeighbors
# from sklearn.datasets import load_iris
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.model_selection import train_test_split
# X,y = load_iris(return_X_y=True)
# X_train, X_test, y_train, y_test = train_test_split(X , y , test_size=0.4, random_state=52)
# clr=KNeighborsClassifier(n_neighbors=10)
# clr.fit(X_train, y_train)
# pred = clr.predict(X_test)
# print(pred)
# accu = clr.score(X_test, y_test)
# print("Accuracy" , accu*100)

# from sklearn.datasets import load_iris
# from sklearn import svm
# from sklearn.model_selection import train_test_split
# X,y = load_iris(return_X_y=True)
# X_train, X_test, y_train, y_test = train_test_split(X , y , test_size=0.4, random_state=52)
# clr=svm.SVC()
# clr.fit(X_train, y_train)
# pred = clr.predict(X_test)
# print(pred)
# accu = clr.score(X_test, y_test)
# print("Accuracy" , accu*100)
















