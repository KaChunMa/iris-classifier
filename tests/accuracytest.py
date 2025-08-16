from sklearn.datasets import load_iris #getting the data
iris = load_iris()
X = iris.data      # shape (150, 4)
y = iris.target    # shape (150,)
print(iris.feature_names, iris.target_names)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#split data 80/20 into train/test
#random_state=42 ensures the shuffle is deterministic for reproducible results

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(random_state=42)
#random_state=42 ensures reproducible results

model.fit(X_train, y_train) #training the model

y_pred = model.predict(X_test) #predict with model, then compare the first 5
print("Predictions:", y_pred[:5])
print("True labels:", y_test[:5])

from sklearn.metrics import accuracy_score #show how accurate the model is
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

if accuracy > 0.9:
    print("The model is very good accuracy-wise.")
else:
    print("The model is not good enough accuracy-wise.")