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


from sklearn.metrics import accuracy_score #show how accurate the model is
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


import os

# Creates outputs folder if it doesn't exist
os.makedirs("../outputs", exist_ok=True)


import joblib

joblib.dump(model,"../outputs/model.pkl") #outputing the model


from sklearn.metrics import confusion_matrix #generate the confusion matrix
confusion_matrix = confusion_matrix(y_test, y_pred)

from sklearn.metrics import ConfusionMatrixDisplay #display the confusion matrix
import matplotlib.pyplot as plt

cm_display = ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ["setosa","versicolor","virginica"])
cm_display.plot()
plt.savefig("../outputs/confusion_matrix.png", dpi=300, bbox_inches='tight') #outputing the matrix
plt.show() #plt.show clears the figure in some environment some save the figure first