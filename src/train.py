from sklearn.datasets import load_iris #getting the data
iris = load_iris()
X = iris.data      # shape (150, 4)
y = iris.target    # shape (150,)
#print(iris.feature_names, iris.target_names)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#split data 80/20 into train/test
#random_state=42 ensures the shuffle is deterministic for reproducible results

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(random_state=42)
#random_state=42 ensures reproducible results

model.fit(X_train, y_train) #training the model

y_pred = model.predict(X_test) #predict with model, then compare the first 5
#print("Predictions:", y_pred[:5])
#print("True labels:", y_test[:5])

from sklearn.metrics import accuracy_score #show how accurate the model is
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

from sklearn.metrics import confusion_matrix #show the confusion matrix
cm = confusion_matrix(y_test, y_pred) #> pred, v true
print("Confusion Matix:\n",cm)

import pandas as pd
import matplotlib.pyplot as plt

data = {
    'True (below) & Predictions (right)': ['setosa','versicolor','virginica'],
    'setosa': [cm[0,0],cm[1,0],cm[2,0]],
    'versicolor': [cm[0,1],cm[1,1],cm[2,1]],
    'virginica': [cm[0,2],cm[1,2],cm[2,2]]
}

df = pd.DataFrame(data)
#print(df)

# Create a figure and axis
fig, ax = plt.subplots(figsize=(10, 5))

# Hide axes
ax.axis('tight')
ax.axis('off')

# Create the table and add it to the axis
table = ax.table(cellText=df.values, 
                 colLabels=df.columns, 
                 loc='center', 
                 cellLoc='center',
                 colColours=['#f2f2f2']*len(df.columns))

# Style the table
table.auto_set_font_size(True)
table.scale(1.2, 2)  # Adjust table size (x,y)

#saving the file to outputs
save_directory = "../outputs"  
filename = "confusion_matrix.png"

import os

save_directory = "../outputs"  #file path
filename = "confusion_matrix.png" #file name

# Create outputs if it doesn't exist
os.makedirs(save_directory, exist_ok=True)

# Joining the directory and filename to create the full path
full_path = os.path.join(save_directory, filename)

# Savivg the figure to the specified path
plt.savefig(full_path, 
            bbox_inches='tight',     # Triming whitespace around the table
            #dpi=300,              # Higher DPI for better quality
            transparent=False)       # Set to True if you want transparent background

plt.close() #close plot