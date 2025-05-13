
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset 
df = pd.read_csv("GLDS-250_GMetagenomics_Metaphlan-taxonomy.tsv", sep="\t", skiprows=1, index_col=0)

# Filter for species level taxa 
species_level = [idx for idx in df.index if "|s__" in idx]
df = df.loc[species_level]

# Transpose the dataset so that rows are samples and columns are species
df = df.transpose()

# Assign binary labels 
labels = [1 if "FLT" in sample else 0 for sample in df.index]
df["label"] = labels

# Prepare features (X) and target (y)
X = df.drop(columns="label").apply(pd.to_numeric, errors='coerce').fillna(0)
y = df["label"]

# Standardize the feature values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)

# Train the decision tree classifier
clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(X_train, y_train)

# Evaluate the model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_mat = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print("Confusion Matrix:\n", conf_mat)
print("Classification Report:\n", report)

# Visualize the decision tree
plt.figure(figsize=(16, 10))
plot_tree(clf, filled=True, feature_names=X.columns, class_names=["Ground", "Flight"], rounded=True)
plt.title("Decision Tree Classifier on GLDS-250 Data")
plt.savefig("decision_tree_visualization.png", dpi=300, bbox_inches='tight')
plt.show()
