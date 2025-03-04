import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, accuracy_score, zero_one_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
#%%
############### Pre-processing #####################3
train_df = pd.read_csv('audit_train.csv')
test_df = pd.read_csv('audit_test.csv')

train_df = train_df.replace('NUH', np.nan)          # Remove rows with missing values and non-numeric values
test_df = test_df.replace('NUH', np.nan)
train_df = train_df.apply(pd.to_numeric, errors='coerce')    # Convert all columns to numeric
test_df = test_df.apply(pd.to_numeric, errors='coerce')
train_df = train_df.dropna()                                 # Remove rows with missing values
test_df = test_df.dropna()
# print("\nTraining set shape after cleaning:", train_df.shape)
# print("Test set shape after cleaning:", test_df.shape)

X_train = train_df.drop('Risk', axis=1)
y_train = train_df['Risk']
X_test = test_df.drop('Risk', axis=1)
y_test = test_df['Risk']
feature_names = list(X_train.columns)            # Convert feature names to list

# Q1 Classification Tree
#  plot the tree and report the training error.
#  Test the performance on the test dataset and report the confusion matrix
print('----------Q1-------------\n')
tree = DecisionTreeClassifier(random_state=14)
tree.fit(X_train, y_train)
plt.figure(figsize=(20,10))
plot_tree(tree, feature_names=feature_names, class_names=['0', '1'], filled=True)
plt.savefig('Problem3Q1.jpg',dpi=500)
y_pred = tree.predict(X_train)
train_error = zero_one_loss(y_train, y_pred)
print(f"Tree Training error: {train_error:.3f}\n")

# Test performance
test_pred = tree.predict(X_test)
test_error = zero_one_loss(y_test,test_pred)
print(f'Tree Test error:{test_error:.3f}\n')
conf_matrix = confusion_matrix(y_test, test_pred)
print(f"Tree Confusion Matrix:\n")
print(conf_matrix)

# Q2 Pruning using CV
# Plot the train error versus the tree size.
# Plot the pruned tree which has the best train error.
# Report the test error.
print('----------Q2-------------\n')
path = tree.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas

clf_trees = []
size_list = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=14, ccp_alpha=ccp_alpha)
    scores = cross_val_score(clf, X_train, y_train, cv=5)
    clf.fit(X_train, y_train)
    tree_size = clf.tree_.node_count  # the total nums of nodes indicate the size of tree
    clf_trees.append((clf, scores.mean()))
    size_list.append(tree_size)

best_clf_tree, best_score = max(clf_trees, key=lambda x: x[1])
best_clf_tree.fit(X_train, y_train)
train_errors = [1 - score for _, score in clf_trees]
plt.figure(figsize=(10,6))
plt.plot(size_list, train_errors, marker='o', drawstyle="steps-post")
plt.xlabel("Tree size")
plt.ylabel("Training Error")
plt.title("Training Error vs. Tree Size")
plt.savefig('Problem3Q2(1).jpg',dpi=500)

plt.figure(figsize=(20,10))
plot_tree(best_clf_tree, filled=True, feature_names=feature_names)
plt.savefig('Problem3Q2(2).jpg',dpi=500)

y_test_pruned_pred = best_clf_tree.predict(X_test)
test_error_pruned = zero_one_loss(y_test, y_test_pruned_pred)
print(f"Test Error after Pruning: {test_error_pruned}")

# Q3 Random Forest (m=13, n=25)
# Use random forest setting m=13 and n=25. Report the training error.
print('----------Q3-------------\n')
rf = RandomForestClassifier(n_estimators=25, max_features=13, random_state=14)
rf.fit(X_train, y_train)
rf_train_pred = rf.predict(X_train)
rf_train_error = zero_one_loss(y_train, rf_train_pred)
print(f"\nRandom Forest training error (m=13): {rf_train_error:.3f}")

# Q4 Random Forest with different m values
print('----------Q4-------------\n')
m_values = [8, 12, 14, 16, 18]
rf_errors = []
for m in m_values:
    rf = RandomForestClassifier(n_estimators=25, max_features=m, random_state=0)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_train)
    error = zero_one_loss(y_train, rf_pred)
    rf_errors.append(error)

best_m = m_values[np.argmin(rf_errors)]
print(f"\nBest m value: {best_m}")
print(f"Error rates for different m values:")
for m, errorvalue in zip(m_values, rf_errors):
    print(f"m={m}: {errorvalue:.4f}")

best_rf = RandomForestClassifier(n_estimators=25, max_features=best_m, random_state=14)
best_rf.fit(X_train, y_train)
rf_final_pred = best_rf.predict(X_test)
final_conf_matrix = confusion_matrix(y_test, rf_final_pred)
print("\nSelected Random Forest Confusion Matrix:")
print(final_conf_matrix)
print(f'\nSelected Random Forest Test Error:{zero_one_loss(y_test,rf_final_pred)}')

# 5. Comparison summary
print('----------Q5-------------\n')
print("\nComparison Summary:")
print(f"Initial Tree Training Error: {train_error:.4f}")
print(f"Initial Tree Test Error: {test_error:.4f}")
print(f"Pruned Tree Test Error: {test_error_pruned:.4f}")
print(f"Random Forest (m=13) Training Error: {rf_train_error:.4f}")
print(f"Best Random Forest (m={best_m}) Test Error: {zero_one_loss(y_test,rf_final_pred):.4f}")