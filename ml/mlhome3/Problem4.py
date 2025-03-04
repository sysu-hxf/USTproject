import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Apply your method to the Hitters training (Hitters train.csv) data
# Using the predictor variables Years, Hits, RBI, Walks, PutOuts, Runs.
# Report the best test error (Hitters test.csv)
class RegressionTree:
    def __init__(self, min_samples_split=2, max_depth=float('inf'), min_loss_reduction=0):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.min_loss_reduction = min_loss_reduction
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def predict(self, X):
        return np.array([self._predict_one(x, self.tree) for x in X])

    def _build_tree(self, X, y, depth=0):
        if len(y) < self.min_samples_split or depth >= self.max_depth:
            return np.mean(y)

        best_split = self._find_best_split(X, y)
        if best_split['loss_reduction'] < self.min_loss_reduction:
            return np.mean(y)

        left_tree = self._build_tree(X[best_split['left_indices']], y[best_split['left_indices']], depth + 1)
        right_tree = self._build_tree(X[best_split['right_indices']], y[best_split['right_indices']], depth + 1)

        return {
            'feature_index': best_split['feature_index'],
            'threshold': best_split['threshold'],
            'left': left_tree,
            'right': right_tree
        }

    def _find_best_split(self, X, y):
        best_split = {'loss_reduction': -float('inf')}
        for feature_index in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_indices = np.where(X[:, feature_index] <= threshold)[0]
                right_indices = np.where(X[:, feature_index] > threshold)[0]

                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue

                loss_reduction = self._calculate_loss_reduction(y, left_indices, right_indices)
                if loss_reduction > best_split['loss_reduction']:
                    best_split = {
                        'feature_index': feature_index,
                        'threshold': threshold,
                        'left_indices': left_indices,
                        'right_indices': right_indices,
                        'loss_reduction': loss_reduction
                    }
        return best_split

    def _calculate_loss_reduction(self, y, left_indices, right_indices):
        left_loss = np.sum(np.abs(y[left_indices] - np.mean(y[left_indices])))
        right_loss = np.sum(np.abs(y[right_indices] - np.mean(y[right_indices])))
        total_loss = np.sum(np.abs(y - np.mean(y)))
        return total_loss - (left_loss + right_loss)

    def _predict_one(self, x, tree):
        if not isinstance(tree, dict):
            return tree
        if x[tree['feature_index']] <= tree['threshold']:
            return self._predict_one(x, tree['left'])
        else:
            return self._predict_one(x, tree['right'])

# Loading
train_data = pd.read_csv('Hitters_train.csv')
test_data  = pd.read_csv('Hitters_test.csv')
# Separating
X_train = train_data[['Years', 'Hits', 'RBI', 'Walks', 'PutOuts', 'Runs']]
y_train = train_data['Salary']
X_test = test_data[['Years', 'Hits', 'RBI', 'Walks', 'PutOuts', 'Runs']]
y_test = test_data['Salary']
# Cleaning
train_data_cleaned = pd.concat([X_train, y_train], axis=1).dropna()
X_train_cleaned = train_data_cleaned[['Years', 'Hits', 'RBI', 'Walks', 'PutOuts', 'Runs']]
y_train_cleaned = train_data_cleaned['Salary']

test_data_cleaned = pd.concat([X_test, y_test], axis=1).dropna()
X_test_cleaned = test_data_cleaned[['Years', 'Hits', 'RBI', 'Walks', 'PutOuts', 'Runs']]
y_test_cleaned = test_data_cleaned['Salary']

# Convert to Numpy
X_train_cleaned = X_train_cleaned.values
y_train_cleaned = y_train_cleaned.values
X_test_cleaned = X_test_cleaned.values
y_test_cleaned = y_test_cleaned.values

# Modeling
# Here we can choose the max_depth(similar to control the terminal nodes) and threshold
max_depth_list = np.arange(4,11,dtype='float')
min_loss_list  = np.linspace(0,1,10)

loc = 1
error = []
plt.figure()
for depth in max_depth_list:
    test_error_list = []
    for loss_threshold in min_loss_list:
        reg_tree = RegressionTree(min_samples_split=2, max_depth = depth, min_loss_reduction= loss_threshold)
        reg_tree.fit(X_train_cleaned, y_train_cleaned)
        y_test_pred = reg_tree.predict(X_test_cleaned)
        test_error = np.mean(np.abs(y_test_cleaned - y_test_pred))
        test_error_list.append(test_error)
        error.append({'max_depth':depth,
                      'min_loss_threshold':loss_threshold,
                     'test_error':test_error})
    plt.subplot(3,3,loc)
    plt.plot(min_loss_list, test_error_list)
    loc +=1

plt.savefig('Problem4(tuning para).jpg',dpi=400)

min_error_entry = min(error, key=lambda x: x['test_error'])
print("After Tuning:\n")
print(f"Max Depth: {min_error_entry['max_depth']}, Min Loss Threshold: {min_error_entry['min_loss_threshold']}")
print(f"Test Error: {test_error}")

plt.figure()
plt.plot(y_test_cleaned)
plt.plot(y_test_pred,'r')
plt.savefig('Problem4(fit_curve).jpg',dpi=400)
