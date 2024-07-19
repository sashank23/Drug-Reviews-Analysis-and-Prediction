import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# Load your training and testing datasets using pd.read_csv
train_data = pd.read_csv('/content/Drugtrain.csv')
test_data = pd.read_csv('/content/Drugtest.csv')

# Handle missing values (you can choose a suitable strategy)
train_data = train_data.dropna()
test_data = test_data.dropna()

# Select features and target column
X_train = train_data[['drugName', 'review', 'rating']]  # Include relevant columns
y_train = train_data['rating']  # Assuming 'target_column' is your target variable
X_test = test_data[['drugName', 'review', 'rating']]  # Include relevant columns
y_test = test_data['rating']  # Assuming 'target_column' is your target variable

# Encode categorical variables with OrdinalEncoder
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X_train_encoded = encoder.fit_transform(X_train)
X_test_encoded = encoder.transform(X_test)
# Define and fit the SVM Classifier model
svm_model = SVC()
svm_model.fit(X_train_encoded, y_train)
# Evaluate the SVM Classifier model
svm_train_accuracy = svm_model.score(X_train_encoded, y_train)
svm_test_accuracy = svm_model.score(X_test_encoded, y_test)
print("SVM Classifier Train Accuracy:", svm_train_accuracy)
print("SVM Classifier Test Accuracy:", svm_test_accuracy)
# Calculate and print Mean Squared Error (MSE) for SVM Classifier
svm_train_pred = svm_model.predict(X_train_encoded)
svm_test_pred = svm_model.predict(X_test_encoded)
svm_train_mse = mean_squared_error(y_train, svm_train_pred)
svm_test_mse = mean_squared_error(y_test, svm_test_pred)
print(f'SVM Classifier Train MSE: {svm_train_mse:.3f}')
print(f'SVM Classifier Test MSE: {svm_test_mse:.3f}')
# Calculate and print R2 Score for SVM Classifier
svm_train_r2 = r2_score(y_train, svm_train_pred)
svm_test_r2 = r2_score(y_test, svm_test_pred)
print(f'SVM Classifier Train R2 Score: {svm_train_r2:.3f}')
print(f'SVM Classifier Test R2 Score: {svm_test_r2:.3f}')
# Plotting the scatter plot of predicted vs actual values for training data
plt.scatter(y_train, svm_train_pred)
plt.xlabel("Actual Ratings")
plt.ylabel("Predicted Ratings")
plt.title("SVM Classifier - Training Data Scatter Plot")
plt.show()
# Plotting the scatter plot of predicted vs actual values for testing data
plt.scatter(y_test, svm_test_pred)
plt.xlabel("Actual Ratings")
plt.ylabel("Predicted Ratings")
plt.title("SVM Classifier - Testing Data Scatter Plot")
plt.show()
# Define and fit the Perceptron model
perceptron_model = Perceptron()
perceptron_model.fit(X_train_encoded, y_train)
# Evaluate the Perceptron model
perceptron_train_accuracy = perceptron_model.score(X_train_encoded, y_train)
perceptron_test_accuracy = perceptron_model.score(X_test_encoded, y_test)
print("Perceptron Train Accuracy:", perceptron_train_accuracy)
print("Perceptron Test Accuracy:", perceptron_test_accuracy)
# Calculate and print Mean Squared Error (MSE) for Perceptron
perceptron_train_pred = perceptron_model.predict(X_train_encoded)
perceptron_test_pred = perceptron_model.predict(X_test_encoded)
perceptron_train_mse = mean_squared_error(y_train, perceptron_train_pred)
perceptron_test_mse = mean_squared_error(y_test, perceptron_test_pred)
print(f'Perceptron Train MSE: {perceptron_train_mse:.3f}')
print(f'Perceptron Test MSE: {perceptron_test_mse:.3f}')
# Calculate and print R2 Score for Perceptron
perceptron_train_r2 = r2_score(y_train, perceptron_train_pred)
perceptron_test_r2 = r2_score(y_test, perceptron_test_pred)
print(f'Perceptron Train R2 Score: {perceptron_train_r2:.3f}')
print(f'Perceptron Test R2 Score: {perceptron_test_r2:.3f}')
# Plotting the scatter plot of predicted vs actual values for training data
plt.scatter(y_train, perceptron_model.predict(X_train_encoded))
plt.xlabel("Actual Ratings")
plt.ylabel("Predicted Ratings")
plt.title("Perceptron - Training Data Scatter Plot")
plt.show()
# Plotting the scatter plot of predicted vs actual values for testing data
plt.scatter(y_test, perceptron_model.predict(X_test_encoded))
plt.xlabel("Actual Ratings")
plt.ylabel("Predicted Ratings")
plt.title("Perceptron - Testing Data Scatter Plot")
plt.show()

# Define and fit the KNN Classifier model
knn_model = KNeighborsClassifier()
knn_model.fit(X_train_encoded, y_train)
# Evaluate the KNN Classifier model
knn_train_accuracy = knn_model.score(X_train_encoded, y_train)
knn_test_accuracy = knn_model.score(X_test_encoded, y_test)
print("KNN Classifier Train Accuracy:", knn_train_accuracy)
print("KNN Classifier Test Accuracy:", knn_test_accuracy)
# Calculate and print Mean Squared Error (MSE) for KNN Classifier
knn_train_pred = knn_model.predict(X_train_encoded)
knn_test_pred = knn_model.predict(X_test_encoded)
knn_train_mse = mean_squared_error(y_train, knn_train_pred)
knn_test_mse = mean_squared_error(y_test, knn_test_pred)
print(f'KNN Classifier Train MSE: {knn_train_mse:.3f}')
print(f'KNN Classifier Test MSE: {knn_test_mse:.3f}')
# Calculate and print R2 Score for KNN Classifier
knn_train_r2 = r2_score(y_train, knn_train_pred)
knn_test_r2 = r2_score(y_test, knn_test_pred)
print(f'KNN Classifier Train R2 Score: {knn_train_r2:.3f}')
print(f'KNN Classifier Test R2 Score: {knn_test_r2:.3f}')
# Plotting the scatter plot of predicted vs actual values for training data
plt.scatter(y_train, knn_train_pred)
plt.xlabel("Actual Ratings")
plt.ylabel("Predicted Ratings")
plt.title("KNN Classifier - Training Data Scatter Plot")
plt.show()
# Plotting the scatter plot of predicted vs actual values for testing data
plt.scatter(y_test, knn_test_pred)
plt.xlabel("Actual Ratings")
plt.ylabel("Predicted Ratings")
plt.title("KNN Classifier - Testing Data Scatter Plot")
plt.show()
