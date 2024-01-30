import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from chefboost import Chefboost as chef
from sklearn.metrics import mean_squared_error, roc_auc_score, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from tabulate import tabulate
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# ____________________________________________________ Data Describtion _____________________________________________________________#
print("________________________________________________________      EDA      _______________________________________________________ \n")

pd.set_option('display.max_rows', None, 'display.max_columns', None)
data = pd.read_csv("Data.csv")
print("                                                      Description using EDA:\n")
des = data.describe(include='all')
print(tabulate(des,headers='keys', tablefmt='psql'))

print("\n#############################################################################################################################\n")

#1- Show the distribution of the class label(Smoker) and indicate any highlights in the distribution

smoker_counts = data["Smoker"].value_counts()
plt.bar(smoker_counts.index, smoker_counts.values, color=['yellow', 'red'])
plt.xlabel("Smoker")
plt.ylabel("People's Number")
plt.title("Distribution of Smoker Class Label")

for index, value in enumerate(smoker_counts):
    plt.text(index, value + 1, str(value), ha='center', va='bottom')
plt.show()

#2- Show the density plot for the age.

plt.figure(figsize=(12, 8))
sns.kdeplot(data=data, x='Age', fill=True)
plt.xlabel("Age")
plt.ylabel("Density")
plt.title('Density Plot For The Age')
plt.show()

#3- Show the density plot for the BMI.

plt.figure(figsize=(12, 8))
sns.kdeplot(data=data, x='BMI', fill=True)
plt.title('Density Plot For The BMI')
plt.xlabel('BMI')
plt.ylabel('Density')
plt.show()

#4- Visualise the scatterplot of data and split based on Region attribute(after cleaning data).
########################################################## Outliers ######################################################################
upper_limitAge = data['Age'].mean() + 3 * data['Age'].std()
lower_limitAge = data['Age'].mean() - 3 * data['Age'].std()

upper_limitBMI = data['BMI'].mean() + 3 * data['BMI'].std()
lower_limitBMI = data['BMI'].mean() - 3 * data['BMI'].std()

upper_limitNoChildren = data['No. Childred'].mean() + 3 * data['No. Childred'].std()
lower_limitNoChildren = data['No. Childred'].mean() - 3 * data['No. Childred'].std()

upper_limitIcharges = data['Insurance Charges'].mean() + 3 * data['Insurance Charges'].std()
lower_limitIcharges = data['Insurance Charges'].mean() - 3 * data['Insurance Charges'].std()

data = data.loc[(data['Age'] > lower_limitAge) & (data['Age'] < upper_limitAge)]
data = data.loc[(data['BMI'] > lower_limitBMI) & (data['BMI'] < upper_limitBMI)]
data = data.loc[(data['No. Childred'] > lower_limitNoChildren) & (data['No. Childred'] < upper_limitNoChildren)]
data = data.loc[(data['Insurance Charges'] > lower_limitIcharges) & (data['Insurance Charges'] < upper_limitIcharges)]
########################################################## Replacements ##################################################################
data['Age'] = data['Age'].replace(to_replace=0, value=data['Age'].mean())
data['BMI'] = data['BMI'].replace(to_replace=0, value=data['BMI'].mean())
data['No. Childred'] = data['No. Childred'].replace(to_replace=0, value=data['No. Childred'].mean())
data['Insurance Charges'] = data['Insurance Charges'].replace(to_replace=0, value=data['Insurance Charges'].mean())
data['Region'] = (data['Region'] == 'north').astype(int)  # South --> 0, North --> 1
data['Gender'] = (data['Gender'] == 'male').astype(int)   # No --> 0, Yes --> 1
data['Smoker'] = (data['Smoker'] == 'yes').astype(int)    # No --> 0, Yes --> 1

# Visualise the scatterplot of data and split based on Region attribute.
x_col = "BMI"  # Replace with the desired column for the x-axis
y_col = "Age"  # Replace with the desired column for the y-axis
region_col = "Region"  # Replace with the column name containing the Region attribute

# For Region: South --> 0, North --> 1

plt.figure(figsize=(12, 8))
sns.scatterplot(x=x_col, y=y_col, hue=region_col, data=data, palette='viridis')
plt.xlabel(x_col)
plt.ylabel(y_col)
plt.title("The Scatterplot Of Data Split by The Region")
plt.legend(title='Region')
plt.show()

#5- Split the dataset into training(80%) and test(20%).
print("\n################################################      Splitting Dataset     #################################################\n")

features = ['Age', 'Gender', 'BMI', 'Region','No. Childred','Insurance Charges']
target = 'Smoker'

# Split the dataset into features(X) and target(y).
X = data[features]
y = data[target]

# Split the dataset into training(80%) and test(20%).
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(" The data shape is: ", data.shape)

# Print the shapes of the resulting sets
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
data_train = pd.concat([X_train, y_train], axis=1)
data_test = pd.concat([X_test, y_test], axis=1)

print("\n_______________________________________________      Training Data     ____________________________________________________________\n")
print(tabulate(data_train.describe(), headers='keys', tablefmt='psql'))

print("\n_______________________________________________      Testing Data     ____________________________________________________________\n")
print(tabulate(data_test.describe(), headers='keys', tablefmt='psql'))
print("#######################################################################################################################################")
#############################################################################################################################################
#############################################################################################################################################
#############################################################################################################################################
#############################################################################################################################################
#############################################################################################################################################
#############################################################################################################################################
#############################################################################################################################################
#############################################################################################################################################
# Tasks

#1- KNN with values of k: (3, 5, 7).
# print("\n______________________________________________________      KNN      ______________________________________________________\n")
k_values = [3, 5, 7]
models = [KNeighborsClassifier(n_neighbors=k) for k in k_values]

for i, model in enumerate(models):
    model.fit(X_train, y_train)

results = []
for i, model in enumerate(models):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    roc_auc = roc_auc_score(y_test, y_pred)
    confusion_mat = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    results.append((k_values[i], mse, rmse, roc_auc, confusion_mat, accuracy))

for result in results:
    k, mse, rmse, roc_auc, confusion_mat, accuracy = result

    print("K:"f"{k:3d}")
    print("\nAccuracy: "f"{accuracy:2f}\n")
    print("MSE: "f"{mse:2f}  |  RMSE: "f"{rmse:2f}  |  ROC/AUC: "f"{roc_auc:2f}\n")
    print("Confusion Matrix: "f"{confusion_mat}")
    print("____________________________________________________________________________________________________________________\n")

print("##############################################################################################################################")

#2- Decision Trees(C4.5 Algorithm).
print("\n______________________________________________________ Decision Tree ______________________________________________________\n")
# Split the dataset into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Combine features and labels for training data.
train_data = pd.concat([X_train, pd.DataFrame({'Smoker': y_train})], axis=1)

# Save combined DataFrame to a file.
train_data.to_csv('train_data.csv', index=False)

# Read training data from the file.
train_data = pd.read_csv('train_data.csv')
train_data = train_data.rename(columns={"Smoker":"Decision"})

print("Training data columns: ", train_data.columns)
print("Training data sample: \n", train_data.head())

# Define the configuration for Chefboost(C4.5 Algorithm).
config = {'algorithm': 'C4.5', 'enableGBM': False, 'criterion': 'entropy'}

# Combine features and labels for testing data.
test_data = pd.concat([X_test, pd.DataFrame({'Decision': y_test})], axis=1)

# Save combined DataFrame to a file.
test_data.to_csv('test_data.csv', index=False)

# Read testing data from the file.
test_data = pd.read_csv('test_data.csv')

print("Testing data columns: ", test_data.columns)
print("Testing data sample:\n", test_data.head())
############################################################## Decision tree Training ##################################################
# Smoker ==> Target Label

if __name__ == '__main__':      # Train the decision tree.
    model = chef.fit(train_data, config, target_label='Decision')

    # Convert testing data(DataFrame) to a list of dictionaries.
    test_data_list = test_data.to_dict(orient='records')

    true_labels = []
    predicted_labels = []

    # for data_point in test_data_list:
    #     # Exclude decision attribute before making predictions.
    #     data_point_without_decision = {key: value for key, value in data_point.items() if key != 'Decision'}
    #
    #     print(data_point_without_decision)
    #     prediction = chef.predict(model, data_point_without_decision)  # Prediction for the current data point.
    #
    #     actual_label = data_point['Smoker']  # Access the actual label from the data point.
    #
    #     true_labels.append(actual_label)  # Append true and predicted labels for confusion matrix
    #     predicted_labels.append(prediction)

    # accuracy = accuracy_score(true_labels, predicted_labels) * 100
    # print(f"Accuracy: {accuracy:.2f}%")
    #
    # # Calculate and print confusion matrix
    # conf_matrix = confusion_matrix(true_labels, predicted_labels)
    # print("Confusion Matrix:")
    # print(conf_matrix)
print("##############################################################################################################################")
# Naive Bayes(NB)
print("\n______________________________________________________ Naive Bayes(NB) _____________________________________________________\n")
model = GaussianNB()
model.fit(X_train, y_train)         # Train the model.

y_pred = model.predict(X_test)      # Predictions on the testing set.
y_pred = model.predict(X_test)      # Evaluate model's performance.

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
roc_auc = roc_auc_score(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)

results.append((k_values[i], mse, rmse, roc_auc, confusion_mat))
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy: "f"{accuracy:2f}\n")
print("MSE: "f"{mse:2f}  |  RMSE: "f"{rmse:2f}  |  ROC/AUC: "f"{roc_auc:2f}\n")
print("Confusion Matrix: "f"{confusion_mat}")

print("##############################################################################################################################")
#4- Artificial Neural Networks(ANN)
print("\n__________________________________________ Artificial Neural Networks(ANN) ______________________________________________\n")

target = "Smoker"       # Smoker is the target.
X_train, X_test, y_train, y_test = train_test_split(data.drop(target, axis=1), data[target], test_size=0.2)

model = Sequential()
model.add(Dense(units = 10, activation = 'sigmoid', input_dim = X_train.shape[1]))
model.add(Dense(units = 1, activation = 'sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=500, batch_size=32, verbose=2)       # Train the model.

loss, accuracy = model.evaluate(X_test, y_test)         # Evaluate the model on testing set.
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

print("\n\nDone by Laith Ghnemat 1200610.")