# imports
import numpy as np
import csv
from sklearn import tree
from sklearn import preprocessing
import matplotlib.pyplot as plt


def read_csv_to_nparray(file_name):
    # Initialize an empty list to store the data
    data = []

    # Read the CSV file
    with open(file_name, 'r') as file:
        csv_reader = csv.reader(file)
        # Skip the header if present
        header = next(csv_reader, None)
        # Iterate over the remaining rows
        for row in csv_reader:
            # Append each row to the data list
            data.append(row)

    # Convert the data list to a NumPy array
    np_array = np.array(data)
    return np_array


def decision_tree_builder(dataset):
    x = dataset[:, 0:10]
    y = dataset[:, 10]
    le_list = []  # List to store individual label encoders
    le = preprocessing.LabelEncoder()
    
    for i in range(10):
        x[:, i] = le.fit_transform(x[:, i])
        
    le_list.append(le)
    le_y = preprocessing.LabelEncoder()
    y = le_y.fit_transform(y)

    dtc = tree.DecisionTreeClassifier(criterion="entropy", splitter="best")
    dtc.fit(x, y)

    fig, ax = plt.subplots(figsize=(10, 10))
    # Increase fontsize value as needed
    tree.plot_tree(dtc, ax=ax, fontsize=10)
    plt.ion()
    plt.show()

    # Classification
    input_values = input(
        'Enter values separated by a space to predict their output: ')
    input_array = input_values.split()

    # Convert input string array to numerical array
    input_numerical_array = np.array(input_array).reshape(1, -1)

    for i, le in enumerate(le_list):
        input_numerical_array[:, i] = le.transform(input_numerical_array[:, i])

    y_pred = dtc.predict(input_numerical_array)
    predicted_output = le_y.inverse_transform(y_pred)
    print("Predicted output: ", predicted_output)


# Call the decision_tree_builder function
print("Decision Tree Builder")
file_name = input("Enter dataset file: ")
dataset = read_csv_to_nparray(file_name)
decision_tree_builder(dataset)

