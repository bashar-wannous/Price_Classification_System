# -*- coding: utf-8 -*-

"""
the used functions

Created on Fri May 17 13:43:51 2024

@author: Bashar
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
import json
from pathlib import Path
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import model_from_json


from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay


classes_dict = {0: "low_cost", 1: "medium_cost", 2: "high_cost", 3: "very_high_cost"}


def check_remove_nulls(df: pd.DataFrame()) -> pd.DataFrame():
    if df.isnull().values.any():
        # True for every row with null

        nan_rows = sum([True for idx, row in df.iterrows() if any(row.isnull())])
        # ratio of rows with nulls

        nan_ratio = nan_rows / len(df)
        print(round(nan_ratio * 100, 3), "% of data rows contains null")
        # drop rows with nulls

        df = df.dropna(how="any", axis=0)
        print(nan_rows, "rows were deleted")
    return df


def check_categorical_exist(df: pd.DataFrame()) -> bool:
    # data explore, numerical or categorical

    col_num_with_categorical = df.select_dtypes(include=["object"]).columns.tolist()
    if not col_num_with_categorical:  # no categorical
        return False
    else:
        return True


def plot_classes_histogram(df_classes_col) -> int:
    # plot histogram to count samples in each class, checking data balance with visualization

    classes_array = np.array(df_classes_col)
    bincounts = np.bincount(classes_array)
    ind = np.nonzero(bincounts)[0]

    fig, ax = plt.subplots()
    ax.bar(ind, bincounts)
    ax.set_xlabel("Price_Range", fontsize=18)
    ax.set_ylabel("Counts", fontsize=18)
    ax.legend()
    plt.show()
    return len(ind)


def plot_two_features(data: pd.DataFrame(), f1: int, f2: int):
    color_dictionarydict = {0: "blue", 1: "orange", 2: "green", 3: "red"}
    gt_labels = np.array(data)[:, -1]
    color = []
    for i in gt_labels:
        color.append(color_dictionarydict[i])
    features = np.array(data)
    plt.scatter(features[:, f1], features[:, f2], c=color[:], s=5, alpha=0.1)
    plt.show()


def display_confusion_matrix(cm: np.array, labels_array):
    cm_display = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=labels_array
    )
    cm_display.plot()


def get_ffnn_model(
    # define feed forward neural network, optimized with adam, one hidden layer
    num_classes,
):

    model = Sequential()
    model.add(Dense(10, input_dim=20, activation="relu"))
    model.add(Dense(num_classes, activation="softmax"))
    model.compile(
        loss="categorical_crossentropy", optimizer="Adam", metrics=["accuracy"]
    )

    return model


def get_rf_model():
    # random forest, a decision tree based classifier

    model = RandomForestClassifier(20)
    return model


def read_json_request(path_to_json_file: Path) -> np.array:
    path_to_json_file = Path(path_to_json_file)
    with open(path_to_json_file, "r") as file:
        features_dict = json.load(file)
    output = np.array(list(features_dict.values())[1:])  # to remove id from features
    output = output.reshape(1, -1)
    return output


def load_model(trained_model_folder):
    # loading the trained classifier, h5,scaler, and json files

    try:
        json_file = open(trained_model_folder + os.sep + "model.json", "r")
        trained_model_json = json_file.read()
        json_file.close()
        model = model_from_json(trained_model_json)

        # load weights into new model

        model.load_weights(trained_model_folder + os.sep + "model.h5")
        # load the scaler as scaler

        scaler = joblib.load(trained_model_folder + os.sep + "scaler.gz")
    except KeyError:
        raise KeyError(
            "cannot get trained model, please check if all needed files are in trained model folder"
        )
    return model, scaler


def predict_sample(model, scaler, sample: np.array):
    scaled_sample = scaler.transform(sample)
    result = np.argmax(model.predict(scaled_sample, verbose = 0))
    result_class_name = classes_dict[result]
    return result, result_class_name


def train_validate_and_store_model(
    data: np.array,
    model_factory,
    desired_accuracy: int,
    num_classes: int,
    n_epochs: int,
    batch_size: int,
    model_output: str,
):

    training_features = data[:, 0:-1]  # remove class column
    training_Labels = data[:, -1]  # only class column

    # scaling our data, scaler will be saved at the end

    scaler = StandardScaler()
    scaler.fit(training_features)
    X = scaler.transform(training_features).astype(float)

    # hot encoder for better performane with ffnn, exa: class 2 will be [0 0 1 0]

    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(training_Labels)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    Y = onehot_encoder.fit_transform(integer_encoded)

    num_of_Labels = num_classes

    # Validation metrics init

    conf_mat_size = num_of_Labels, num_of_Labels
    cross_val_conf_mat = np.zeros(conf_mat_size)
    f1score = np.zeros(num_of_Labels)

    n_split = 5  # k-fold cross validation
    cross_val_accuracy = np.zeros(n_split)

    loop_index = 0
    results_arr_df = pd.DataFrame({"predicted_label": [], "desired_label": []})

    for train_index, test_index in KFold(n_split).split(X):
        current_split_results_arr_df = pd.DataFrame(
            {"predicted_label": [], "desired_label": []}
        )
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        print("\n Training", loop_index + 1, "/", n_split, " ...")
        if model_factory == "ffnn":
            model = get_ffnn_model(num_of_Labels)  # create_model()
            model.fit(
                x_train, y_train, epochs=n_epochs, batch_size=batch_size, verbose=0
            )
            cross_val_accuracy[loop_index] = model.evaluate(x_test, y_test)[1]
            current_split_results_arr_df["predicted_label"] = np.argmax(
                model.predict(x_test), axis=1
            )
            current_split_results_arr_df["desired_label"] = np.argmax(y_test, axis=1)
            results_arr_df = pd.concat([results_arr_df, current_split_results_arr_df])
            newrow = f1_score(
                np.argmax(y_test, axis=1),
                np.argmax(model.predict(x_test), axis=1),
                average=None,
            )
        elif model_factory == "rf":
            model = get_rf_model()
            model.fit(x_train, training_Labels[train_index])
            cross_val_accuracy[loop_index] = accuracy_score(
                model.predict(x_test), training_Labels[test_index]
            )
            current_split_results_arr_df["predicted_label"] = model.predict(x_test)
            current_split_results_arr_df["desired_label"] = training_Labels[test_index]
            results_arr_df = pd.concat([results_arr_df, current_split_results_arr_df])
            newrow = f1_score(
                model.predict(x_test), training_Labels[test_index], average=None
            )
        else:
            return KeyError(
                "only ffnn or rf can be used in model factory, please specify one of them"
            )
        if loop_index == 0:
            f1score = newrow
        else:
            f1score = np.append(f1score, newrow)
        print("f1score", np.mean(f1score))
        print("Model evaluation ", cross_val_accuracy[loop_index])
        loop_index += 1
    cross_val_accuracy = np.mean(cross_val_accuracy)
    print("\n Cross Validation Final Accuracy")
    print(cross_val_accuracy)
    # print(cross_val_conf_mat)

    cross_val_conf_mat = confusion_matrix(
        results_arr_df["desired_label"],results_arr_df["predicted_label"] 
    )
    print(cross_val_conf_mat)
    try:
        f1score = np.reshape(f1score, (-1, num_of_Labels))[1:]
    except ValueError:
        raise ValueError(
            "error f1score shape!, try to lower the number of folds 'n_split' "
        )
    f1score = np.mean(f1score, axis=0)

    # Final Training :Fit the model with all training data

    print("\n Final Training ...")
    if model_factory == "ffnn":
        model = get_ffnn_model(num_of_Labels)  # create_model()
        model.fit(X, Y, epochs=n_epochs, batch_size=batch_size, verbose=0)
    elif model_factory == "rf":
        model = get_rf_model()
        model.fit(X, training_Labels)
    # model=get_model(num_of_Labels)

    if 100 * cross_val_accuracy < desired_accuracy:
        print(
            "\n\n Training failed because of low accuracy = ", 100 * cross_val_accuracy
        )
        print(" desired_accuracy = ", desired_accuracy)
    else:

        # save model

        model_name = "model"
        model_path = os.path.join(model_output, model_name)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        if model_factory == "ffnn":
            model.save(model_path + ".h5", save_format="h5")

            model_json = model.to_json()
            with open(model_output + os.sep + "model.json", "w") as json_file:
                json_file.write(model_json)
            joblib.dump(scaler, model_output + os.sep + "scaler.gz")
            print("Trained ffnn model was saved as model.h5")
        elif model_factory == "rf":
            joblib.dump(model, "random_forest.joblib")
            print("Trained random forest model was saved as random_forest.joblib")
            # loaded_rf = joblib.load("my_random_forest.joblib")
    return cross_val_conf_mat
