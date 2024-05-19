# -*- coding: utf-8 -*-

"""
How to run?
1) go to local_paths.py, and set the paths for training data, testing data ,trained_model_folder, json sample file
2) run main.py !

Notes:
*) functions.py contains all needed functions
*)requirements.txt may help in the used versions of libraries, if compability problems happened
*) two classifiers are trained: feedforward neural network and random forest
*) you can change index_example, almost at the end of this file, to predict a result for a device from testing data
*) app.py is  RESTful API, also to predict the price for any device that can be an input as json file
*) this python project was tested successfully in windows11 system, python 3.10
Created on Fri May 17 13:02:14 2024
@author: Bashar Wannous: basharw773@gmail.com
"""

import pandas as pd
import numpy as np

from functions import (
    predict_sample,
    display_confusion_matrix,
    load_model,
    check_remove_nulls,
    plot_two_features,
    check_categorical_exist,
    plot_classes_histogram,
    train_validate_and_store_model,
)
from local_paths import training_data_path, testing_data_path, trained_model_folder
from used_case_specs import N_EPOCHES, BATCH_SIZE

# read csv data

training_data = pd.read_csv(training_data_path)
testing_data = pd.read_csv(testing_data_path)

# preprocess & EDA
# check duplication to drop them if any duplication was founded

training_data[training_data.duplicated()]  # No duplication found

# check nulls and nulls ratio if any nulls were founded

training_data = check_remove_nulls(training_data)
testing_data = check_remove_nulls(testing_data)

# nulls were dropped  from training_data, less than 1% so it is ok to remove


# check categorical data

if check_categorical_exist(training_data):
    print("some categorical data has been detected")
# plot class histogram to check data balance, as data balance effect some classifier types like Neural Network classifier

num_of_Labels = plot_classes_histogram(
    training_data["price_range"]
)  # plot shows the data is balanced

# check min max values, to know if we need to scale in case of Neural Network classifier

data_summarize = np.array(
    [np.min(np.array(training_data), axis=0), np.max(np.array(training_data), axis=0)]
)
# so data scaling is needed with Neural Network classifier

# choose two features to plot with colored classes, might help to explore features more

plot_two_features(
    training_data, 11, 12
)  # px height and px width are corelated, but let us keep them as they are

# check columns

training_data.columns  # No device id in training data
testing_data.columns  # there is device id in testing data
# we will train two classifiers, compare results, and choose the best
# first one is Feed Forward Neural Network 'ffnn', its known with its high accuracy
# second one is random forest 'rf', a Decision Tree based classifer, it gives fast results


model_factory_list = ["rf", "ffnn"]  #'ffnn'or 'rf'
labels_array = np.unique(np.array(training_data)[:, -1])  # labels

for model_factory in model_factory_list:
    # train, cross validation

    cross_val_conf_mat = train_validate_and_store_model(
        training_data.values,
        model_factory,
        desired_accuracy=90,  #  0~100 %
        num_classes=num_of_Labels,
        n_epochs=N_EPOCHES,
        batch_size=BATCH_SIZE,
        model_output=trained_model_folder,
    )
    display_confusion_matrix(cross_val_conf_mat, labels_array)
# ffnn classifier gives a better accuracy ~93%, confusion matrix is better, and it is fast enough, so we will continue  with ffnn.
# confusion matrix for ffnn classifier show that, error might happen with the neighbor class only
# exa: very_high_cost device might (with low probability) be classified as high_cost device, but can never classified as medium_cost or low_cost

# let us take some of testing data and predict the result

for index_example in range(100,110):
    model, scaler = load_model(trained_model_folder)
    sample = np.array(testing_data.iloc[index_example - 1])[1:].reshape(
        1, -1
    )  # we should remove id frpm testing sample
    result, result_class_name = predict_sample(model, scaler, sample)
    print(f"the prediction of sample {index_example} is :", result_class_name)
