import pandas as pd
import numpy as np
import argparse, os, logging
from sklearn.model_selection import train_test_split



def stratify_data(path, ratio, fields):

    print("\nStarting splitting process..\n\n----------\n")
    data = pd.read_csv(path).dropna() # Using dropna to remove rows with missing data.   nrows=10
    headers = data.columns.to_list()

    if ratio == 0 :
        print("All of the data was set to train. No data will go into test, therefore data_test.csv file won't be created.")
        data.to_csv("data_train.csv", index=False, header=headers)
        # log here
        return
    elif ratio == 1:
        print("All of the data was set to test. No data will go into train, therefore data_train.csv file won't be created.")
        data.to_csv("data_test.csv", index=False, header=headers)
        # log here
        return

    print("Data before split:")
    for field in fields:
        print(data[field].value_counts().rename_axis(field))
        # log here
    print("\n----------\n")
    
    data = data.to_numpy()

    # Creating a dict which includes the fields & its index.
    indexes = list(range(0, len(headers)))
    fields_dict = dict(zip(headers, indexes))

    # To dynamically create the Y vector using the fields we are getting from the user.
    field_idx = []
    for field in fields:
        field_idx.append(fields_dict[field])
    
    Y = []
    temp_list_item = []
    for el in data:
        for i in field_idx:
            temp_list_item.append(el[i])
        Y.append(temp_list_item)
        temp_list_item = []

    X_train, X_test, y_train, y_test = train_test_split(data, Y, test_size=ratio, stratify=Y)

    df_xtrain = pd.DataFrame(X_train)
    df_xtest = pd.DataFrame(X_test)

    print("Train data with stratify:")
    for idx in field_idx:
        print(df_xtrain[idx].value_counts(), "\n")
        # log here

    print("Test data with stratify:")
    for idx in field_idx:
        print(df_xtest[idx].value_counts(), "\n")
        # log here

    df_xtrain.to_csv("data_train.csv", index=False, header=headers)
    df_xtest.to_csv("data_test.csv", index=False, header=headers)
    # log here


# Creating the arguments to get the parameters through CLI.
def parse_args():
    parser = argparse.ArgumentParser(description="Data Stratify project")
    parser.add_argument('-path', type=str, required=True, help="The path to the .csv file.")
    parser.add_argument('-ratio', type=float, required=True, help="The ratio for the test/train split, number between 0 to 1.")
    parser.add_argument('-fields', type=str, required=True, help="The fields of the data that the stratifying will be based on.\n Choose between: device, location, date, os, source, sale.")
    args = parser.parse_args()

    is_file = os.path.isfile(args.path)
    if not is_file:
        print("Cannot open the file in this path. Please check the path/make sure to address the .csv file and try again.")
        # Log here
        exit(1)

    if args.ratio > 1 or args.ratio < 0:
        print("Ratio value is not correct! Please insert a number between 0 to 1.")
        # log here
        exit(1)

    args_fields = args.fields.split(',')
    args_fields = [s.strip() for s in args_fields]
    fields = ["device", "location", "date", "os", "source", "sale"]
    shared_args = [x for x in args_fields if x in fields]

    if not shared_args == args_fields:
        print("Fields are incorrect. Please choose from the following:\ndevice, location, date, os, source, sale.")
        # log here
        exit(1)

    return (args.path, args.ratio, args_fields)


if __name__ == '__main__':
    try:
        # start logging
        path, ratio, fields = parse_args()
        stratify_data(path, ratio, fields)
    except:
        print("An error has occurred. Please check log file for more information.")
        # close log here

