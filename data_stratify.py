import pandas as pd
import numpy as np
import argparse, os
from sklearn.model_selection import train_test_split



def stratify_data(path, ratio, fields):
    data = pd.read_csv(path)
    headers = data.columns.to_list()
    print("fields data before start:")
    print(data['device'].value_counts())
    data = data.to_numpy()

    # Creating a dict which includes the fields & its index.
    indexes = list(range(0, len(headers)))
    fields_dict = dict(zip(headers, indexes))

    Y = []
    for el in data:
        Y.append(el[fields_dict[fields[0]]])

    # print("\n", Y, "\n\n")

    X_train, X_test, y_train, y_test = train_test_split(data, Y, test_size=ratio, stratify=Y)
    # print("X_train:\n", X_train)
    # print("\nX_test:\n", X_test)

    df_xtrain = pd.DataFrame(X_train)
    df_xtest = pd.DataFrame(X_test)
    print("\ndf_xtrain:")
    print(df_xtrain[0].value_counts())

    print("\ndf_xtest:")
    print(df_xtest[0].value_counts())


# Creating the arguments to get the parameters through CLI.
def parse_args():
    parser = argparse.ArgumentParser(description="Data Stratify project")
    parser.add_argument('-path', type=str, required=True, help="The path to the .csv file.")
    parser.add_argument('-ratio', type=float, required=True, help="The ratio for the test/train split, number between 0 to 1.")
    parser.add_argument('-fields', type=str, required=True, help="The fields of the data that the stratifying will be based on.\n Choose between: device, location, date, os, source, sale.")
    args = parser.parse_args()

    is_file = os.path.isfile(args.path)
    if not is_file:
        print("The file in the path that was entered is not exist. Please check the path and try again.")
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
        print(shared_args, args_fields)
        print("Fields are incorrect. Please choose from the following:\ndevice, location, date, os, source, sale.")
        # log here
        exit(1)

    return (args.path, args.ratio, args_fields)


if __name__ == '__main__':
    path, ratio, fields = parse_args()
    stratify_data(path, ratio, fields)

