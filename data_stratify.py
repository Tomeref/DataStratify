import pandas as pd
import argparse, os, logging
from datetime import datetime
from sklearn.model_selection import train_test_split


def init_log():
    cwd = os.getcwd()
    log_dir = os.path.join(cwd, "Logging_Files")
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)

    try:
        now = datetime.now()
        log_fie = log_dir + "\Log " + now.strftime("%m.%d.%Y %H-%M-%S" + ".log") 
        logging.basicConfig(filename=log_fie, filemode='w', level=logging.INFO, format='%(message)s')
    except:
        print("Could not create a log file.")
        logging.basicConfig(level=logging.CRITICAL, format='%(message)s')

################

def data_stratify(chunk, fields, ratio, path_train, path_test):
    # headers = chunk.columns.to_list()
    grouped_data_test=chunk.groupby(fields, group_keys=False).apply(lambda x: x.sample(frac=ratio)) 
    test_size = grouped_data_test.value_counts()
    print("\n------\n")
    # print(test_size)

    print("\n------\n")

    grouped_data_train = chunk[~chunk.isin(grouped_data_test)].dropna()

    grouped_data_train.to_csv(path_train, mode='a', index=False, header=False)
    grouped_data_test.to_csv(path_test, mode='a', index=False, header=False)


    # group_data = chunk.groupby(fields)
    # group_data_size = group_data.size()
    # group_data_size_indexes = group_data_size .index.tolist()
    # group_data_values = group_data_size .values.tolist()
    # print(group_data)
    # print(group_data_size )
    # print(group_data_size_indexes)
    # data_proportion = np.array(group_data_values)/np.array(chunk.shape[0])
    # print(data_proportion)

    # ungrouped_data = group_data.head(group_data.ngroup().size) # set it back to a DF.


################

def stratify(path, ratio, fields):
    print("\nStarting splitting process..\n\n----------\n")
    logging.info("Starting splitting process..\n\n----------\n")
    headers = pd.read_csv(path, nrows=0).columns.to_list()

    # Initializing new data_train.csv & new data_test.csv files.
    data_train = pd.DataFrame(columns=headers)
    path_train = os.path.join(os.path.dirname(path), "data_train.csv")
    data_train.to_csv(path_train, index=False)

    data_test = pd.DataFrame(columns=headers)
    path_test = os.path.join(os.path.dirname(path), "data_test.csv")
    data_test.to_csv(path_test, index=False)

    chunksize = 50000
    for chunk in pd.read_csv(path, chunksize=chunksize, iterator=True):
        data_stratify(chunk, fields, ratio, path_train, path_test)
        



def stratify_data(path, ratio, fields):
    print("\nStarting splitting process..\n\n----------\n")
    logging.info("Starting splitting process..\n\n----------\n")
    data = pd.read_csv(path).dropna() # Using dropna to remove rows with missing data.
    headers = data.columns.to_list()

    if ratio == 0 :
        print("All of the data was set to train. No data will go into test, therefore data_test.csv file won't be created.")
        data.to_csv("data_train.csv", index=False, header=headers)
        logging.info("All of the data was set to train. No data will go into test, therefore data_test.csv file won't be created.\ndata_train.csv was successfully created.")
        return
    elif ratio == 1:
        print("All of the data was set to test. No data will go into train, therefore data_train.csv file won't be created.")
        data.to_csv("data_test.csv", index=False, header=headers)
        logging.info("All of the data was set to test. No data will go into test, therefore data_train.csv file won't be created.\ndata_test.csv was successfully created.")
        return

    print("Data count per field to stratify before the split:\n")
    logging.info("- Data count per field to stratify before the split -\n")
    for field in fields:
        print(data[field].value_counts().to_frame(field), "\n")
        logging.info(f"{data[field].value_counts().to_frame(field)}\n")
    print("----------\n")
    logging.info("----------\n")

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

    try:
        X_train, X_test, y_train, y_test = train_test_split(data, Y, test_size=ratio, stratify=Y)
    except:
        print("An error has occurred while splitting the data.\nThere might be too many fields to stratify versus the data that was provided.")
        logging.error("Error:\nAn error has occurred while splitting the data.\nThere might be too many fields to stratify versus the data that was provided.")
        return

    df_xtrain = pd.DataFrame(X_train)
    df_xtest = pd.DataFrame(X_test)

    print("Train data count per field to stratify after the split :\n")
    logging.info("- Train data count per field to stratify after the split -\n")
    for idx in field_idx:
        print(df_xtrain[idx].value_counts().to_frame(fields[idx]), "\n")
        logging.info(f"{df_xtrain[idx].value_counts().to_frame(fields[idx])}\n")

    print("Test data count per field to stratify after the split:")
    logging.info("\n- Test data count per field to stratify after the split -\n")
    for idx in field_idx:
        print(df_xtest[idx].value_counts().to_frame(fields[idx]), "\n")
        logging.info(f"{df_xtest[idx].value_counts().to_frame(fields[idx])}\n")

    df_xtrain.to_csv("data_train.csv", index=False, header=headers)
    df_xtest.to_csv("data_test.csv", index=False, header=headers)
    print("----------\n\ndata_train.csv was successfully created.\ndata_test.csv was successfully created.\n")
    logging.info("----------\n\ndata_train.csv was successfully created.\ndata_test.csv was successfully created.")


# Creating the arguments to get the parameters through CLI.
def parse_args():
    parser = argparse.ArgumentParser(description="Data Stratify project")
    parser.add_argument('-path', type=str, required=True, help="The path to the .csv file.")
    parser.add_argument('-ratio', type=float, required=True, help="The ratio for the test/train split, number between 0 to 1.")
    parser.add_argument('-fields', type=str, required=True, help="The fields of the data that the stratifying will be based on.\n Choose between: device, location, date, os, source, sale.")
    args = parser.parse_args()

    is_file = os.path.isfile(args.path)
    if not is_file:
        raise Exception("Cannot open the file in this path. Please check the path/make sure to address the .csv file and try again.")

    if args.ratio > 1 or args.ratio < 0:
        raise Exception("Ratio value is not correct! Please insert a number between 0 to 1.")

    args_fields = args.fields.split(',')
    args_fields = [s.strip() for s in args_fields]
    fields = ["device", "location", "date", "os", "source", "sale"]
    shared_args = [x for x in args_fields if x in fields]

    if not shared_args == args_fields:
        raise Exception("Fields are incorrect. Please choose from the following:\ndevice, location, date, os, source, sale.")

    logging.info(f"The variables that were inserted:\nPath: {args.path}     Ratio: {args.ratio}     Fields: {args_fields}\n")
    return (args.path, args.ratio, args_fields)

################

if __name__ == '__main__':
    try:
        init_log()
        path, ratio, fields = parse_args()
        # stratify_data(path, ratio, fields)
        stratify(path, ratio, fields)
    except Exception as e:
        print(f"An error has occurred. Please check log file for more information:\n{e}")
        logging.error(f"Error:\nAn error has occurred:\n{e}")
    
    
    logging.shutdown()
    print("Log file was successfully created. Program will shutdown now..")