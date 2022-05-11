import pandas as pd
import argparse, os, logging
from datetime import datetime


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
    grouped_data_test=chunk.groupby(fields, group_keys=False).apply(lambda x: x.sample(frac=ratio)) 
    grouped_data_train = chunk[~chunk.isin(grouped_data_test)].dropna()

    for field in fields:
        print(f"Field: {field}")
        logging.info(f"Field: {field}")
        print("Chunk data before split:")
        logging.info("Chunk data before split:")
        chunk_count = chunk[field].value_counts()
        print(chunk_count.to_frame(field), "\n")
        logging.info(f"{chunk_count.to_frame(field)}\n")

        train_count = grouped_data_train[field].value_counts()
        print("Train data after split:")
        logging.info("Train data after split:")
        print(train_count.to_frame(field), "\n")
        logging.info(f"{train_count.to_frame(field)}\n")

        test_count = grouped_data_test[field].value_counts()
        print("Test data after split:")
        logging.info("Test data after split:")
        print(test_count.to_frame(field))
        logging.info(f"{test_count.value_counts().to_frame(field)}")
        
        print("\n----------\n")
        logging.info("\n----------\n")

    grouped_data_train.to_csv(path_train, mode='a', index=False, header=False)
    grouped_data_test.to_csv(path_test, mode='a', index=False, header=False)


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

    # Splitting the data into chunks, choosing a set amount of chunksize.
    # In this way we will be able to read larger files.
    chunksize = 50000
    chunk_count = 1
    for chunk in pd.read_csv(path, chunksize=chunksize, iterator=True):
        print(f"- Chunk {chunk_count} -")
        logging.info(f"- Chunk {chunk_count} -")
        data_stratify(chunk, fields, ratio, path_train, path_test)
        chunk_count += 1
    
    print("data_train.csv was successfully created.\ndata_test.csv was successfully created.\n")
    logging.info("data_train.csv was successfully created.\ndata_test.csv was successfully created.")


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
        stratify(path, ratio, fields)
    except Exception as e:
        print(f"An error has occurred. Please check log file for more information:\n{e}")
        logging.error(f"Error:\nAn error has occurred:\n{e}")
    logging.shutdown()
    print("Log file was successfully created. Program will shutdown now..")