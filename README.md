# DataStratify
Data stratifying model with Python, without using designated ML libraries.


## User Manual:
1. Clone this repo to your local machine.
2. Create venv in the project directory. Make sure to activate the venv before dependencies installation.
3. Install all the dependencies via the 'requirements.txt' file:
### 'requirements.txt' installation:
    
    pip install -r requirements.txt

## Running the project:
We will run this project through CLI.
### Input example:
    data_stratify.py -path "PathToFileDir\data.csv" -ratio 0.2 -fields "device,location" 

A logging directory will be created upon first use. After every run, a log file will be created in this directory.
