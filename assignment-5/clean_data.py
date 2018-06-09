#
# Advanced Python for Everybody
# Assignment 5
# Leesa Hicks
#

import os
import sys
from datetime import datetime
#from tabulate import tabulate

import pandas as pd
import numpy as np

import config

def fill_empty_column_values_with_bool(ds, label, boolean_value):
    '''
    Replaces empty/null values with the supplied boolean value for a column
    '''
    ds[label].fillna(boolean_value, inplace=True)

def main():
    # Get default project directory path
    project_directory_path = os.path.dirname(sys.argv[0])  

    # Assemble file names
    input_file_path = os.path.join(project_directory_path, config.INPUT_FILE_PATH) 
    output_file_path = os.path.join(project_directory_path, config.OUTPUT_FILE_PATH) 
    
    df_data = pd.read_csv(input_file_path, sep=",")  
    print(df_data)

    # Drop duplicate rows and keep first
    df_data = df_data.drop_duplicates(keep = "first")
    print(df_data)

    # Replace empty values with max value in column 1
    print(df_data["1"].max())
    df_data["1"].fillna(df_data["1"].max(), inplace=True)
    print(df_data)

    # Replace empty values with min value in column 2
    print(df_data["2"].min())
    df_data["2"].fillna(df_data["2"].min(), inplace=True)
    print(df_data)

    # Replace empty values with "none" in column 4
    df_data["4"].fillna("none", inplace=True)
    print(df_data)
    
    # Capitalize the first letter in column 4
    df_data["4"] = df_data["4"].str.capitalize()
    print(df_data)

    # Replace empty values with False in column 5
    fill_empty_column_values_with_bool(df_data, "5", False)
    print(df_data)

    # Convert False to 0 and True to 1 in column 5
    df_data["5"] = df_data["5"].apply(lambda b: 1 if b else 0)
    print(df_data)

    # Replace empty values with the current month/year in column 6
    #todays_date = datetime.date.today()
    df_data["6"].fillna(datetime.now().strftime("%m/1/%Y"), inplace=True)
    print(df_data)



        
if __name__ == '__main__':
    main()