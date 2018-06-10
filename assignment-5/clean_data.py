#
# Advanced Python for Everybody
# Assignment 5
# Leesa Hicks
#

import os
import sys
import re
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
    print()
    print("Drop duplicates")
    df_data = df_data.drop_duplicates(keep = "first")
    print(df_data)

    # Replace empty values with max value in column 1
    print()
    print("Column 1")
    df_data["1"].fillna(df_data["1"].max(), inplace=True)
    print(df_data)

    # Replace empty values with min value in column 2
    print()
    print("Column 2")
    df_data["2"].fillna(df_data["2"].min(), inplace=True)
    print(df_data)

    # Replace empty values with mode value in column 3
    print()
    print("Column 3")
    df_data["3"].fillna((df_data["3"].mode()).iloc[0], inplace=True)
    print(df_data)

    # Replace empty values with "none" in column 4
    print()
    print("Column 4")
    df_data["4"].fillna("none", inplace=True)
    print(df_data)
    
    # Capitalize the first letter in column 4
    df_data["4"] = df_data["4"].str.capitalize()
    print(df_data)

    # Replace empty values with False in column 5
    print()
    print("Column 5")
    fill_empty_column_values_with_bool(df_data, "5", False)
    print(df_data)

    # Convert False to 0 and True to 1 in column 5
    df_data["5"] = df_data["5"].apply(lambda b: 1 if b else 0)
    print(df_data)

    # Replace empty values with the current month/year in column 6
    #todays_date = datetime.date.today()
    print()
    print("Column 6")
    df_data["6"].fillna(datetime.now().strftime("%m/01/%Y"), inplace=True)
    print(df_data)

    # Make the date format consistent for all values in column 6
    df_data["6"] = df_data["6"].apply(lambda d: datetime.strftime(datetime.strptime(d, "%m/%d/%Y"), "%m/%d/%Y"))
    print(df_data)

    # Replace empty values with "$0.0 in column 7"
    print()
    print("Column 7")
    df_data["7"].fillna("$0.0", inplace=True)
    print(df_data)

    # Convert the dollar amount to a rounded integer value
    df_data["7"] = df_data["7"].apply(lambda s: int(round(float(re.sub(r'\s*\$([0-9,\.]*)\s*', r'\1', s)), 0)))
    print(df_data)

    # Combine first and last name in columns 8 and 9 to create column 10
    print()
    print("Columns 8, 9, and 10")
    df_data["10"] = df_data["8"] + " " + df_data["9"]
    print(df_data)

    # Replace empty values in columns 8, 9, and 10 with "N/A"
    df_data.fillna({"8":"N/A", "9":"N/A", "10":"N/A"}, inplace=True)
    print(df_data)

    # Write out the clean data
    df_data.to_csv(output_file_path)
        
if __name__ == '__main__':
    main()