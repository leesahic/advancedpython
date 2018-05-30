
import time
from extract.extract_data  import ExtractData
from transform.transform_data  import TransformData
from load.load_data  import LoadData
import config

def main():

    print("ETL process starting")
    
    print("1. Extracting  the data...")
    extract_data = ExtractData()
#   publisher data
    publisher_data_rows, publisher_data_columns = extract_data.publisher_data_extract()
    magazine_data_rows, magazine_data_columns = extract_data.publisher_magazine_data_extract()  

    print("2. Transforming the data...")
    transform_data = TransformData()
    df_publisher_data = transform_data.publisher_data_transform(publisher_data_rows, publisher_data_columns)
    df_publisher_data_final = transform_data.magazine_list_by_publisher_transform(magazine_data_rows, magazine_data_columns, df_publisher_data)

    print("3. Loading the data...")
    load_data =  LoadData()    
    load_data.file_load(df_publisher_data, config.PUBLISHER_DATA, new_index_column="publisher_id",  file_type="CSV")    
    load_data.file_load(df_publisher_data_final, config.PUBLISHER_DATA_FULL, new_index_column="publisher_id",  file_type="CSV")    
 
    print("ETL process completed")

if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print("Program Runtime: " + str(round(end_time - start_time, 1)) + " seconds" + "\n")

    