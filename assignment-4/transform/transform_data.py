
from library.etl_library import ETLLibrary

import pandas as pd
# import numpy as np

class TransformData(ETLLibrary):

    def __init__(self):
#         ETLLibrary.__init__(self)
        super().__init__()

    def publisher_data_transform(self, publisher_data_rows, publisher_data_columns):
        """
        Convert the publisher data into a dataframe
        """
        df_publisher_data = None
        try:
            if len(publisher_data_rows) > 0:
                df_publisher_data = pd.DataFrame.from_records(data=publisher_data_rows, columns=publisher_data_columns)
        except Exception:
            self.print_exception_message()
        
        return df_publisher_data

    def magazine_list_by_publisher_transform(self, magazine_data_rows, magazine_data_columns, df_publisher_data):
        """
        Calculate the number of magazines from each publisher and add the totals to the publisher data
        """
        df_publisher_magazines = None
        try:
            if len(magazine_data_rows) > 0:
                df_magazines = pd.DataFrame.from_records(data=magazine_data_rows, columns=magazine_data_columns)
                df_magazines['total_magazines'] = '1'
                df_magazines = df_magazines.groupby(["publisher_id"], as_index=False)["total_magazines"].count()
                df_publisher_magazines = pd.merge(df_magazines, df_publisher_data, on="publisher_id")
        except:
            self.print_exception_message()
        return df_publisher_magazines