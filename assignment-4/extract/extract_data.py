
from extract.mysql_database import MySQLDatabase
import config

from library.etl_library import ETLLibrary

class ExtractData(ETLLibrary):

    def __init__(self):
#         ETLLibrary.__init__(self)
        super().__init__()


    def publisher_data_extract(self):
        """
        Extract publisher data
        """
        publisher_data_row = None
        publisher_data_column = None
        try:
#           initialize mysql database object
            mysql_database = MySQLDatabase(self.b64decode_string(config.USER), self.b64decode_string(config.PASSWORD), 
                self.b64decode_string(config.HOST), self.b64decode_string(config.DATABASE))
#           open mysql connection object
            mysql_connection = mysql_database.mysql_open_connection()                
#           sql select statement
            sql = "".join(["SELECT publisher_id, company_name, contact_name, phone, fax, email, website ",
                           "FROM publisher ORDER BY publisher_id"])                  
#             get publisher data rows and columns
            publisher_data_rows, publisher_data_columns  = mysql_database.mysql_cursor_select(mysql_connection, sql)             
        except Exception:     
            self.print_exception_message()                      
        finally:
#             close mysql connection object
            mysql_database.mysql_close_connection(mysql_connection)
        return publisher_data_rows, publisher_data_columns

    def publisher_magazine_data_extract(self):
        """
        Get the magazines from each publisher
        """
        magazine_data_rows = None
        magazine_data_columns = None

        try:
            mysql_database = MySQLDatabase(self.b64decode_string(config.USER), self.b64decode_string(config.PASSWORD), 
                self.b64decode_string(config.HOST), self.b64decode_string(config.DATABASE))
#           open mysql connection object
            mysql_connection = mysql_database.mysql_open_connection()                
#           sql select statement
            sql = "".join(["SELECT publisher.publisher_id ",
                           "FROM publisher ",
                           "INNER JOIN magazine ",
                           "ON (publisher.publisher_id = magazine.publisher_id) ",
                           "ORDER BY publisher_id"])                  
#           get magazine data rows and columns
            magazine_data_rows, magazine_data_columns = mysql_database.mysql_cursor_select(mysql_connection, sql)
        except Exception:     
            self.print_exception_message()                      
        finally:
#             close mysql connection object
            mysql_database.mysql_close_connection(mysql_connection)
        return magazine_data_rows, magazine_data_columns      
              
