from database import DatabaseConnector
import config
import logging

_logger = None

class MagazineData:
    issn_number, name, publisher_id = range(3)

class PublisherData:
    company_name, contact_name, contact_title = range(3)

'''
Configure logging
'''
def configure_logger (log_level = logging.INFO):
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
            '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(log_level)
    return logger


'''
Get magazine data
'''
def get_magazine_data(db, db_connect):
    try:
        sql = ("SELECT issn_number, name, publisher_id FROM magazine WHERE department_id > %s")
        parameters = (str(5), )

        cursor, data = db.mysql_execute_cursor_select(db_connect, sql, parameters)                    
        if cursor.rowcount > 0:
            print ('Magazine data:')
            print (str('ISSN').ljust(16), str('Name').ljust(46), str('PubID').rjust(5), sep='')             
            for row in data :
                print ( str(row[MagazineData.issn_number]).ljust(16),
                        str(row[MagazineData.name]).ljust(46),
                        str(row[MagazineData.publisher_id]).rjust(5),
                        sep=''
                      )     
    except Exception as ex:     
        print('An error occurred: ' + str(ex) + '. Contact your System Administrator.')
    finally:                
        db.mysql_close_cursor(cursor)


'''
Get publisher data
'''
def get_publisher_data(db, db_connect):
    try:
        sql = ("SELECT company_name, contact_name, contact_title FROM publisher WHERE contact_title LIKE %s");
        parameters = ("Senior%", )

        cursor, data = db.mysql_execute_cursor_select(db_connect, sql, parameters)                    
        if cursor.rowcount > 0:
            print ('Publisher data:')             
            for row in data :
                print ( str(row[PublisherData.company_name]).ljust(51),
                        str(row[PublisherData.contact_name]).ljust(31),
                        str(row[PublisherData.contact_title]).ljust(31),
                        sep=''
                      )     
    except Exception as ex:     
        print('An error occurred: ' + str(ex) + '. Contact your System Administrator.')
    finally:                
        db.mysql_close_cursor(cursor)


'''
Open a mysql database
'''
def open_mysql_database( user = None, password = None, host = None, db = None):
    db = None
    db_connect = None

    try:
        db = DatabaseConnector(config.USER, config.PASSWORD, config.HOST, config.DATABASE)
        db_connect = db.mysql_open_connection()            
    except Exception as ex:
        _logger.error('Unable to open the ' + config.DATABASE + ' database: ')
        _logger.error(str(ex))
        db = None
        db_connect = None

    return (db, db_connect)


def main():
    global _logger
    magazine_db = None
    magazine_db_connect = None

    _logger = configure_logger(log_level = logging.DEBUG)
    _logger.info('test log message')

    (magazine_db, magazine_db_connect) = open_mysql_database( user = config.USER,
                                                              password = config.PASSWORD,
                                                              host = config.HOST,
                                                              db = config.DATABASE
                                                            )

    if (magazine_db == None or magazine_db_connect == None):
        SystemExit(1)

    get_magazine_data(magazine_db, magazine_db_connect)
    get_publisher_data(magazine_db, magazine_db_connect)
    magazine_db_connect.close()


if __name__ == '__main__':
    main()