import mysql.connector
import sys


def main():

    cnx = None

    try :
        cnx = mysql.connector.connect(database='books', host='localhost', user='leesa', password='Luna6Cat')
    except mysql.connector.Error as err:
        print(err)
        exit(1)

    if (cnx != None):
        cnx.close()


if __name__ == '__main__':
    main()