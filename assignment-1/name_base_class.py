import sys
import config
class Name(object):
    '''
    Name base class
    '''

    _NAME_SEPARATOR = config.STRING_ONE_SPACE

    def __init__(self, firstname, middlename, lastname):
        '''
        class constructor
        '''
        self.firstname = firstname
        self.middlename = middlename
        self.lastname = lastname

    @classmethod
    def from_tuple(cls, name_tuple):
        '''
        Static factory method to create Name instance from a tuple
        '''
        firstname, middlename, lastname = name_tuple
        name = cls(firstname, middlename, lastname)
        return name

    @property
    def firstname(self):
        return self.__firstname

    @firstname.setter
    def firstname(self, firstname):
        self.__firstname = firstname

    @property
    def middlename(self):
        return self.__middlename

    @middlename.setter
    def middlename(self, middlename):
        self.__middlename = middlename

    @property
    def lastname(self):
        return self.__lastname

    @lastname.setter
    def lastname(self, lastname):
        self.__lastname = lastname

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def GetFullName(self):
        '''
        Build full name
        '''
        exception_message = None
        try:
            full_name = self.__firstname \
                        + Name._NAME_SEPARATOR + self.__middlename \
                        + Name._NAME_SEPARATOR + self.__lastname
        except:
            exception_message = sys.exc_info()[0]
        return full_name, exception_message
