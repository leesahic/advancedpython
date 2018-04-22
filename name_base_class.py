class Name(object):
    '''
    Name base class
    '''

    def __init__(self, firstname, middlename, lastname):
        '''
        class constructor
        '''
        self.firstname = firstname
        self.middlename = middlename
        self.lastname = lastname

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    @classmethod
    def from_tuple(cls, name_tuple):
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


    # @staticmethod
    # def show_message(message):
    #     print("The message is {}".format(message)