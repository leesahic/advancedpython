import sys
from name_base_class import Name

class FormalName(Name):
    '''
    Formal name derived class
    '''

    def __init__(self, title, firstname, middlename, lastname):
        super().__init__(firstname, middlename, lastname)
        self.title = title

    @classmethod
    def from_tuple(cls, name_tuple):
        '''
        Static factory method to create FormalName instance from a tuple
        '''
        title, firstname, middlename, lastname = name_tuple
        name = cls(title, firstname, middlename, lastname)
        return name

    @property
    def title(self):
        return self.__title

    @title.setter
    def title(self, title):
        self.__title = title

    def GetFullName(self):
        '''
        Build full name
        '''
        exception_message = None
        try:
            full_name, exception_message = super().GetFullName()
            if exception_message is None:
                full_name = self.__title + Name._NAME_SEPARATOR + full_name
        except:
            exception_message = sys.exc_info()[0]
        return full_name, exception_message
