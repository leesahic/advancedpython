import unittest
from name_base_class import Name

class TestNameClass(unittest.TestCase):

    __test_name = ('Leesa', 'M', 'Hicks')
    __name = None
 
    @classmethod
    def setUpClass(cls):
        firstname, middlename, lastname = cls.__test_name
        cls.__name = Name(firstname, middlename, lastname)

    def test_constructor(self):
        firstname, middlename, lastname = TestNameClass.__test_name

        self.assertEqual(firstname, TestNameClass.__name.firstname)
        self.assertEqual(middlename, TestNameClass.__name.middlename)
        self.assertEqual(lastname, TestNameClass.__name.lastname)

    def test_factory_with_tuple(self):
        name_from_tuple = Name.from_tuple(TestNameClass.__test_name)
        self.assertTrue(TestNameClass.__name == name_from_tuple)

    def test_name_getters(self):
        firstname, middlename, lastname = TestNameClass.__test_name
        self.assertTrue(firstname, TestNameClass.__name.firstname)
        self.assertTrue(middlename, TestNameClass.__name.middlename)
        self.assertTrue(lastname, TestNameClass.__name.lastname)

    def test_firstname_setter(self):
        new_firstname = 'Joanna'
        TestNameClass.__name.firstname = new_firstname
        self.assertTrue(new_firstname, TestNameClass.__name.firstname)
  
    def test_middlename_setter(self):
        new_middlename = 'Lynne'
        TestNameClass.__name.middlename = new_middlename
        self.assertTrue(new_middlename, TestNameClass.__name.middlename)

    def test_lastname_setter(self):
        new_lastname = 'Barnes'
        TestNameClass.__name.lastname = new_lastname
        self.assertTrue(new_lastname, TestNameClass.__name.lastname)

if __name__ == '__main__':
    unittest.main()