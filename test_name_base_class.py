import unittest
from name_base_class import Name

class TestNameClass(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        test_name = ('Leesa', 'M', 'Hicks')
        firstname, middlename, lastname = test_name
        name = Name(firstname, middlename, lastname)
        return test_name, name

    def test_constructor(self):
        test_name, name = TestNameClass.setUpClass()
        firstname, middlename, lastname = test_name

        self.assertEqual(firstname, name.firstname)
        self.assertEqual(middlename, name.middlename)
        self.assertEqual(lastname, name.lastname)

    def test_factory_with_tuple(self):
        test_name, name = TestNameClass.setUpClass()
        name_from_tuple = Name.from_tuple(test_name)
        self.assertTrue(name == name_from_tuple)
        

if __name__ == '__main__':
    unittest.main()