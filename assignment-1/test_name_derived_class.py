import unittest
from name_derived_class import FormalName

class TestFormalNameClass(unittest.TestCase):

    __test_name = ('Ms', 'Leesa', 'M', 'Hicks')
    __name = None

    @classmethod
    def setUpClass(cls):
        title, firstname, middlename, lastname = cls.__test_name
        cls.__name = FormalName(title, firstname, middlename, lastname)

    def test_constructor(self):
        title, firstname, middlename, lastname = TestFormalNameClass.__test_name

        self.assertEqual(title, TestFormalNameClass.__name.title)
        self.assertEqual(firstname, TestFormalNameClass.__name.firstname)
        self.assertEqual(middlename, TestFormalNameClass.__name.middlename)
        self.assertEqual(lastname, TestFormalNameClass.__name.lastname)

    def test_factory_with_tuple(self):
        name_from_tuple = FormalName.from_tuple(TestFormalNameClass.__test_name)
        self.assertTrue(TestFormalNameClass.__name == name_from_tuple)
        
    def test_title_setter(self):
        new_title = 'Duchess'
        TestFormalNameClass.__name.title = new_title
        self.assertTrue(new_title, TestFormalNameClass.__name.title)
  


if __name__ == '__main__':
    unittest.main()