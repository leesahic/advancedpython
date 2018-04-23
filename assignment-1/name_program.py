from name_base_class import Name
from name_derived_class import FormalName


def main():
    test_names = [('Ms', 'Leesa', 'M', 'Hicks'), ('Mr', 'David', 'H', 'Penney')]

    for test_name in test_names:

        title, firstname, middlename, lastname = test_name
        base_name = Name(firstname, middlename, lastname)
        formal_name = FormalName(title, firstname, middlename, lastname)

        name, exception_msg = base_name.GetFullName()
        if exception_msg is None:
            print("Name=", name)
        
        name, exception_msg = formal_name.GetFullName()
        if exception_msg is None:
            print ("Formal name=", name)

if __name__ == '__main__':
    main()