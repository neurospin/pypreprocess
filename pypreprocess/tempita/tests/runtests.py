import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import doctest

if __name__ == '__main__':
    doctest.testfile('test_template.txt')
    doctest.testfile('../docs/index.txt')
    
