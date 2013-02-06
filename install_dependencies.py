"""
:Module: install_dependencies.py
:Synopsis: Simple dependency-installer for pypreprocess.
:Author: dohmatob elvis dopgima

"""

import os
import sys

if __name__ == '__main__':
    os.chdir(os.path.dirname(
            os.path.abspath(sys.argv[0])))

    other_options_for_pip = "--user"
    if len(sys.argv) > 1:
        other_options_for_pip = " ".join(sys.argv[1:])

    cmd1 = 'pip install -r urgent_dependencies.txt %s' % other_options_for_pip

    cmd2 = 'pip install -r questionable_dependencies.txt %s' \
        % other_options_for_pip

    cmd3 = 'pip install -r dependencies.txt %s' % other_options_for_pip

    print "Running: %s\r\n\r\n" % cmd1
    os.system(cmd1)

    print "Running: %s\r\n" % cmd2
    os.system(cmd2)

    print "Running: %s\r\n" % cmd3
    os.system(cmd3)
