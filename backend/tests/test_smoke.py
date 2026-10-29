import sys

import lote


def test_python_version():
    assert sys.version_info[:2] == (3, 13)


def test_package_imports():
    assert lote.__doc__
