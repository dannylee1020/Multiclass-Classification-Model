
'''
Need setup.py for python to find modules and use it to code
pip install -e to install setups and modularize other .py files
python setup.py develop

'''

from setuptools import setup, find_packages


setup(
	name='classification_project', 
	version='1.0', 
	packages=find_packages()
)