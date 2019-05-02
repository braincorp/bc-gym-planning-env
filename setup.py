from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name='bc_gym_planning_env',
    version='0.0.1',
    description='Brain Corp planning environments for OpenAI Gym',
    long_description=long_description,
    author='Oleg Sinyavskiy',
    author_email='olegsinyavskiy@gmail.com',
    url='https://github.com/braincorp/bc_gym_planning_env',
    download_url='',
    license='Braincorp',
    install_requires=['numpy==1.13.3',
                      'gym',
                      'attrs==19.1.0',
                      'opencv-python'],
    package_data={'': ['input']},
    include_package_data=True,
        extras_require={
        'tests': ['pytest==4.3.0',
                  'pytest-pep8==1.0.6',
                  'pytest-xdist==1.26.1',
                  'pylint==1.9.2',
                  'astroid==1.6.5'
                  ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],
    packages=find_packages(),
)
