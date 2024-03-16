#!/usr/bin/env python

from setuptools import setup

setup(
    entry_points={
        'console_scripts': [
            'comos = comos.comos:main',
        ]
    }
)