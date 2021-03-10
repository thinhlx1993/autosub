#!/usr/bin/env python
from __future__ import unicode_literals

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

long_description = (
    'Autosub is a utility for automatic speech recognition and subtitle generation. '
    'It takes a video or an audio file as input, performs voice activity detection '
    'to find speech regions, makes parallel requests to Google Web Speech API to '
    'generate transcriptions for those regions, (optionally) translates them to a '
    'different language, and finally saves the resulting subtitles to disk. '
    'It supports a variety of input and output languages (to see which, run the '
    'utility with --list-src-languages and --list-dst-languages as arguments '
    'respectively) and can currently produce subtitles in either the SRT format or '
    'simple JSON.'
)

setup(
    name='autosub_v2',
    version='0.1.0',
    description='Auto-generates subtitles for any video or audio file using google apis',
    long_description=long_description,
    author='Anastasis Germanidis',
    author_email='thinhle.ict@gmail.com',
    url='https://github.com/thinhlx1993/autosub_v2',
    packages=['autosub_v2'],
    entry_points={
        'console_scripts': [
            'autosub_v2 = autosub_v2:main',
        ],
    },
    install_requires=[
        'google-api-python-client>=1.4.2',
        'opencv-python',
        'google-cloud-vision>=2.2.0',
        'google-cloud-translate>=3.0.2',
        'requests>=2.3.0',
        'pysrt>=1.0.1',
        'progressbar2>=3.34.3',
        'six>=1.11.0',
        'paddleocr==2.0.1'
    ],
    license=open("LICENSE").read()
)
