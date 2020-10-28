from setuptools import setup

setup(
    name='distillate',
    version='0.1dev',
    install_requires=[
        'click',
        'tqdm',
        'Pillow',
        'google-cloud-storage',
        'beautifulsoup4',
        "layoutlm @ git+git://github.com/microsoft/unilm.git#egg=pkg&subdirectory=layoutlm"
    ],
    packages=['distillate',],
    entry_points={
        'console_scripts': [
            'distillate = distillate.commands.commands:cli',
        ],
    },
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    long_description=open('README.md').read(),
    test_suite='nose.collector',
    tests_require=['nose']
)
