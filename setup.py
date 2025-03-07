from setuptools import setup, find_packages

#TODO(AOS) Add missing fields
setup(
    name="pytop",
    version='1.0.0',
    packages=['pytop'],
    entry_points={
        "console_scripts": [
            "pytop=pytop:main",
        ],
    },
    install_requires=[
        'psutil>=5.8.0',
        'urwid>=2.1.0',
    ],
    extra_requires={
        "test": [
            'pytest>=5.3.1',
        ],
    },
    description='Htop copycat implemented in Python.',
    author='Andrii Oshtuk',
    url='https://github.com/AndriiOshtuk/pytop',
    license='MIT',
    platforms='Linux',
)
