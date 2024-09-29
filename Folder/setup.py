from typing import List
from setuptools import find_packages, setup

def get_requirements(file_path: str) -> List[str]:
    requirements = []
    with open(file_path) as fileObj:
        requirements = fileObj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if '-e .' in requirements:
            requirements.remove('-e .')
    return requirements


setup(
    name = "",
    version= "0.0.1",
    author = "Rishik Reddy Yesgari",
    author_email= "rishikreddy.yesgari@gmail.com",
    packages= find_packages(),
    install_requires = get_requirements('requirements.txt')
)