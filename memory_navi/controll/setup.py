from setuptools import find_packages, setup

package_name = 'controll'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='kuko',
    maintainer_email='kuko@todo.todo',
    description='Memory-Navi control layer — task decomposition, multi-model arbitration, and navigation decision engine.',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'control = controll.control:main'
        ],
    },
)
