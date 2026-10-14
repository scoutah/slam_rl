from glob import glob
import os

from setuptools import find_packages, setup

package_name = 'tb3_slam'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob(os.path.join('launch', '*launch.py'))),
        (os.path.join('share', package_name, 'rviz'), glob(os.path.join('rviz', '*.rviz'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='tombielert',
    maintainer_email='Tom.Bielert@deutschebahn.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'cmd_vel_relay = tb3_slam.cmd_vel_relay:main',
            'episode_manager = tb3_slam.episode_manager:main',
            'map_merger = tb3_slam.map_merger:main',
        ],
    },
)
