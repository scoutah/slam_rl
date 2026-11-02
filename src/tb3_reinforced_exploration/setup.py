from setuptools import find_packages, setup

package_name = 'tb3_reinforced_exploration'

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
            'reinforced_exploration = tb3_reinforced_exploration.rl_agent:main',
            'rl_explorer = tb3_reinforced_exploration.multi_robot_explorer:main',
        ],
    },
)
