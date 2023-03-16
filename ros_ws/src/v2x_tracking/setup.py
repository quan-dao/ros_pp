from distutils.core import setup
setup(
    name='v2x_tracking',
    version='0.0.1',
    packages=['v2x_tracking'],
    install_requires=[
    'rospy',
    'math',
    'torch',
    'matplotlib.pyplot'
    ],
    scripts = ['scripts/v2x_tracking_node.py','scripts/viz_node.py','scripts/communication_node.py' ],
    package_dir={'': 'src'}
)