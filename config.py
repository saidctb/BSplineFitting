"""
Settings for the whole project
"""
# The range of number of control points
N_min = 4
N_max = 15
# Image number of open and closed curves for a fixed number of control points
# The first element is for train dataset and the second element is for validation
Images_op = [1000, 200]
Images_cl = [1000, 200]
# Degree of generated curves
Deg = 3
# Image settings
Width = 512
Height = 512
Dpi = 100
Linewidth = 2.5
# Training parameters
Epoch = 30
Model_types = ['simple']
Curve_types = ['open', 'closed', 'all']
