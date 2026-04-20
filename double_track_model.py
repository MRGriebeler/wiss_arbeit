import numpy as np

#1 Vehicle parameters:

#1.1 Dimensions
track_back_m = 1.5
track_front_m = 1.5
cg_to_front_m = 2
cg_to_rear_m = 2

#1.2 Inertia
mass_kg = 1000
yaw_inertia_multiplier = 1

#1.3 Steering system
steer_ratio = 1
toe_front_deg = 0
toe_back_deg = 0
ackermann_factor = 1
'''
1 = 100% Ackermann
0 = Parallel steering
-1 = 100% Anti-Ackermann
'''

# Yaw Inertia Calculation
track_mean_m = np.mean(track_front_m, track_back_m)
wheelbase = cg_to_front_m + cg_to_rear_m
yaw_inertia_kgm2 = 1/12*mass_kg*(track_mean_m**2 + wheelbase**2)
yaw_inertia_kgm2 = yaw_inertia_multiplier*yaw_inertia_kgm2
'''
Yaw inertia is calculated based on the moment of inertia of a rectangle
with the previously informed dimensions
'''

#2 Tire parameters
cornering_stiffness_NperRad = 30000
relaxation_length_m = 5