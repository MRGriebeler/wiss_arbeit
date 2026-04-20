import numpy as np

class Vehicle:
    def __init__(
        self,
        track_back_m: int = 1.5,
        track_front_m: int = 1.5,
        cg_to_front_m: int = 2,
        cg_to_rear_m: int = 2,
        mass_kg: int = 1000,
        yaw_inertia_multiplier: int = 1,
        steer_ratio: int = 1,
        toe_front_deg: int = 0,
        toe_back_deg:  int = 0,
        ackermann_factor: int = 1
        ):

        self.track_back_m = track_back_m
        self.track_front_m = track_front_m
        self.cg_to_front_m = cg_to_front_m
        self.cg_to_rear_m = cg_to_rear_m
        self.wheelbase_m = cg_to_front_m + cg_to_rear_m
        self.mass_kg = mass_kg
        self.yaw_inertia_multiplier = yaw_inertia_multiplier
        
        self.yaw_inertia_kgm2 = self.calculate_inertia(
            track_front_m,
            track_back_m,
            self.wheelbase_m,
            mass_kg,
            yaw_inertia_multiplier)
        
        self.steer_ratio = steer_ratio
        self.toe_front_deg = toe_front_deg
        self.toe_front_rad = np.deg2rad(toe_front_deg)
        self.toe_back_deg = toe_back_deg
        self.toe_back_rad = np.deg2rad(toe_back_deg)
        self.ackermann_factor = ackermann_factor

    def calculate_inertia(
        track_front_m: int,
        track_back_m: int,
        wheelbase_m: int,
        mass_kg: int,
        yaw_inertia_multiplier: int
        ) -> int:
        '''
        Yaw inertia is calculated based on the moment of inertia of a rectangle
        with the previously informed dimensions
        '''
        track_mean_m = np.mean(track_front_m, track_back_m)
        yaw_inertia_kgm2 = 1/12*mass_kg*(track_mean_m**2 + wheelbase_m**2)
        yaw_inertia_kgm2 = yaw_inertia_multiplier*yaw_inertia_kgm2

        return yaw_inertia_kgm2

class Tire:
    def __init__(self, cornering_stiffness_NperRad, relaxation_length_m):
        cornering_stiffness_NperRad: int = 3000
        relaxation_length_m: int = 5
