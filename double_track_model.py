import numpy as np
from scipy.optimize import root

class Vehicle:
    def __init__(
        self,
        track_back_m: int = 1.5,
        track_front_m: int = 1.5,
        cg_to_front_m: int = 2,
        cg_to_back_m: int = 2,
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
        self.cg_to_back_m = cg_to_back_m
        self.wheelbase_m = cg_to_front_m + cg_to_back_m
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
        self,
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
        track_mean_m = np.mean([track_front_m, track_back_m])
        yaw_inertia_kgm2 = 1/12*mass_kg*(track_mean_m**2 + wheelbase_m**2)
        yaw_inertia_kgm2 = yaw_inertia_multiplier*yaw_inertia_kgm2

        return yaw_inertia_kgm2

class Tire:
    def __init__(
        self,
        cornering_stiffness_NperRad: int = 30000,
        relaxation_length_m: int = 5
        ):
        self.cornering_stiffness_NperRad = cornering_stiffness_NperRad
        self.relaxation_length_m = relaxation_length_m

class Simulation:
    '''
    Subscript reference:
    _br = _back_right
    _bl = _back_left
    _fr = _front_right
    _fl = _front_left
    '''
    def __init__(
        self,
        velocity_kph: int,
        steer_before_rack_deg: int,
        car: Vehicle,
        tire_front: Tire,
        tire_rear: Tire
    ):
        self.velocity_kph = velocity_kph
        self.velocity_mps = velocity_kph/3.6
        self.steer_before_rack_deg = steer_before_rack_deg
        self.steer_before_rack_rad = np.deg2rad(self.steer_before_rack_deg)
        self.car = car
        self.tire_front = tire_front
        self.tire_rear = tire_rear
        self.steer_after_rack_deg = self.steer_before_rack_deg/car.steer_ratio
        self.steer_after_rack_rad = self.steer_before_rack_rad/car.steer_ratio
        
        # First Guess of Vehicle state parameters
        self.path_radius_m = car.wheelbase_m/self.steer_after_rack_rad
        self.side_slip_rad = car.cg_to_back_m/self.path_radius_m
        self.side_slip_deg = np.rad2deg(self.side_slip_rad)
        self.accel_cent_mps2 = self.velocity_mps**2/self.path_radius_m

        # Initialization of wheel/tire angles
        self.steer_fl_rad = 0
        self.steer_fr_rad = 0
        self.alpha_bl_rad = 0
        self.alpha_br_rad = 0
        self.alpha_fl_rad = 0
        self.alpha_fr_rad = 0

        # Initialization of convertion to degrees
        self.steer_fl_deg = 0
        self.steer_fr_deg = 0
        self.alpha_bl_deg = 0
        self.alpha_br_deg = 0
        self.alpha_fl_deg = 0
        self.alpha_fr_deg = 0

        # Initialization of tire forces
        self.forceX_fl_N = 0
        self.forceY_fl_N = 0
        self.forceX_fr_N = 0
        self.forceY_fr_N = 0
        self.forceX_bl_N = 0
        self.forceY_bl_N = 0
        self.forceX_br_N = 0
        self.forceY_br_N = 0

        # Initialization of resultant forces/moments
        self.force_centripetal_N = 0
        self.force_centrifugal_N = 0
        self.yaw_moment_Nm = 0

    def calc_wheel_angles(self):
        steer = self.steer_after_rack_rad
        beta = self.side_slip_rad
        T_f = self.car.track_front_m
        R = self.path_radius_m
        ack = self.car.ackermann_factor
        
        # Steering angle from individual wheels
        self.steer_fl_rad = np.atan2(steer, np.cos(beta) - ack*T_f/(2*R))
        self.steer_fr_rad = np.atan2(steer, np.cos(beta) + ack*T_f/(2*R))

        # Conversion to degrees
        self.steer_fl_deg = np.rad2deg(self.steer_fl_rad)
        self.steer_fr_deg = np.rad2deg(self.steer_fr_rad)
        
        b = self.car.cg_to_back_m
        toe_b = self.car.toe_back_rad

        # Influence of track width on kinematic parameters
        factor_left = np.cos(beta) - T_f/(2*R)
        factor_right = np.cos(beta) + T_f/(2*R)

        # Tire slip angles (Rear)
        self.alpha_bl_rad = \
            np.atan2(np.sin(beta) + b/R + toe_b*factor_left, factor_left)
        self.alpha_br_rad = \
            np.atan2(np.sin(beta) + b/R - toe_b*factor_right, factor_right)
        
        # Conversion to degrees
        self.alpha_bl_deg = np.rad2deg(self.alpha_bl_rad)
        self.alpha_br_deg = np.rad2deg(self.alpha_br_rad)

        a = self.car.cg_to_front_m
        d_fr = self.steer_fr_rad
        d_fl = self.steer_fl_rad
        toe_f = self.car.toe_front_rad

        # Tire slip angles (Front)
        self.alpha_fl_rad = np.atan2(np.sin(beta) - 
                            a/R + (d_fl - toe_f)*factor_left, factor_left)
        self.alpha_fr_rad = np.atan2(np.sin(beta) - 
                            a/R + (d_fr + toe_f)*factor_right, factor_right)
        
        # Conversion to degrees
        self.alpha_fl_deg = np.rad2deg(self.alpha_fl_rad)
        self.alpha_fr_deg = np.rad2deg(self.alpha_fr_rad)

    def calc_forces(self):

        Cr = self.tire_rear.cornering_stiffness_NperRad
        toe_b = self.car.toe_back_rad
        
        self.forceY_bl_N = -Cr*(self.alpha_bl_rad)*np.cos(toe_b)
        self.forceX_bl_N = -Cr*(self.alpha_bl_rad)*np.sin(toe_b)
        
        self.forceY_br_N = -Cr*(self.alpha_br_rad)*np.cos(toe_b)
        self.forceX_br_N = -Cr*(self.alpha_br_rad)*np.sin(toe_b)

        Cf = self.tire_front.cornering_stiffness_NperRad
        toe_f = self.car.toe_front_rad
        d_fl = self.steer_fl_rad
        d_fr = self.steer_fr_rad
        
        self.forceY_fl_N = -Cf*(self.alpha_fl_rad*np.cos(d_fl-toe_f))
        self.forceX_fl_N = -Cf*(self.alpha_fl_rad*np.sin(d_fl-toe_f))
        
        self.forceY_fr_N = -Cf*(self.alpha_fr_rad*np.cos(d_fr+toe_f))
        self.forceX_fr_N = -Cf*(self.alpha_fr_rad*np.sin(d_fr+toe_f))
    
    def calc_static_solution(self):

        def objective_function(x):
            self.path_radius_m = x[0]
            self.side_slip_rad = x[1]
            self.side_slip_deg = np.rad2deg(self.side_slip_rad)
            self.accel_cent_mps2 = self.velocity_mps**2/self.path_radius_m

            self.calc_wheel_angles()
            self.calc_forces()

            # Rotate forces to tangential-normal coordinates to velocity vector
            to_normal = lambda x,y,ang: x*np.sin(ang) + y*np.cos(ang) 
            
            force_norm_fl = \
                to_normal(self.forceX_fl_N, self.forceY_fl_N, self.side_slip_rad)
            force_norm_fr = \
                to_normal(self.forceX_fr_N, self.forceY_fr_N, self.side_slip_rad)
            force_norm_bl = \
                to_normal(self.forceX_bl_N, self.forceY_bl_N, self.side_slip_rad)
            force_norm_br = \
                to_normal(self.forceX_br_N, self.forceY_br_N, self.side_slip_rad)
            
            # Dynamical equilibrium condition for normal forces
            self.force_centripetal_N = \
                force_norm_fl + force_norm_fr + force_norm_bl + force_norm_br
            self.force_centrifugal_N = \
                self.car.mass_kg*self.velocity_mps**2/self.path_radius_m
            
            # Dynamical equilibrium condition for yaw moment
            self.yaw_moment_Nm = \
                + self.car.cg_to_front_m*self.forceY_fl_N \
                - self.car.track_front_m/2*self.forceX_fl_N \
                + self.car.cg_to_front_m*self.forceY_fr_N \
                + self.car.track_front_m/2*self.forceX_fr_N \
                - self.car.cg_to_back_m*self.forceY_bl_N \
                - self.car.track_back_m/2*self.forceX_bl_N \
                - self.car.cg_to_back_m*self.forceY_br_N \
                + self.car.track_back_m/2*self.forceX_br_N
            
            return [self.force_centripetal_N - self.force_centrifugal_N,
                     self.yaw_moment_Nm]
        
        x0 = [self.path_radius_m, self.side_slip_rad]
        sol = root(objective_function, x0 = x0)
        print(f"Root found at: sol.x")

if __name__ == "__main__":
    car = Vehicle(cg_to_front_m=2, cg_to_back_m=2)
    tire_front = Tire(cornering_stiffness_NperRad=20000)
    tire_rear = Tire()
    simulation = Simulation(velocity_kph=100, steer_before_rack_deg=0.5, 
                        car=car, tire_front=tire_front, tire_rear=tire_rear)
    
    simulation.calc_static_solution()