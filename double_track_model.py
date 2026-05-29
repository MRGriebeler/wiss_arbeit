import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt
from typing import Any

class Vehicle:
    def __init__(
        self,
        track_back_m: float = 1.5,
        track_front_m: float = 1.5,
        cg_to_front_m: float = 2.0,
        cg_to_back_m: float = 2.0,
        mass_kg: float = 1000,
        yaw_inertia_multiplier: float = 1.0,
        steer_ratio: float = 1.0,
        toe_front_deg: float = 0.0,
        toe_back_deg:  float = 0.0,
        ackermann_factor: float = 1.0
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
        track_front_m: float,
        track_back_m: float,
        wheelbase_m: float,
        mass_kg: float,
        yaw_inertia_multiplier: float
        ) -> np.floating[Any]:
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
        car: Vehicle,
        tire_front: Tire,
        tire_rear: Tire
    ):
        self.car = car
        self.tire_front = tire_front
        self.tire_rear = tire_rear
        
        # Initialization of vehicle state variables
        self.position_x_m = 0
        self.position_y_m = 0
        self.yaw_angle_deg = 90
        self.yaw_angle_rad = np.deg2rad(self.yaw_angle_deg)
        self.velocity_kph = 0
        self.velocity_mps = 0
        self.steer_before_rack_deg = 0
        self.steer_before_rack_rad = 0
        self.steer_after_rack_deg = 0
        self.steer_after_rack_rad = 0
        self.path_radius_m = 0
        self.side_slip_rad = 0
        self.side_slip_deg = 0
        self.accel_cent_mps2 = 0

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
        self.accel_cent_mps2 = self.velocity_mps**2/R
        
        # Conversion to degrees
        self.steer_after_rack_deg = np.rad2deg(self.steer_after_rack_rad)
        self.side_slip_deg = np.rad2deg(self.side_slip_rad)
        
        # Steering angle from individual wheels
        self.steer_fl_rad = np.atan2(steer, 1    - ack*T_f/(2*R))
        self.steer_fr_rad = np.atan2(steer, 1 + ack*T_f/(2*R))
        # TO DO - sign inversion should depend on the steering sign instead
        # of curvature radius sign

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
            np.atan2(np.sin(beta) - b/R, factor_left) + toe_b*np.sign(R)
        self.alpha_br_rad = \
            np.atan2(np.sin(beta) - b/R, factor_right) - toe_b*np.sign(R)
        
        # Conversion to degrees
        self.alpha_bl_deg = np.rad2deg(self.alpha_bl_rad)
        self.alpha_br_deg = np.rad2deg(self.alpha_br_rad)

        a = self.car.cg_to_front_m
        toe_f = self.car.toe_front_rad

        # Tire slip angles (Front)
        self.alpha_fl_rad = np.atan2(np.sin(beta) + a/R, factor_left) \
            - self.steer_fl_rad + toe_f*np.sign(R)
        self.alpha_fr_rad = np.atan2(np.sin(beta) + a/R, factor_right) \
            - self.steer_fr_rad - toe_f*np.sign(R)
        
        # Conversion to degrees
        self.alpha_fl_deg = np.rad2deg(self.alpha_fl_rad)
        self.alpha_fr_deg = np.rad2deg(self.alpha_fr_rad)

    def calc_tire_forces(self):

        Cr = self.tire_rear.cornering_stiffness_NperRad
        toe_b = self.car.toe_back_rad
        R = self.path_radius_m
        
        self.forceY_bl_N = -Cr*(self.alpha_bl_rad)*np.sin(np.pi/2 - toe_b)
        self.forceX_bl_N = -Cr*(self.alpha_bl_rad)*np.cos(np.pi/2 - toe_b)
        
        self.forceY_br_N = -Cr*(self.alpha_br_rad)*np.sin(np.pi/2 + toe_b)
        self.forceX_br_N = -Cr*(self.alpha_br_rad)*np.cos(np.pi/2 + toe_b)

        Cf = self.tire_front.cornering_stiffness_NperRad
        toe_f = self.car.toe_front_rad
        d_fl = self.steer_fl_rad
        d_fr = self.steer_fr_rad
        
        self.forceY_fl_N = -Cf*(self.alpha_fl_rad*np.sin(np.pi/2 + d_fl - toe_f))
        self.forceX_fl_N = -Cf*(self.alpha_fl_rad*np.cos(np.pi/2 + d_fl - toe_f))
        
        self.forceY_fr_N = -Cf*(self.alpha_fr_rad*np.sin(np.pi/2 + d_fr + toe_f))
        self.forceX_fr_N = -Cf*(self.alpha_fr_rad*np.cos(np.pi/2 + d_fr + toe_f))

        to_normal = lambda x,y,ang: x*np.sin(ang) + y*np.cos(ang)

        self.force_norm_fl = \
            to_normal(self.forceX_fl_N, self.forceY_fl_N, self.side_slip_rad)
        self.force_norm_fr = \
            to_normal(self.forceX_fr_N, self.forceY_fr_N, self.side_slip_rad)
        self.force_norm_bl = \
            to_normal(self.forceX_bl_N, self.forceY_bl_N, self.side_slip_rad)
        self.force_norm_br = \
            to_normal(self.forceX_br_N, self.forceY_br_N, self.side_slip_rad)
    
    def calc_yaw_moment(self):
        self.yaw_moment_Nm = \
            + self.car.cg_to_front_m*self.forceY_fl_N \
            - self.car.track_front_m/2*self.forceX_fl_N \
            + self.car.cg_to_front_m*self.forceY_fr_N \
            + self.car.track_front_m/2*self.forceX_fr_N \
            - self.car.cg_to_back_m*self.forceY_bl_N \
            - self.car.track_back_m/2*self.forceX_bl_N \
            - self.car.cg_to_back_m*self.forceY_br_N \
            + self.car.track_back_m/2*self.forceX_br_N

    def visualize_vehicle_state(self):
        
        rot_x_pts = lambda x,y,ang: x*np.cos(ang) - y*np.sin(ang)
        rot_y_pts = lambda x,y,ang: x*np.sin(ang) + y*np.cos(ang)

        local_variables = locals()
        if 'fig' not in local_variables and 'ax' not in local_variables:
            fig, ax = plt.subplots()

        a = self.car.cg_to_front_m
        b = self.car.cg_to_back_m
        wf = self.car.track_front_m
        wb = self.car.track_back_m

        body_x_for_plot = np.array([+a,+a,+a,-b,-b,-b])
        body_y_for_plot = np.array([+wf/2,-wf/2,0,0,+wb/2,-wb/2])

        vel_dir_glob_ref_rad = self.yaw_angle_rad + self.side_slip_rad

        body_x_for_plot_glob_ref = \
            rot_x_pts(body_x_for_plot,
                      body_y_for_plot,
                      self.yaw_angle_rad)
        
        body_y_for_plot_glob_ref = \
            rot_y_pts(body_x_for_plot,
                      body_y_for_plot,
                      self.yaw_angle_rad)

        inputs = (body_x_for_plot_glob_ref, body_y_for_plot_glob_ref)
        
        body_lines_glob_ref = ax.plot(*inputs)

        body_pts_x_m = np.array([+a,+a,-b,-b])
        body_pts_y_m = np.array([+wf/2,-wf/2,-wb/2,+wb/2])

        body_x_glob_ref = \
            rot_x_pts(body_pts_x_m,
                      body_pts_y_m,
                      self.yaw_angle_rad)
        
        body_y_glob_ref = \
            rot_y_pts(body_pts_x_m,
                      body_pts_y_m,
                      self.yaw_angle_rad)
        
        wheel_pts_x_m = np.array([+0.5, +0.5, -0.5, -0.5])
        wheel_pts_y_m = np.array([+0.2, -0.2, -0.2, +0.2])
        
        R = self.path_radius_m
        
        wheel_angle_rad = [
            self.steer_fl_rad - np.sign(R)*self.car.toe_front_rad,
            self.steer_fr_rad + np.sign(R)*self.car.toe_front_rad,
            +np.sign(self.steer_fr_rad)*self.car.toe_back_rad,
            -np.sign(self.steer_fl_rad)*self.car.toe_back_rad]

        wheel_lines = list()
        for x_shift, y_shift, ang in zip(body_pts_x_m,
                                         body_pts_y_m,
                                         wheel_angle_rad):
            
            wheel_x_car_ref = x_shift + \
                rot_x_pts(wheel_pts_x_m, wheel_pts_y_m, ang)
            wheel_y_car_ref = y_shift + \
                rot_y_pts(wheel_pts_x_m, wheel_pts_y_m, ang)
            
            wheel_x_glob_ref = \
                rot_x_pts(wheel_x_car_ref,
                          wheel_y_car_ref,
                          self.yaw_angle_rad)
            wheel_y_glob_ref = \
                rot_y_pts(wheel_x_car_ref,
                          wheel_y_car_ref,
                          self.yaw_angle_rad)

            inputs = (np.append(wheel_x_glob_ref, wheel_x_glob_ref[0]),
                      np.append(wheel_y_glob_ref, wheel_y_glob_ref[0]))
            
            wheel_lines.append(ax.plot(*inputs, color='black'))

        wheel_forcesX = [self.forceX_fl_N,
                         self.forceX_fr_N,
                         self.forceX_br_N,
                         self.forceX_bl_N]
        
        wheel_forcesY = [self.forceY_fl_N,
                         self.forceY_fr_N,
                         self.forceY_br_N,
                         self.forceY_bl_N]

        force_graphics = list()
        for forceX, forceY, locationX, locationY in zip(wheel_forcesX,
                                                        wheel_forcesY,
                                                        body_x_glob_ref,
                                                        body_y_glob_ref):
        
            forceX_glob_ref = rot_x_pts(forceX,
                                        forceY,
                                        self.yaw_angle_rad)
            
            forceY_glob_ref = rot_y_pts(forceX,
                                        forceY,
                                        self.yaw_angle_rad)

            force_graphics.append(
                ax.quiver(
                    locationX,
                    locationY,
                    forceX_glob_ref,
                    forceY_glob_ref,
                    scale=1000,
                    scale_units='x',
                    width=0.005,
                    headlength=3,
                    headaxislength=3,
                    color='red')
            
                # ax.annotate(
                #     text = f"{np.hypot(self.forceX_fl_N, self.forceY_fl_N):.2f}",
                #     xy = (locationX, locationY),
                #     xytext = (locationX + forceX_glob_ref/1000,
                #         locationY + forceY_glob_ref/1000),
                #     # xy = (locationX + forceX_glob_ref/1000,
                #     #     locationY + forceY_glob_ref/1000),
                #     # xytext = (locationX, locationY),
                #     arrowprops = dict(
                #                 arrowstyle = '<-',
                #                 relpos = (0,0),
                #                 shrinkA = 0,
                #                 shrinkB = 0,
                #                 edgecolor = 'red'
                #                 )
                # )
                )

            vel_cg_x = self.velocity_mps*np.cos(self.side_slip_rad)
            vel_cg_y = self.velocity_mps*np.sin(self.side_slip_rad)
            vel_cg_x_glob_ref = rot_x_pts(vel_cg_x, vel_cg_y, self.yaw_angle_rad)
            vel_cg_y_glob_ref = rot_y_pts(vel_cg_x, vel_cg_y, self.yaw_angle_rad)
            vel_cg_graphic = \
                ax.quiver(
                    0,
                    0,
                    vel_cg_x_glob_ref,
                    vel_cg_y_glob_ref,
                    scale=10,
                    scale_units='x',
                    width=0.005,
                    headlength=3,
                    headaxislength=3,
                    color='black')
                

        ax.set_aspect('equal')
        ax.grid(visible=True)
        fig.show()
        print()

    def find_stationary_point(self,
                              state1_name: str,
                              state1_value_SI: float,
                              state2_name: str,
                              state2_value_SI: float,
                              state3_name = None,
                              state3_initial_guess_SI = None,
                              state4_name = None,
                              state4_initial_guess_SI = None):
        
        def validate_state_input(input, valid_state_names):
            """ Validate whether 'state' has unique partial 
            match in the valid_state_input"""

            matches = [state for state in valid_state_names 
                       if state.startswith(input.lower())]
            
            if len(matches) == 0:
                raise ValueError(f"No match found for '{input}'. \
                                 Options: {valid_state_names}")
            elif len(matches) > 1:
                raise ValueError(f"Ambiguous match '{input}'. \
                                 Could be {matches}")
            
            return matches[0]
        
        def objective_function(x):

            setattr(self, states_dict[state3_name], x[0])
            setattr(self, states_dict[state4_name], x[1])

            self.calc_wheel_angles()
            self.calc_tire_forces()

            self.force_centripetal_N = \
                self.force_norm_fl + self.force_norm_fr + \
                    self.force_norm_bl + self.force_norm_br

            self.force_centrifugal_N = \
                self.car.mass_kg*self.velocity_mps**2/self.path_radius_m
            
            centrip_minus_centrif = \
                self.force_centripetal_N - self.force_centrifugal_N
            
            self.calc_yaw_moment()
            
            return [centrip_minus_centrif, self.yaw_moment_Nm]
        
        valid_state_names = \
            ["velocity", "steering", "radius", "sideslip"]

        state1_name = validate_state_input(state1_name, valid_state_names)
        state2_name = validate_state_input(state2_name, valid_state_names)

        states_dict = {
                    "radius": "path_radius_m",
                    "sideslip": "side_slip_rad",
                    "velocity": "velocity_mps",
                    "steering": "steer_after_rack_rad",
                }

        setattr(self, states_dict[state1_name], state1_value_SI)
        setattr(self, states_dict[state2_name], state2_value_SI)

        initial_guesses_dict = {
            "radius": -30.0,
            "sideslip": np.deg2rad(5),
            "velocity": 50/3.6,
            "steering": np.deg2rad(-20),
            # Initial guesses need to adapt depending on a predicted left or
            # right hand turn
        }
        
        states3and4 = [x for x in valid_state_names 
                       if x not in [state1_name, state2_name]]

        if state3_name == None:           
            state3_name = states3and4[0]
            state3_initial_guess_SI = initial_guesses_dict[state3_name]
        else:
            state3_name = validate_state_input(state3_name, 
                                               valid_state_names=states3and4)
            
        if state4_name == None:
            state4_name = states3and4[1]
            state4_initial_guess_SI = initial_guesses_dict[state4_name]
        else:
            state4_name = validate_state_input(state4_name,
                                               valid_state_names=states3and4)


        x0 = [state3_initial_guess_SI, state4_initial_guess_SI]
        
        sol = root(objective_function, x0 = x0)
        self.visualize_vehicle_state()

if __name__ == "__main__":
    car = Vehicle(cg_to_front_m = 2.0,
                  cg_to_back_m = 2.0,
                  toe_front_deg = 0.0,
                  toe_back_deg = 0.0)
    
    tire_front = Tire(cornering_stiffness_NperRad=20000)
    tire_rear = Tire()
    simulation = Simulation(car=car,
                            tire_front=tire_front,
                            tire_rear=tire_rear)
    
    simulation.find_stationary_point(state1_name="steer",
                                     state1_value_SI=np.deg2rad(-5),
                                     state2_name="vel",
                                     state2_value_SI=100/3.6)