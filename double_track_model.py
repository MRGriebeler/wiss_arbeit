import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt
from typing import Any, List

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

        # Initialization of tire moments
        self.moment_yaw_tire_fl_Nm = 0
        self.moment_yaw_tire_fr_Nm = 0
        self.moment_yaw_tire_bl_Nm = 0
        self.moment_yaw_tire_br_Nm = 0

        # Initialization of axle moments
        self.moment_yaw_axle_f_Nm = 0
        self.moment_yaw_axle_b_Nm = 0
        
        # Initialization of resultant forces/moments
        self.force_centripetal_N = 0
        self.force_centrifugal_N = 0
        self.moment_yaw_Nm = 0

        # Initialization of simulation related parameters
        self.valid_state_names = \
            ["velocity", "steering", "radius", "sideslip"]
        self.state_name_dict = {
                    "radius": "path_radius_m",
                    "sideslip": "side_slip_rad",
                    "velocity": "velocity_mps",
                    "steering": "steer_after_rack_rad"
                }

    def calc_wheel_angles(self):

        steer = self.steer_after_rack_rad
        toe_b = self.car.toe_back_rad
        toe_f = self.car.toe_front_rad
        beta = self.side_slip_rad
        wf = self.car.track_front_m
        wb = self.car.track_back_m
        lf = self.car.cg_to_front_m
        lb = self.car.cg_to_back_m
        R = self.path_radius_m
        ack = self.car.ackermann_factor
        self.accel_cent_mps2 = self.velocity_mps**2/R
        
        # Conversion to degrees
        self.steer_after_rack_deg = np.rad2deg(self.steer_after_rack_rad)
        self.side_slip_deg = np.rad2deg(self.side_slip_rad)
        
        # Steering angle from individual wheels
        self.steer_fl_rad = \
            np.atan2(np.tan(steer), 1 - ack*wf/2*np.tan(steer)/(lf+lb))
        self.steer_fr_rad = \
            np.atan2(np.tan(steer), 1 + ack*wf/2*np.tan(steer)/(lf+lb))

        # Conversion to degrees
        self.steer_fl_deg = np.rad2deg(self.steer_fl_rad)
        self.steer_fr_deg = np.rad2deg(self.steer_fr_rad)

        # Tire slip angles (Rear)
        self.alpha_bl_rad = \
            np.atan2(np.sin(beta) - lb/R, np.cos(beta) - wb/2/R) \
                + toe_b*np.sign(R)
        
        self.alpha_br_rad = \
            np.atan2(np.sin(beta) - lb/R, np.cos(beta) + wb/2/R) \
                - toe_b*np.sign(R)
        
        # Conversion to degrees
        self.alpha_bl_deg = np.rad2deg(self.alpha_bl_rad)
        self.alpha_br_deg = np.rad2deg(self.alpha_br_rad)

        # Tire slip angles (Front)
        self.alpha_fl_rad = \
            np.atan2(np.sin(beta) + lf/R, np.cos(beta) - wf/2/R) \
                    - self.steer_fl_rad + toe_f*np.sign(R)
        
        self.alpha_fr_rad = \
            np.atan2(np.sin(beta) + lf/R, np.cos(beta) + wf/2/R) \
                    - self.steer_fr_rad - toe_f*np.sign(R)
        
        # Conversion to degrees
        self.alpha_fl_deg = np.rad2deg(self.alpha_fl_rad)
        self.alpha_fr_deg = np.rad2deg(self.alpha_fr_rad)

    def calc_forces(self):

        Cb = self.tire_rear.cornering_stiffness_NperRad
        toe_b = self.car.toe_back_rad
        a_bl = self.alpha_bl_rad
        a_br = self.alpha_br_rad
        
        self.forceX_bl_N = +Cb*(a_bl)*np.sin(-toe_b)
        self.forceY_bl_N = -Cb*(a_bl)*np.cos(-toe_b)
        
        self.forceX_br_N = +Cb*(a_br)*np.sin(+toe_b)
        self.forceY_br_N = -Cb*(a_br)*np.cos(+toe_b)

        Cf = self.tire_front.cornering_stiffness_NperRad
        toe_f = self.car.toe_front_rad
        d_fl = self.steer_fl_rad
        d_fr = self.steer_fr_rad
        a_fl = self.alpha_fl_rad
        a_fr = self.alpha_fr_rad

        self.forceX_fl_N = +Cf*(a_fl*np.sin(d_fl - toe_f))
        self.forceY_fl_N = -Cf*(a_fl*np.cos(d_fl - toe_f))
        
        self.forceX_fr_N = +Cf*(a_fr*np.sin(d_fr + toe_f))
        self.forceY_fr_N = -Cf*(a_fr*np.cos(d_fr + toe_f))

        beta = self.side_slip_rad
        self.force_norm_fl = Cf*a_fl*np.cos(beta - d_fl)
        self.force_norm_fr = Cf*a_fr*np.cos(beta - d_fr)
        self.force_norm_bl = Cb*a_bl*np.cos(beta)
        self.force_norm_br = Cb*a_br*np.cos(beta)
    
    def calc_moments(self):
        self.moment_yaw_tire_fl_Nm = \
            + self.car.cg_to_front_m*self.forceY_fl_N \
            - self.car.track_front_m/2*self.forceX_fl_N
        
        self.moment_yaw_tire_fr_Nm = \
            + self.car.cg_to_front_m*self.forceY_fr_N \
            + self.car.track_front_m/2*self.forceX_fr_N
        
        self.moment_yaw_tire_bl_Nm = \
            - self.car.cg_to_back_m*self.forceY_bl_N \
            - self.car.track_back_m/2*self.forceX_bl_N
        
        self.moment_yaw_tire_br_Nm = \
            - self.car.cg_to_back_m*self.forceY_br_N \
            + self.car.track_back_m/2*self.forceX_br_N

        self.moment_yaw_axle_f_Nm = \
            self.moment_yaw_tire_fl_Nm + self.moment_yaw_tire_fr_Nm
        
        self.moment_yaw_axle_b_Nm = \
            self.moment_yaw_tire_bl_Nm + self.moment_yaw_tire_br_Nm
        
        self.moment_yaw_Nm = \
            self.moment_yaw_axle_f_Nm + self.moment_yaw_axle_b_Nm
            
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
                              state_names: List[str],
                              state_values_SI: List[float]):
        
        def parse_state_input(input_state_names,
                              input_state_values_SI):
            
            def input_validation(input, valid_state_names):
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
            
            validated_state_names = list()
            for state_name, state_value in \
                zip(input_state_names, input_state_values_SI):
                
                validated_state_name = \
                    input_validation(state_name, self.valid_state_names)
                
                validated_state_names.append(validated_state_name)

                attribute_name = self.state_name_dict[validated_state_name]
                setattr(self, attribute_name, state_value)
            
            if len(validated_state_names) != 2:
                raise ValueError(f"Exactly two states from \
                                 '{self.valid_state_names}' should be informed")
            
            return validated_state_names
        
        def calc_init_guess(state_names):
            m = self.car.mass_kg
            lf = self.car.cg_to_front_m
            lb = self.car.cg_to_back_m
            mf = m*lb/(lf+lb)
            mb = m*lf/(lf+lb)
            cf_twin = 2*self.tire_front.cornering_stiffness_NperRad
            cb_twin = 2*self.tire_rear.cornering_stiffness_NperRad

            EG = mf/cf_twin - mb/cb_twin
            SG = -mb/cb_twin

            # {"velocity", "steering", "radius", "sideslip"}
            if set(state_names) == set(["velocity", "steering"]):
                v = self.velocity_mps
                s = self.steer_after_rack_rad
                
                R = ((lf+lb) + EG*v**2)/s
                slip = s*((lb + SG*v**2)/((lf+lb) + EG*v**2))

                states_to_solve_for = ["radius", "sideslip"]
                x0 = [R, slip]

            elif set(state_names) == set(["velocity", "radius"]):
                v = self.velocity_mps
                R = self.path_radius_m

                s = ((lf+lb) + EG*v**2)/R
                slip = (lb + SG*v**2)/R

                states_to_solve_for = ["sideslip", "steer"]
                x0 = [slip, s]

            elif set(state_names) == set(["velocity", "sideslip"]):
                v = self.velocity_mps
                slip = self.side_slip_rad
                
                R = (lb + SG*v**2)/slip
                s = slip*((lf+lb) + EG*v**2)/(lb + SG*v**2)
                
                states_to_solve_for = ["radius", "steer"]
                x0 = [R, s]

            elif set(state_names) == set(["steering", "radius"]):
                s = self.steer_after_rack_rad
                R = self.path_radius_m

                v = np.sqrt((R*s - (lf+lb))/EG)
                slip = lb/R + SG/EG*(s - (lf+lb)/R)

                states_to_solve_for = ["velocity", "sideslip"]
                x0 = [v, slip]

            elif set(state_names) == set(["steering", "sideslip"]):
                s = self.steer_after_rack_rad
                slip = self.side_slip_rad

                v = np.sqrt((s*lb - slip*(lf+lb))/(slip*EG - s*SG))
                R = ((lf+lb) + EG*v**2)/s

                states_to_solve_for = ["velocity", "radius"]
                x0 = [v, R]

            elif set(state_names) == set(["sideslip", "radius"]):
                slip = self.side_slip_rad
                R = self.path_radius_m

                s = (lf+lb)/R + EG/SG*(slip - lb/R)
                v = np.sqrt((R*slip - lb)/SG)

                states_to_solve_for = ["velocity", "steering"]
                x0 = [v, s]

            return states_to_solve_for, x0

        def objective_function(x):
            for state, value in zip(states_to_solve_for, x):
                setattr(self, self.state_name_dict[state], value)

            self.calc_wheel_angles()
            self.calc_forces()

            self.force_centripetal_N = \
                self.force_norm_fl + self.force_norm_fr + \
                    self.force_norm_bl + self.force_norm_br

            self.force_centrifugal_N = \
                self.car.mass_kg*self.velocity_mps**2/self.path_radius_m
            
            force_resultant = \
                self.force_centripetal_N + self.force_centrifugal_N
            
            self.calc_moments()
            
            return [force_resultant, self.moment_yaw_Nm]

        validated_state_names = parse_state_input(state_names, state_values_SI)

        states_to_solve_for, x0 = calc_init_guess(validated_state_names)
        
        sol = root(objective_function, x0 = x0)
        self.visualize_vehicle_state()

if __name__ == "__main__":
    car = Vehicle(cg_to_front_m = 2.0,
                  cg_to_back_m = 2.0,
                  toe_front_deg = 0.0,
                  toe_back_deg = 0.0,
                  track_front_m=1.8,
                  track_back_m=2)
    
    tire_front = Tire(cornering_stiffness_NperRad=20000)
    tire_rear = Tire()
    simulation = Simulation(car=car,
                            tire_front=tire_front,
                            tire_rear=tire_rear)
    
    simulation.find_stationary_point(state_names=["vel", "steer"],
                                     state_values_SI=[10/3.6, 35/57.3])