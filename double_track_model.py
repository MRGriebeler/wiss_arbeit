import numpy as np

from scipy.optimize import root
from scipy.integrate import solve_ivp

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from typing import Callable
from dataclasses import dataclass
import functools

@dataclass
class Vehicle:
    track_back_m: float = 1.5
    track_front_m: float = 1.5
    cg_to_front_m: float = 2.0
    cg_to_back_m: float = 2.0
    mass_kg: float = 1000
    yaw_inertia_multiplier: float = 1.0
    yaw_inertia_kgm2: float = np.nan
    steer_ratio_before_over_after_rack: float = 1.0
    toe_out_front_deg: float = 0.0
    toe_out_back_deg:  float = 0.0
    ackermann_factor: float = 1.0

    def wheelbase_m(self) -> float:
        return self.cg_to_front_m + self.cg_to_back_m
    
    def yaw_inertia_estimation_kgm2(self) -> float:
        '''
        Yaw inertia is calculated based on the moment of inertia of a rectangle
        with the previously informed dimensions
        '''
        track_mean_m = (self.track_front_m + self.track_back_m)/2
        rectangle_inertia_kgm2 = \
            1/12*self.mass_kg*(track_mean_m**2 + self.wheelbase_m**2)

        return rectangle_inertia_kgm2*self.yaw_inertia_multiplier

@dataclass
class Tire:
    cornering_stiffness_NperRad: float = 30000.0
    relaxation_length_m: float = 5.0

@dataclass
class WheelAngles:
    steer_fl: float
    steer_fr: float
    alpha_fl: float
    alpha_fr: float
    alpha_bl: float
    alpha_br: float

def calc_wheel_angles(
        car: Vehicle,
        *,
        steering_wheel_input_rad: float,
        side_slip_rad: float,
        dyaw_dt_rad_s: float,
        velocity_m_s: float
        ) -> WheelAngles:
    '''
    Calculate the following quantities:
    - Steering angle from the front left wheel [rad]
    - Steering angle from the front right wheel [rad]
    - Tire slip angle from the front left wheel [rad]
    - Tire slip angle from the front right wheel [rad]
    - Tire slip angle from the back left wheel [rad]
    - Tire slip angle from the back right wheel [rad]

    The output is provided as a Wheel_Angles object
    '''
    #region - Calculate steering angle from each individual wheel    
    
    # Shorthand for the necessary input
    steer = \
        steering_wheel_input_rad/car.steer_ratio_before_over_after_rack
    ack = car.ackermann_factor
    wf = car.track_front_m
    lf = car.cg_to_front_m
    lb = car.cg_to_back_m
    toe_f = np.deg2rad(car.toe_out_front_deg)

    # Calculation
    steer_fl_rad = \
        np.atan2(np.tan(steer), 1 - ack*wf/2*np.tan(steer)/(lf+lb)) \
            + toe_f*np.sign(steer)

    steer_fr_rad = \
        np.atan2(np.tan(steer), 1 + ack*wf/2*np.tan(steer)/(lf+lb)) \
            - toe_f*np.sign(steer)
    
    #endregion

    #region - Calculate tire slip angles at the front axle
    
    # Shorthand for the additional necessary input
    beta = side_slip_rad
    vel = velocity_m_s
    dyaw_dt = dyaw_dt_rad_s
    
    # Calculation
    alpha_fl_rad = \
        np.atan2(np.sin(beta) + dyaw_dt*lf/vel, np.cos(beta) - 
            (dyaw_dt/vel)*wf/2) - steer_fl_rad + toe_f*np.sign(steer)
        
    alpha_fr_rad = \
        np.atan2(np.sin(beta) + dyaw_dt*lf/vel, np.cos(beta) + 
            (dyaw_dt/vel)*wf/2) - steer_fr_rad - toe_f*np.sign(steer)

    #endregion
    
    #region - Calculate tire slip angles at the rear axle

    # Shorthand for the additional necessary input
    wb = car.track_back_m
    toe_b = np.deg2rad(car.toe_out_back_deg)

    # Calculation
    alpha_bl_rad = \
        np.atan2(np.sin(beta) - dyaw_dt*lb/vel, np.cos(beta) - 
            (dyaw_dt/vel)*wb/2) + toe_b*np.sign(steer)
        
    alpha_br_rad = \
        np.atan2(np.sin(beta) - dyaw_dt*lb/vel, np.cos(beta) + 
            (dyaw_dt/vel)*wb/2) - toe_b*np.sign(steer)
    
    #endregion

    #region - Define output and return statement
    output_rad = WheelAngles(steer_fl_rad,
                              steer_fr_rad,
                              alpha_fl_rad,
                              alpha_fr_rad,
                              alpha_bl_rad,
                              alpha_br_rad)
    
    return output_rad

    #endregion

@dataclass
class WheelForces:
    Fx_fl: float
    Fy_fl: float
    Fx_fr: float
    Fy_fr: float
    Fx_bl: float
    Fy_bl: float
    Fx_br: float
    Fy_br: float
    Fn_fl: float
    Ft_fl: float
    Fn_fr: float
    Ft_fr: float
    Fn_bl: float
    Ft_bl: float
    Fn_br: float
    Ft_br: float

def calc_wheel_forces(
        car: Vehicle,
        *,
        tire_front: Tire,
        tire_rear: Tire,
        wheel_angles: WheelAngles,
        side_slip_rad: float
        ) -> WheelForces:
    ''' 
    Calculates X and Y compoenents of the force on each tire with respect to
    the vehicle reference frame. Also calculates the force component in the
    normal direction to the CG velocity vector
    '''
    #region - Calculate front axle wheel forces on vehicle reference frame

    # Shorthand for the necessary input
    c_f = tire_front.cornering_stiffness_NperRad
    toe_f = car.toe_out_front_deg
    d_fl = wheel_angles.steer_fl
    d_fr = wheel_angles.steer_fr
    a_fl = wheel_angles.alpha_fl
    a_fr = wheel_angles.alpha_fr

    # Calculation
    Fx_fl_N = +c_f*(a_fl*np.sin(d_fl + toe_f))
    Fy_fl_N = -c_f*(a_fl*np.cos(d_fl + toe_f))

    Fx_fr_N = +c_f*(a_fr*np.sin(d_fl - toe_f))
    Fy_fr_N = -c_f*(a_fr*np.cos(d_fl - toe_f))

    #endregion

    #region - Calculate back axle wheel forces on vehicle reference frame

    # Shorthand for the additional necessary input
    c_b = tire_rear.cornering_stiffness_NperRad
    toe_b = car.toe_out_back_deg
    a_bl = wheel_angles.alpha_bl
    a_br = wheel_angles.alpha_br

    # Calculation
    Fx_bl_N = +c_b*a_bl*np.sin(+toe_b)
    Fy_bl_N = -c_b*a_bl*np.cos(+toe_b)

    Fx_br_N = +c_b*a_br*np.sin(-toe_b)
    Fy_br_N = -c_b*a_br*np.cos(-toe_b)

    #endregion
    
    #region - Calculate the wheel force component normal and tangent to the CG 

    Fn_fl_N = c_f*a_fl*np.cos(side_slip_rad - d_fl)
    Fn_fr_N = c_f*a_fr*np.cos(side_slip_rad - d_fr)
    Fn_bl_N = c_b*a_bl*np.cos(side_slip_rad)
    Fn_br_N = c_b*a_br*np.cos(side_slip_rad)
    
    Ft_fl_N = c_f*a_fl*(-np.sin(side_slip_rad - d_fl))
    Ft_fr_N = c_f*a_fr*(-np.sin(side_slip_rad - d_fr))
    Ft_bl_N = c_b*a_bl*(-np.sin(side_slip_rad))
    Ft_br_N = c_b*a_br*(-np.sin(side_slip_rad))

    #endregion

    #region - Define output and return statement
    output_N = WheelForces(Fx_fl_N, Fy_fl_N,
                            Fx_fr_N, Fy_fr_N,
                            Fx_bl_N, Fy_bl_N,
                            Fx_br_N, Fy_br_N,
                            Fn_fl_N, Ft_fl_N,
                            Fn_fr_N, Ft_fr_N,
                            Fn_bl_N, Ft_bl_N,
                            Fn_br_N, Ft_br_N)
    
    return output_N

    #endregion

@dataclass
class WheelMoments:
    Myaw_fl: float
    Myaw_fr: float
    Myaw_bl: float
    Myaw_br: float

def calc_wheel_moments(
        car: Vehicle,
        wheel_forces: WheelForces
        ):
    '''
    Calculates the yaw moment generated by each individual wheel
    '''
    #region - Calculate yaw moment generated by each individual wheel

    # Shorthand for the necessary input
    l_f = car.cg_to_front_m
    l_b = car.cg_to_back_m
    w_f = car.track_front_m
    w_b = car.track_back_m

    # Calculation
    Myaw_fl_Nm = + l_f*wheel_forces.Fy_fl - w_f/2*wheel_forces.Fx_fl
    Myaw_fr_Nm = + l_f*wheel_forces.Fy_fr + w_f/2*wheel_forces.Fx_fr
    Myaw_bl_Nm = - l_b*wheel_forces.Fy_bl - w_b/2*wheel_forces.Fx_bl
    Myaw_br_Nm = - l_b*wheel_forces.Fy_br + w_b/2*wheel_forces.Fx_br

    #endregion
    
    #region - Define output and return statement

    output_Nm = WheelMoments(Myaw_fl_Nm,
                              Myaw_fr_Nm,
                              Myaw_bl_Nm,
                              Myaw_br_Nm)
    
    return output_Nm

    #endregion

VALID_SSC_INPUT_NAMES = {"radius_of_turn_m",
                         "side_slip_rad",
                         "steering_wheel_input_rad",
                         "velocity_m_s"}

def validate_ssc_input(function):
    '''
    Custom decorator that validates the input given to the
    steady_state_cornering() function. Could also be implemented directly in
    steady_state_cornering(), but has been separated into a decorator to allow
    steady_state_cornering() to have only the physical calculation
    '''
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        input = kwargs.pop("input", None)

        if input is None:
            raise ValueError("The 'input' keyword argument is mandatory")

        if not isinstance(input, dict):
            raise TypeError("Input must be a dictionary")
        if len(input) != 2:
            raise ValueError("Dictionary must contain exactly two itens")
        
        validated_input = dict()

        for key, value in input.items():
            matches = [valid_name for valid_name in VALID_SSC_INPUT_NAMES \
                       if valid_name.startswith(key.lower())]
            
            if not matches:
                raise ValueError(f"'{key}' does not match any of the valid \
                                input ({VALID_SSC_INPUT_NAMES})")
            if len(matches) > 1:
                raise ValueError(f"'{key}' is ambiguous. \
                                 Matches found: {matches}")

            if matches[0] in validated_input:
                raise ValueError(f"Duplicated input '{matches[0]}' detected")
            
            validated_input[matches[0]] = value

            kwargs["input"] = validated_input

        return function(*args, **kwargs)
    return wrapper

@dataclass
class MotionState:
    radius_of_turn: float
    side_slip: float
    steering_wheel_input: float
    velocity: float
    dyaw_dt: float
    d2yaw_dt2: float
    dside_slip_dt: float

@dataclass
class ResultSet:
    car: Vehicle
    tire_front: Tire
    tire_rear: Tire
    motion_state: MotionState

@validate_ssc_input
def ssc_single_track(
        car: Vehicle,
        *,
        tire_front: Tire,
        tire_rear: Tire,
        input: dict
        ) -> ResultSet:
    '''
    Outputs velocity, turning radius, side slip angle and steering angle
    for a given vehicle and tires using the equations for a linear single
    track model. Two of the four parameters need to be passed as input.
    The remaining two parameters are calculated.
    '''
    # Shorthand for the necessary parameters
    m = car.mass_kg
    lf = car.cg_to_front_m
    lb = car.cg_to_back_m
    mf = m*lb/(lf+lb)
    mb = m*lf/(lf+lb)
    cf_twin = 2*tire_front.cornering_stiffness_NperRad
    cb_twin = 2*tire_rear.cornering_stiffness_NperRad
    steer_ratio = car.steer_ratio_before_over_after_rack

    # Definition of Eigenlenkgradient and Schwimmwinkelgradient
    EG = + mf/cf_twin - mb/cb_twin
    SG = - mb/cb_twin

    # Calculation of vehicle parameters depending on input
    sorted_keys = tuple(sorted(input.keys()))
    match sorted_keys:
        case ("radius_of_turn_m", "side_slip_rad"):
            R = input["radius_of_turn_m"]
            slip = input["side_slip_rad"]
            s = (lf+lb)/R + EG/SG*(slip - lb/R)
            v = np.sqrt((R*slip - lb)/SG)

        case ("radius_of_turn_m", "steering_wheel_input_rad"):
            R = input["radius_of_turn_m"]
            s = input["steering_wheel_input_rad"]/steer_ratio
            v = np.sqrt((R*s - (lf+lb))/EG)
            slip = lb/R + SG/EG*(s - (lf+lb)/R)

        case ("radius_of_turn_m", "velocity_m_s"):
            R = input["radius_of_turn_m"]
            v = input["velocity_m_s"]
            s = ((lf+lb) + EG*v**2)/R
            slip = (lb + SG*v**2)/R

        case ("side_slip_rad", "steering_wheel_input_rad"):
            slip = input["side_slip_rad"]
            s = input["steering_wheel_input_rad"]/steer_ratio
            v = np.sqrt((s*lb - slip*(lf+lb))/(slip*EG - s*SG))
            R = ((lf+lb) + EG*v**2)/s

        case ("side_slip_rad", "velocity_m_s"):
            slip = input["side_slip_rad"]
            v = input["velocity_m_s"]
            R = (lb + SG*v**2)/slip
            s = slip*((lf+lb) + EG*v**2)/(lb + SG*v**2)

        case ("steering_wheel_input_rad", "velocity_m_s"):
            s = input["steering_wheel_input_rad"]/steer_ratio
            v = input["velocity_m_s"]
            R = ((lf+lb) + EG*v**2)/s
            slip = s*((lb + SG*v**2)/((lf+lb) + EG*v**2))

    # Definition of output and return statement
    motion_state = MotionState(R, slip, s*steer_ratio, v, v/R, 0, 0)
    result_set = ResultSet(car, tire_front, tire_rear, motion_state)
    
    return result_set

@validate_ssc_input
def steady_state_cornering(
        car: Vehicle,
        *,
        tire_front: Tire,
        tire_rear: Tire,
        input: dict
        ):
    '''
    Outputs velocity, turning radius, side slip angle and steering angle
    for a given vehicle and tires using the equations for a non-linear
    (no small angles assumption) double track model.
    Two of the four parameters need to be passed as input.
    The remaining two parameters are calculated.
    '''
    single_track_result = ssc_single_track(car,
                                     tire_front=tire_front,
                                     tire_rear=tire_rear,
                                     input=input)
    
    def motion_equations(car: Vehicle,
                         *,
                         tire_front: Tire,
                         tire_rear: Tire,
                         steering_wheel_input_rad: float,
                         side_slip_rad: float,
                         velocity_m_s: float,
                         radius_of_turn_m: float):
        
        wheel_angles_rad = calc_wheel_angles(
            car,
            steering_wheel_input_rad=steering_wheel_input_rad,
            side_slip_rad=side_slip_rad,
            dyaw_dt_rad_s=velocity_m_s/radius_of_turn_m,
            velocity_m_s=velocity_m_s)
        
        wheel_forces_N = calc_wheel_forces(
            car,
            tire_front=tire_front,
            tire_rear=tire_rear,
            wheel_angles=wheel_angles_rad,
            side_slip_rad=side_slip_rad)
        
        force_centripetal_N = wheel_forces_N.Fn_fl + wheel_forces_N.Fn_fr \
            + wheel_forces_N.Fn_bl + wheel_forces_N.Fn_br
        
        force_centrifugal_N = car.mass_kg*velocity_m_s**2/radius_of_turn_m

        force_resultant_N = force_centripetal_N + force_centrifugal_N
        
        wheel_moments_Nm = calc_wheel_moments(car, wheel_forces=wheel_forces_N)
        
        yaw_moment_Nm = wheel_moments_Nm.Myaw_fl + wheel_moments_Nm.Myaw_fr \
            + wheel_moments_Nm.Myaw_bl + wheel_moments_Nm.Myaw_br
        
        return (force_resultant_N, yaw_moment_Nm)
    
    #region - Definition of parameters to solve for based on user input

    sorted_keys = tuple(sorted(input.keys()))
    match sorted_keys:
        case ("radius_of_turn_m", "side_slip_rad"):
            objective_function = \
                lambda x: motion_equations(
                    car = car,
                    tire_front=tire_front,
                    tire_rear=tire_rear,
                    radius_of_turn_m=input["radius_of_turn_m"],
                    side_slip_rad=input["side_slip_rad"],
                    steering_wheel_input_rad=x[0],
                    velocity_m_s=x[1])
            
            x0 = (single_track_result.steering_wheel_input,
                  single_track_result.velocity)

        case ("radius_of_turn_m", "steering_wheel_input_rad"):
            objective_function = \
                lambda x: motion_equations(
                    car = car,
                    tire_front=tire_front,
                    tire_rear=tire_rear,
                    radius_of_turn_m=input["radius_of_turn_m"],
                    side_slip_rad=x[0],
                    steering_wheel_input_rad=input["steering_wheel_input_rad"],
                    velocity_m_s=x[1])

            x0 = (single_track_result.side_slip,
                  single_track_result.velocity)

        case ("radius_of_turn_m", "velocity_m_s"):
            objective_function = \
                lambda x: motion_equations(
                    car = car,
                    tire_front=tire_front,
                    tire_rear=tire_rear,
                    radius_of_turn_m=input["radius_of_turn_m"],
                    side_slip_rad=x[0],
                    steering_wheel_input_rad=x[1],
                    velocity_m_s=input["velocity_m_s"])

            x0 = (single_track_result.motion_state.side_slip,
                  single_track_result.motion_state.steering_wheel_input)

        case ("side_slip_rad", "steering_wheel_input_rad"):
            objective_function = \
                lambda x: motion_equations(
                    car = car,
                    tire_front=tire_front,
                    tire_rear=tire_rear,
                    radius_of_turn_m=x[0],
                    side_slip_rad=input["side_slip_rad"],
                    steering_wheel_input_rad=input["steering_wheel_input_rad"],
                    velocity_m_s=x[1])

            x0 = (single_track_result.motion_state.radius_of_turn,
                  single_track_result.motion_state.velocity)

        case ("side_slip_rad", "velocity_m_s"):
            objective_function = \
                lambda x: motion_equations(
                    car = car,
                    tire_front=tire_front,
                    tire_rear=tire_rear,
                    radius_of_turn_m=x[0],
                    side_slip_rad=input["side_slip_rad"],
                    steering_wheel_input_rad=x[1],
                    velocity_m_s=input["velocity_m_s"])

            x0 = (single_track_result.motion_state.radius_of_turn,
                  single_track_result.motion_state.steering_wheel_input)

        case ("steering_wheel_input_rad", "velocity_m_s"):
            objective_function = \
                lambda x: motion_equations(
                    car = car,
                    tire_front=tire_front,
                    tire_rear=tire_rear,
                    radius_of_turn_m=x[0],
                    side_slip_rad=x[1],
                    steering_wheel_input_rad=input["steering_wheel_input_rad"],
                    velocity_m_s=input["velocity_m_s"])

            x0 = (single_track_result.motion_state.radius_of_turn,
                  single_track_result.motion_state.side_slip)
            
    #endregion

    solution = root(objective_function, x0=x0)
    
    #region - Definition of motion_state based on user informed input

    match sorted_keys:
        case ("radius_of_turn_m", "side_slip_rad"):
            motion_state = MotionState(
                input["radius_of_turn_m"],
                input["side_slip_rad"],
                solution.x[0],
                solution.x[1],
                np.nan,
                np.nan,
                np.nan)

        case ("radius_of_turn_m", "steering_wheel_input_rad"):
            motion_state = MotionState(
                input["radius_of_turn_m"],
                solution.x[0],
                input["steering_wheel_input_rad"],
                solution.x[1],
                np.nan,
                np.nan,
                np.nan)

        case ("radius_of_turn_m", "velocity_m_s"):
            motion_state = MotionState(
                input["radius_of_turn_m"],
                solution.x[0],
                solution.x[1],
                input["velocity_m_s"],
                np.nan,
                np.nan,
                np.nan)
            
        case ("side_slip_rad", "steering_wheel_input_rad"):
            motion_state = MotionState(
                solution.x[0],
                input["side_slip_rad"],
                input["steering_wheel_input_rad"],
                solution.x[1],
                np.nan,
                np.nan,
                np.nan)

        case ("side_slip_rad", "velocity_m_s"):
            motion_state = MotionState(
                solution.x[0],
                input["side_slip_rad"],
                solution.x[1],
                input["velocity_m_s"],
                np.nan,
                np.nan,
                np.nan)

        case ("steering_wheel_input_rad", "velocity_m_s"):
            motion_state = MotionState(
                solution.x[0],
                solution.x[1],
                input["steering_wheel_input_rad"],
                input["velocity_m_s"],
                np.nan,
                np.nan,
                np.nan)

    #endregion

    #region - Definition of result_set to be returned
    
    # Motion state parameters given steady state condition
    motion_state.dyaw_dt = motion_state.velocity/motion_state.radius_of_turn
    motion_state.d2yaw_dt2 = 0
    motion_state.dside_slip_dt = 0

    result_set = ResultSet(car, tire_front, tire_rear, motion_state)

    #endregion

    return result_set, solution

def draw_moment(ax: Axes, pos_x: float, pos_y: float, moment: float,
                scale_factor: float = 5e-5, color = 'red', linewidth = 1,
                markersize=5):
    
    radius = abs(moment)*scale_factor
    theta = np.linspace(np.radians(-45), np.radians(225), 100)

    x = pos_x + radius * np.cos(theta)
    y = pos_y + radius * np.sin(theta)
    
    ax.plot(x, y, color=color, linewidth=linewidth)

    if moment >= 0:
        x_tip = x[-1]
        y_tip = y[-1]
        dx_tip = x[-2] - x[-1]
        dy_tip = y[-2] - y[-1]
    elif moment < 0:
        x_tip = x[0]
        y_tip = y[0]
        dx_tip = x[1] - x[0]
        dy_tip = y[1] - y[0]

    tip_angle_deg = np.degrees(np.arctan2(dy_tip, dx_tip))
    num_sides = 3
    rotated_triangle = (num_sides, 0, tip_angle_deg + 90)

    ax.plot(x_tip, y_tip, marker=rotated_triangle, markersize=markersize,
            color=color)

def visualize_vehicle(
        ax: Axes,
        result_set: ResultSet
        ):

    #region - Parameters used in multiple following sections

    rot_x_pts = lambda x,y,ang: x*np.cos(ang) - y*np.sin(ang)
    rot_y_pts = lambda x,y,ang: x*np.sin(ang) + y*np.cos(ang)
    
    lf = result_set.car.cg_to_front_m
    lb = result_set.car.cg_to_back_m
    wf = result_set.car.track_front_m
    wb = result_set.car.track_back_m

    wheel_pos_x_m = np.array([+lf,+lf,-lb,-lb])
    wheel_pos_y_m = np.array([+wf/2,-wf/2,+wb/2,-wb/2])
    
    #endregion

    #region - Vehicle body visualization

    body_lines_x = np.array([+lf,+lf,+lf,-lb,-lb,-lb])
    body_lines_y = np.array([+wf/2,-wf/2,0,0,+wb/2,-wb/2])

    rot_global = np.pi/2

    body_lines_x_glob_ref = \
                rot_x_pts(body_lines_x,
                          body_lines_y,
                          rot_global)
    
    body_lines_y_glob_ref = \
                rot_y_pts(body_lines_x,
                          body_lines_y,
                          rot_global)
    
    inputs = (body_lines_x_glob_ref, body_lines_y_glob_ref)
    ax.plot(*inputs, color='silver')
    
    #endregion

    #region - Vehicle wheels visualization
    
    wheel_lines_x_m = np.array([+0.5, +0.5, -0.5, -0.5])
    wheel_lines_y_m = np.array([+0.2, -0.2, -0.2, +0.2])

    motion_state = result_set.motion_state

    wheel_angles = calc_wheel_angles(
        result_set.car,
        steering_wheel_input_rad=motion_state.steering_wheel_input,
        side_slip_rad=motion_state.side_slip,
        dyaw_dt_rad_s=motion_state.dyaw_dt,
        velocity_m_s=motion_state.velocity)

    # Shorthand for necessary parameters
    st_fl = wheel_angles.steer_fl
    st_fr = wheel_angles.steer_fr
    toe_f = np.deg2rad(car.toe_out_front_deg)
    toe_b = np.deg2rad(car.toe_out_back_deg)

    wheel_angle_rad = [
        st_fl - np.sign(st_fl)*toe_f,
        st_fr + np.sign(st_fr)*toe_f,
        +np.sign(st_fl)*toe_b,
        -np.sign(st_fr)*toe_b]
    
    wheel_lines_glob_ref = list()
    for x_shift, y_shift, wheel_ang in zip(wheel_pos_x_m,
                                           wheel_pos_y_m,
                                           wheel_angle_rad):
        
        wheel_x_car_ref = x_shift + \
            rot_x_pts(wheel_lines_x_m, wheel_lines_y_m, wheel_ang)
        
        wheel_y_car_ref = y_shift + \
            rot_y_pts(wheel_lines_x_m, wheel_lines_y_m, wheel_ang)
        
        wheel_x_glob_ref = \
            rot_x_pts(wheel_x_car_ref, wheel_y_car_ref, rot_global)
        
        wheel_y_glob_ref = \
            rot_y_pts(wheel_x_car_ref, wheel_y_car_ref, rot_global)

        inputs = (np.append(wheel_x_glob_ref, wheel_x_glob_ref[0]), 
                  np.append(wheel_y_glob_ref, wheel_y_glob_ref[0]))
        
        wheel_lines_glob_ref.append(ax.plot(*inputs, color='black'))

    #endregion

    #region - Velocity of CG visualization

    vel_cg_x = motion_state.velocity*np.cos(result_set.motion_state.side_slip)
    vel_cg_y = motion_state.velocity*np.sin(result_set.motion_state.side_slip)
    
    vel_cg_x_glob_ref = rot_x_pts(vel_cg_x, vel_cg_y, rot_global)
    vel_cg_y_glob_ref = rot_y_pts(vel_cg_x, vel_cg_y, rot_global)
    
    vel_cg_graphic = ax.quiver(0, 0, vel_cg_x_glob_ref, vel_cg_y_glob_ref,
                               scale=10, scale_units='x', width=0.001,
                               headlength=3, headaxislength=3, color='black')    
    
    #endregion

    #region - Velocity of individual wheels visualization
    
    vel_wheel_graphics = list()
    for wheel_x, wheel_y in zip(wheel_pos_x_m, wheel_pos_y_m):
        
        # The yaw velocity component is the cross product between yaw rate
        # vector and wheel position with respect to the vehicle CG:
        vel_yaw_x = -motion_state.dyaw_dt*wheel_y
        vel_yaw_y = +motion_state.dyaw_dt*wheel_x

        vel_wheel_x = vel_cg_x + vel_yaw_x
        vel_wheel_y = vel_cg_y + vel_yaw_y
        
        vel_wheel_x_glob_ref = \
            rot_x_pts(vel_wheel_x, vel_wheel_y, rot_global)
        vel_wheel_y_glob_ref = \
            rot_y_pts(vel_wheel_x, vel_wheel_y, rot_global)
        
        wheel_x_glob_ref = \
            rot_x_pts(wheel_x, wheel_y, rot_global)
        wheel_y_glob_ref = \
            rot_y_pts(wheel_x, wheel_y, rot_global)
        
        vel_wheel_graphics.append(
            ax.quiver(wheel_x_glob_ref, wheel_y_glob_ref,
                      vel_wheel_x_glob_ref, vel_wheel_y_glob_ref,
                      scale=10, scale_units='x', width=0.001, headlength=3, 
                      headaxislength=3, color='black'))
        
    #endregion

    #region - Path radius of CG visualization
    
    unit_normal_to_vel_cg_x = -vel_cg_y/motion_state.velocity
    unit_normal_to_vel_cg_y = +vel_cg_x/motion_state.velocity
    
    path_radius_cg_x = motion_state.radius_of_turn*unit_normal_to_vel_cg_x
    path_radius_cg_y = motion_state.radius_of_turn*unit_normal_to_vel_cg_y

    path_radius_cg_glob_ref_x = \
        rot_x_pts(path_radius_cg_x, path_radius_cg_y, rot_global)
    path_radius_cg_glob_ref_y = \
        rot_y_pts(path_radius_cg_x, path_radius_cg_y, rot_global)

    path_radius_cg_graphic = \
        ax.plot([0, path_radius_cg_glob_ref_x],
                [0, path_radius_cg_glob_ref_y],
                color='grey', marker='x', linestyle='dashed', linewidth=1)
    
    #endregion

    #region - Path radii of individual wheels
    
    path_radius_wheel_graphics = list()
    for wheel_x, wheel_y in zip(wheel_pos_x_m, wheel_pos_y_m):
        
        wheel_glob_ref_x = \
            rot_x_pts(wheel_x, wheel_y, rot_global)
        wheel_glob_ref_y = \
            rot_y_pts(wheel_x, wheel_y, rot_global)
        
        path_radius_wheel_graphics.append(
            ax.plot([wheel_glob_ref_x, path_radius_cg_glob_ref_x],
                    [wheel_glob_ref_y, path_radius_cg_glob_ref_y],
                    color='grey', linestyle='dashed', linewidth=1))
        
    #endregion

    #region - Force of individual wheels visualization

    wheel_forces = calc_wheel_forces(
        car=result_set.car,
        tire_front=result_set.tire_front,
        tire_rear=result_set.tire_rear,
        wheel_angles=wheel_angles,
        side_slip_rad=result_set.motion_state.side_slip)

    wheel_forcesX = [wheel_forces.Fx_fl, wheel_forces.Fx_fr,
                     wheel_forces.Fx_bl, wheel_forces.Fx_br]

    wheel_forcesY = [wheel_forces.Fy_fl, wheel_forces.Fy_fr,
                     wheel_forces.Fy_bl, wheel_forces.Fy_br]
    
    force_graphics = list()
    for forceX, forceY, pos_x, pos_y in zip(wheel_forcesX,
                                                    wheel_forcesY,
                                                    wheel_pos_x_m,
                                                    wheel_pos_y_m):

        pos_x_glob_ref = rot_x_pts(pos_x, pos_y, rot_global)
        pos_y_glob_ref = rot_y_pts(pos_x, pos_y, rot_global)
        
        forceX_glob_ref = rot_x_pts(forceX, forceY, rot_global)
        forceY_glob_ref = rot_y_pts(forceX, forceY, rot_global)

        force_graphics.append(
            ax.quiver(pos_x_glob_ref, pos_y_glob_ref,
                      forceX_glob_ref, forceY_glob_ref,
                      scale=1000, scale_units='x', width=0.001, headlength=3,
                      headaxislength=3, color='red'))
        
    #endregion

    #region - Moment of individual wheels visualization
    
    wheel_moments = calc_wheel_moments(car=result_set.car, 
                                       wheel_forces=wheel_forces)

    wheel_moment_list = [wheel_moments.Myaw_fl, wheel_moments.Myaw_fr,
                          wheel_moments.Myaw_bl, wheel_moments.Myaw_br]
    
    moment_graphics = list()
    for moment, pos_x, pos_y in zip(wheel_moment_list, 
                                            wheel_pos_x_m,
                                            wheel_pos_y_m):

        pos_x_glob_ref = rot_x_pts(pos_x, pos_y, rot_global)
        pos_y_glob_ref = rot_y_pts(pos_x, pos_y, rot_global)

        moment_graphics.append(
            draw_moment(ax, pos_x=pos_x_glob_ref, pos_y=pos_y_glob_ref,
                        moment=moment, scale_factor=5e-5, color='red',
                        linewidth=1, markersize=5))

    #endregion

    #region - Resultant force visualization

    force_res_x = (wheel_forces.Fx_fl + wheel_forces.Fx_fr + 
                   wheel_forces.Fx_bl + wheel_forces.Fx_br)
    
    force_res_y = (wheel_forces.Fy_fl + wheel_forces.Fy_fr + 
                   wheel_forces.Fy_bl + wheel_forces.Fy_br)
    
    force_res_x_glob_ref = rot_x_pts(force_res_x, force_res_y, rot_global)
    force_res_y_glob_ref = rot_y_pts(force_res_x, force_res_y, rot_global)

    force_res_graphics = (
        ax.quiver(0, 0, force_res_x_glob_ref, force_res_y_glob_ref,
                  scale=1000, scale_units='x', width=0.001, headlength=3,
                  headaxislength=3, color='royalblue'))
    
    force_res_norm = (wheel_forces.Fn_fl + wheel_forces.Fn_fr +
                      wheel_forces.Fn_bl + wheel_forces.Fn_br)
    
    force_res_norm_x = +force_res_norm*np.sin(result_set.motion_state.side_slip)
    force_res_norm_y = -force_res_norm*np.cos(result_set.motion_state.side_slip)
    
    force_res_norm_x_glob_ref = (
        rot_x_pts(force_res_norm_x, force_res_norm_y, rot_global))
    
    force_res_norm_y_glob_ref = (
        rot_y_pts(force_res_norm_x, force_res_norm_y, rot_global))
    
    force_res_norm_graphics = (
        ax.quiver(0, 0, force_res_norm_x_glob_ref, force_res_norm_y_glob_ref,
                  scale=1000, scale_units='x', width=0.001, headlength=3,
                  headaxislength=3, color='turquoise'))

    force_res_tang = (wheel_forces.Ft_fl + wheel_forces.Ft_fr +
                      wheel_forces.Ft_bl + wheel_forces.Ft_br)

    force_res_tang_x = force_res_tang*np.cos(result_set.motion_state.side_slip)
    force_res_tang_y = force_res_tang*np.sin(result_set.motion_state.side_slip)

    force_res_tang_x_glob_ref = (
        rot_x_pts(force_res_tang_x, force_res_tang_y, rot_global))
    
    force_res_tang_y_glob_ref = (
        rot_y_pts(force_res_tang_x, force_res_tang_y, rot_global))
    
    force_res_norm_graphics = (
        ax.quiver(0, 0, force_res_tang_x_glob_ref, force_res_tang_y_glob_ref,
                  scale=1000, scale_units='x', width=0.001, headlength=3,
                  headaxislength=3, color='turquoise'))

    #endregion

    #region - Resultant moment visualization

    moment_res = (wheel_moments.Myaw_fl + wheel_moments.Myaw_fr +
                  wheel_moments.Myaw_bl + wheel_moments.Myaw_br)

    draw_moment(ax, pos_x=0, pos_y=0, moment=moment_res, scale_factor=5e-5,
                color='royalblue', linewidth=1, markersize=5)

    #endregion

if __name__ == "__main__":
    car = Vehicle(cg_to_front_m = 2.0,
                  cg_to_back_m = 2.0,
                  toe_out_front_deg = 0.0,
                  toe_out_back_deg = 0.0,
                  track_front_m=1.8,
                  track_back_m=2,
                  ackermann_factor=1,
                  steer_ratio_before_over_after_rack= 13)
    
    tire_front = Tire(cornering_stiffness_NperRad=20000)
    tire_rear = Tire()

    input = {"radius": 12, "velocity": 20/3.6}
    result_set_nonlin,_ = steady_state_cornering(car,
                           tire_front=tire_front,
                           tire_rear=tire_rear,
                           input=input)
    
    result_set_linear = ssc_single_track(car,
                           tire_front=tire_front,
                           tire_rear=tire_rear,
                           input=input)
    
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    visualize_vehicle(ax, result_set_nonlin)
    fig.show()
    print()