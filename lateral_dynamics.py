import numpy as np
import numpy.typing as npt

from scipy.optimize import root

import matplotlib.pyplot as plt
from matplotlib.artist import Artist
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.quiver import Quiver
from matplotlib.animation import FuncAnimation

from dataclasses import dataclass, fields
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
        wheelbase_m = self.cg_to_front_m + self.cg_to_back_m
        track_mean_m = (self.track_front_m + self.track_back_m)/2
        rectangle_inertia_kgm2 = \
            1/12*self.mass_kg*(track_mean_m**2 + wheelbase_m**2)

        return rectangle_inertia_kgm2*self.yaw_inertia_multiplier

@dataclass
class Tire:
    cornering_stiffness_NperRad: float = 30000.0
    relaxation_length_m: float = 5.0

@dataclass
class WheelAngles:
    steer: npt.ArrayLike
    steer_fl: npt.ArrayLike
    steer_fr: npt.ArrayLike
    alpha_f: npt.ArrayLike
    alpha_fl: npt.ArrayLike
    alpha_fr: npt.ArrayLike
    alpha_b: npt.ArrayLike
    alpha_bl: npt.ArrayLike
    alpha_br: npt.ArrayLike

    def __getitem__(self, index):
        sliced_data = {}
        for field in fields(self):
            field_name = field.name
            field_value = getattr(self, field_name)

            sliced_data[field_name] = field_value[index]

        return self.__class__(**sliced_data)

def calc_wheel_angles(
        car: Vehicle,
        *,
        steering_wheel_input_rad: float,
        side_slip_rad: float,
        dyaw_dt_rad_s: float,
        velocity_m_s: float,
        single_track_on: bool = False,
        small_angles_on: bool = False
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
    #region - Calculate steering angle at each individual wheel    
    
    steer = \
        steering_wheel_input_rad/car.steer_ratio_before_over_after_rack
    ack = car.ackermann_factor
    wf = car.track_front_m
    lf = car.cg_to_front_m
    lb = car.cg_to_back_m
    toe_f = np.deg2rad(car.toe_out_front_deg)

    steer_fl_rad = np.nan
    steer_fr_rad = np.nan
    
    if not single_track_on:
        if small_angles_on:
            steer_fl_rad = \
                steer/(1 - ack*(wf/2)*steer/(lf+lb)) + toe_f*np.sign(steer)

            steer_fr_rad = \
                steer/(1 + ack*(wf/2)*steer/(lf+lb)) - toe_f*np.sign(steer)
        
        elif not small_angles_on:
            steer_fl_rad = \
                np.atan2(np.tan(steer), 1 - ack*(wf/2)*np.tan(steer)/(lf+lb)) \
                    + toe_f*np.sign(steer)

            steer_fr_rad = \
                np.atan2(np.tan(steer), 1 + ack*(wf/2)*np.tan(steer)/(lf+lb)) \
                    - toe_f*np.sign(steer)
    
    #endregion

    #region - Calculate tire slip angles
    
    wb = car.track_back_m
    toe_b = np.deg2rad(car.toe_out_back_deg)
    beta = side_slip_rad
    vel = velocity_m_s
    dyaw_dt = dyaw_dt_rad_s            
    
    alpha_f_rad = np.nan
    alpha_b_rad = np.nan
    alpha_fl_rad = np.nan
    alpha_fr_rad = np.nan
    alpha_bl_rad = np.nan
    alpha_br_rad = np.nan
    
    if small_angles_on:
        if single_track_on:
            alpha_f_rad = beta + dyaw_dt*lf/vel - steer
            alpha_b_rad = beta - dyaw_dt*lb/vel
        
        elif not single_track_on:
            alpha_fl_rad = ((beta + dyaw_dt*lf/vel)/(1 - (dyaw_dt/vel)*wf/2)
                            - steer_fl_rad + toe_f*np.sign(steer))
                
            alpha_fr_rad = ((beta + dyaw_dt*lf/vel)/(1 + (dyaw_dt/vel)*wf/2)
                            - steer_fr_rad - toe_f*np.sign(steer))
            
            alpha_bl_rad = ((beta - dyaw_dt*lb/vel)/(1 - (dyaw_dt/vel)*wb/2) 
                            + toe_b*np.sign(steer))
                
            alpha_br_rad = ((beta - dyaw_dt*lb/vel)/(1 + (dyaw_dt/vel)*wb/2)
                            - toe_b*np.sign(steer))
        
    elif not small_angles_on:
        if single_track_on:
            alpha_f_rad = np.atan2(np.sin(beta) + dyaw_dt*lf/vel, 
                                    np.cos(beta)) - steer
            
            alpha_b_rad = np.atan2(np.sin(beta) - dyaw_dt*lf/vel, 
                                    np.cos(beta))
        
        elif not single_track_on:
            alpha_fl_rad = \
                np.atan2(np.sin(beta) + dyaw_dt*lf/vel, np.cos(beta) - 
                    (dyaw_dt/vel)*wf/2) - steer_fl_rad + toe_f*np.sign(steer)
                
            alpha_fr_rad = \
                np.atan2(np.sin(beta) + dyaw_dt*lf/vel, np.cos(beta) + 
                    (dyaw_dt/vel)*wf/2) - steer_fr_rad - toe_f*np.sign(steer)
            
            alpha_bl_rad = \
                np.atan2(np.sin(beta) - dyaw_dt*lb/vel, np.cos(beta) - 
                    (dyaw_dt/vel)*wb/2) + toe_b*np.sign(steer)
                
            alpha_br_rad = \
                np.atan2(np.sin(beta) - dyaw_dt*lb/vel, np.cos(beta) + 
                    (dyaw_dt/vel)*wb/2) - toe_b*np.sign(steer)
    
    #endregion

    #region - Define output
    output_rad = WheelAngles(steer=steer,
                             steer_fl=steer_fl_rad,
                             steer_fr=steer_fr_rad,
                             alpha_f=alpha_f_rad,
                             alpha_fl=alpha_fl_rad,
                             alpha_fr=alpha_fr_rad,
                             alpha_b=alpha_b_rad,
                             alpha_bl=alpha_bl_rad,
                             alpha_br=alpha_br_rad)
    
    #endregion

    return output_rad

@dataclass
class WheelForces:
    Fx_f: npt.ArrayLike
    Fy_f: npt.ArrayLike
    Fx_b: npt.ArrayLike
    Fy_b: npt.ArrayLike
    Fn_f: npt.ArrayLike
    Ft_f: npt.ArrayLike
    Fn_b: npt.ArrayLike
    Ft_b: npt.ArrayLike
    Fx_fl: npt.ArrayLike
    Fy_fl: npt.ArrayLike
    Fx_fr: npt.ArrayLike
    Fy_fr: npt.ArrayLike
    Fx_bl: npt.ArrayLike
    Fy_bl: npt.ArrayLike
    Fx_br: npt.ArrayLike
    Fy_br: npt.ArrayLike
    Fn_fl: npt.ArrayLike
    Ft_fl: npt.ArrayLike
    Fn_fr: npt.ArrayLike
    Ft_fr: npt.ArrayLike
    Fn_bl: npt.ArrayLike
    Ft_bl: npt.ArrayLike
    Fn_br: npt.ArrayLike
    Ft_br: npt.ArrayLike

    def __getitem__(self, index):
        sliced_data = {}
        for field in fields(self):
            field_name = field.name
            field_value = getattr(self, field_name)

            sliced_data[field_name] = field_value[index]

        return self.__class__(**sliced_data)

def calc_wheel_forces(
        car: Vehicle,
        *,
        tire_front: Tire,
        tire_rear: Tire,
        wheel_angles: WheelAngles,
        side_slip_rad: float,
        single_track_on: bool = False,
        small_angles_on: bool = False
        ) -> WheelForces:
    ''' 
    Calculates X and Y compoenents of the force on each tire with respect to
    the vehicle reference frame. Also calculates the force component in the
    normal and tangential direction to the CG velocity vector
    '''
    #region - Calculate wheel forces on vehicle reference frame

    c_f = tire_front.cornering_stiffness_NperRad
    c_b = tire_rear.cornering_stiffness_NperRad
    toe_f = np.radians(car.toe_out_front_deg)
    toe_b = np.radians(car.toe_out_back_deg)

    Fx_fl_N = np.nan
    Fy_fl_N = np.nan
    Fx_fr_N = np.nan
    Fy_fr_N = np.nan
    Fx_bl_N = np.nan
    Fy_bl_N = np.nan
    Fx_br_N = np.nan
    Fy_br_N = np.nan

    if single_track_on:
        d = wheel_angles.steer
        a_f = wheel_angles.alpha_f
        a_b = wheel_angles.alpha_b

        Fx_b_N = 0
        Fy_b_N = -2*c_b*a_b
        if small_angles_on:
            Fx_f_N = 0
            Fy_f_N = -2*c_f*a_f
        
        elif not small_angles_on:
            Fx_f_N = +2*c_f*(a_f*np.sin(d))
            Fy_f_N = -2*c_f*(a_f*np.cos(d))

    elif not single_track_on:
        d_fl = wheel_angles.steer_fl
        d_fr = wheel_angles.steer_fr
        a_fl = wheel_angles.alpha_fl
        a_fr = wheel_angles.alpha_fr
        a_bl = wheel_angles.alpha_bl
        a_br = wheel_angles.alpha_br

        if small_angles_on:
            Fx_fl_N = +c_f*a_fl*(d_fl + toe_f)
            Fy_fl_N = -c_f*a_fl

            Fx_fr_N = +c_f*a_fr*(d_fl - toe_f)
            Fy_fr_N = -c_f*a_fr

            Fx_bl_N = +c_b*a_bl*(+toe_b)
            Fy_bl_N = -c_b*a_bl

            Fx_br_N = +c_b*a_br*(-toe_b)
            Fy_br_N = -c_b*a_br

        elif not small_angles_on:
            Fx_fl_N = +c_f*(a_fl*np.sin(d_fl + toe_f))
            Fy_fl_N = -c_f*(a_fl*np.cos(d_fl + toe_f))

            Fx_fr_N = +c_f*(a_fr*np.sin(d_fl - toe_f))
            Fy_fr_N = -c_f*(a_fr*np.cos(d_fl - toe_f))

            Fx_bl_N = +c_b*a_bl*np.sin(+toe_b)
            Fy_bl_N = -c_b*a_bl*np.cos(+toe_b)

            Fx_br_N = +c_b*a_br*np.sin(-toe_b)
            Fy_br_N = -c_b*a_br*np.cos(-toe_b)
        
        Fx_f_N = Fx_fl_N + Fx_fr_N
        Fy_f_N = Fy_fl_N + Fy_fr_N
        Fx_b_N = Fx_bl_N + Fx_br_N
        Fy_b_N = Fy_bl_N + Fy_br_N

    #endregion
    
    #region - Calculate the wheel force component normal and tangent to the CG 

    Fn_fl_N = np.nan
    Fn_fr_N = np.nan
    Fn_bl_N = np.nan
    Fn_br_N = np.nan
    Ft_fl_N = np.nan
    Ft_fr_N = np.nan
    Ft_bl_N = np.nan
    Ft_br_N = np.nan

    if single_track_on:
        if small_angles_on:
            Fn_f_N = 2*c_f*a_f
            Fn_b_N = 2*c_b*a_b
            Ft_f_N = 0
            Ft_b_N = 0

        elif not small_angles_on:
            Fn_f_N = 2*c_f*a_f*np.cos(side_slip_rad - d)
            Fn_b_N = 2*c_b*a_b*np.cos(side_slip_rad)
            Ft_f_N = 2*c_f*a_f*(-np.sin(side_slip_rad - d))
            Ft_b_N = 2*c_b*a_b*(-np.sin(side_slip_rad))

    elif not single_track_on:
        if small_angles_on:
            Fn_fl_N = c_f*a_fl
            Fn_fr_N = c_f*a_fr
            Fn_bl_N = c_b*a_bl
            Fn_br_N = c_b*a_br
            
            Ft_fl_N = c_f*a_fl*(-(side_slip_rad - d_fl))
            Ft_fr_N = c_f*a_fr*(-(side_slip_rad - d_fr))
            Ft_bl_N = c_b*a_bl*(-(side_slip_rad))
            Ft_br_N = c_b*a_br*(-(side_slip_rad))

        elif not small_angles_on:
            Fn_fl_N = c_f*a_fl*np.cos(side_slip_rad - d_fl)
            Fn_fr_N = c_f*a_fr*np.cos(side_slip_rad - d_fr)
            Fn_bl_N = c_b*a_bl*np.cos(side_slip_rad)
            Fn_br_N = c_b*a_br*np.cos(side_slip_rad)
            
            Ft_fl_N = c_f*a_fl*(-np.sin(side_slip_rad - d_fl))
            Ft_fr_N = c_f*a_fr*(-np.sin(side_slip_rad - d_fr))
            Ft_bl_N = c_b*a_bl*(-np.sin(side_slip_rad))
            Ft_br_N = c_b*a_br*(-np.sin(side_slip_rad))

        Fn_f_N = Fn_fl_N + Fn_fr_N
        Fn_b_N = Fn_bl_N + Fn_br_N
        Ft_f_N = Ft_fl_N + Ft_fr_N
        Ft_b_N = Ft_bl_N + Ft_br_N

    #endregion

    #region - Define output
    output_N = WheelForces(Fx_f=Fx_f_N, Fy_f=Fy_f_N,
                           Fx_b=Fx_b_N, Fy_b=Fy_b_N,
                           Fn_f=Fn_f_N, Ft_f=Ft_f_N,
                           Fn_b=Fn_b_N, Ft_b=Ft_b_N,
                           Fx_fl=Fx_fl_N, Fy_fl=Fy_fl_N,
                           Fx_fr=Fx_fr_N, Fy_fr=Fy_fl_N,
                           Fx_bl=Fx_bl_N, Fy_bl=Fy_br_N,
                           Fx_br=Fx_br_N, Fy_br=Fy_br_N,
                           Fn_fl=Fn_fl_N, Ft_fl=Ft_fl_N,
                           Fn_fr=Fn_fr_N, Ft_fr=Ft_fr_N,
                           Fn_bl=Fn_bl_N, Ft_bl=Ft_bl_N,
                           Fn_br=Fn_br_N, Ft_br=Ft_br_N)
    
    #endregion

    return output_N

@dataclass
class WheelMoments:
    Myaw_f: npt.ArrayLike
    Myaw_fl: npt.ArrayLike
    Myaw_fr: npt.ArrayLike
    Myaw_b: npt.ArrayLike
    Myaw_bl: npt.ArrayLike
    Myaw_br: npt.ArrayLike

    def __getitem__(self, index):
        sliced_data = {}
        for field in fields(self):
            field_name = field.name
            field_value = getattr(self, field_name)

            sliced_data[field_name] = field_value[index]

        return self.__class__(**sliced_data)

def calc_wheel_moments(
        car: Vehicle,
        wheel_forces: WheelForces,
        single_track_on: bool = False
        ):
    '''
    Calculates the yaw moment generated by each individual wheel
    '''
    #region - Calculate yaw moment generated by each individual wheel
    
    l_f = car.cg_to_front_m
    l_b = car.cg_to_back_m

    Myaw_fl_Nm = np.nan
    Myaw_fr_Nm = np.nan
    Myaw_bl_Nm = np.nan
    Myaw_br_Nm = np.nan

    if single_track_on:
        Myaw_f_Nm = +l_f*wheel_forces.Fy_f
        Myaw_b_Nm = -l_b*wheel_forces.Fy_b

    elif not single_track_on: 
        w_f = car.track_front_m
        w_b = car.track_back_m

        Myaw_fl_Nm = + l_f*wheel_forces.Fy_fl - w_f/2*wheel_forces.Fx_fl
        Myaw_fr_Nm = + l_f*wheel_forces.Fy_fr + w_f/2*wheel_forces.Fx_fr
        Myaw_bl_Nm = - l_b*wheel_forces.Fy_bl - w_b/2*wheel_forces.Fx_bl
        Myaw_br_Nm = - l_b*wheel_forces.Fy_br + w_b/2*wheel_forces.Fx_br

        Myaw_f_Nm = Myaw_fl_Nm + Myaw_fr_Nm
        Myaw_b_Nm = Myaw_bl_Nm + Myaw_br_Nm

    #endregion
    
    #region - Define output

    output_Nm = WheelMoments(Myaw_f=Myaw_f_Nm, Myaw_b=Myaw_b_Nm,
                             Myaw_fl=Myaw_fl_Nm, Myaw_fr=Myaw_fr_Nm,
                             Myaw_bl=Myaw_bl_Nm, Myaw_br=Myaw_br_Nm)
    
    #endregion

    return output_Nm

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
    radius_of_turn: npt.ArrayLike
    side_slip: npt.ArrayLike
    steering_wheel_input: npt.ArrayLike
    velocity: npt.ArrayLike
    dyaw_dt: npt.ArrayLike
    d2yaw_dt2: npt.ArrayLike
    dside_slip_dt: npt.ArrayLike
    Fcent: npt.ArrayLike
    Myaw: npt.ArrayLike

    def __getitem__(self, index):
        sliced_data = {}
        for field in fields(self):
            field_name = field.name
            field_value = getattr(self, field_name)

            if isinstance(field_value, np.ndarray):
                sliced_data[field_name] = field_value[index]
            else:
                sliced_data[field_name] = field_value

        return self.__class__(**sliced_data)

@dataclass
class ResultSet:
    car: Vehicle
    tire_front: Tire
    tire_rear: Tire
    motion_state: MotionState
    wheel_angles: WheelAngles
    wheel_forces: WheelForces
    wheel_moments: WheelMoments
    time: npt.ArrayLike
    single_track_on: bool
    small_angles_on: bool

    def __getitem__(self, index):
        motion_state_sliced = self.motion_state[index]
        wheel_angles_sliced = self.wheel_angles[index]
        wheel_forces_sliced = self.wheel_forces[index]
        wheel_moments_sliced = self.wheel_moments[index]
        time_sliced = self.time[index]

        return self.__class__(self.car,
                              self.tire_front,
                              self.tire_rear,
                              motion_state_sliced,
                              wheel_angles_sliced,
                              wheel_forces_sliced,
                              wheel_moments_sliced,
                              time_sliced,
                              self.single_track_on,
                              self.small_angles_on)

@validate_ssc_input
def steady_state_cornering(
        car: Vehicle,
        *,
        tire_front: Tire,
        tire_rear: Tire,
        input: dict,
        single_track_on: bool = False,
        small_angles_on: bool = False):
    '''
    Outputs velocity, turning radius, side slip angle and steering angle
    for a given vehicle and tires using the equations for a non-linear
    (no small angles assumption) double track model.
    Two of the four parameters need to be passed as input.
    The remaining two parameters are calculated.
    '''
    
    def motion_equations(car: Vehicle,
                         *,
                         tire_front: Tire,
                         tire_rear: Tire,
                         steering_wheel_input_rad: float,
                         side_slip_rad: float,
                         velocity_m_s: float,
                         radius_of_turn_m: float,
                         single_track_on: bool,
                         small_angles_on: bool):
        '''
        Function that performs the calculations for force and moment
        equilibrium. This function is passed to the solver, which then finds
        the solution for the steady state cornering condition
        '''
        
        wheel_angles_rad = calc_wheel_angles(
            car,
            steering_wheel_input_rad=steering_wheel_input_rad,
            side_slip_rad=side_slip_rad,
            dyaw_dt_rad_s=velocity_m_s/radius_of_turn_m,
            velocity_m_s=velocity_m_s,
            single_track_on=single_track_on,
            small_angles_on=small_angles_on)
        
        wheel_forces_N = calc_wheel_forces(
            car,
            tire_front=tire_front,
            tire_rear=tire_rear,
            wheel_angles=wheel_angles_rad,
            side_slip_rad=side_slip_rad,
            single_track_on=single_track_on,
            small_angles_on=small_angles_on)
        
        force_centripetal_N = wheel_forces_N.Fn_f + wheel_forces_N.Fn_b
        
        force_centrifugal_N = car.mass_kg*velocity_m_s**2/radius_of_turn_m

        force_resultant_N = force_centripetal_N + force_centrifugal_N
        
        wheel_moments_Nm = calc_wheel_moments(car,
                                              wheel_forces=wheel_forces_N,
                                              single_track_on=single_track_on)
        
        moment_yaw_Nm = wheel_moments_Nm.Myaw_f + wheel_moments_Nm.Myaw_b
        
        return (force_resultant_N, moment_yaw_Nm)
    
    #region - Definition of variables to solve for and initial guesses
    
    m = car.mass_kg
    lf = car.cg_to_front_m
    lb = car.cg_to_back_m
    mf = m*lb/(lf+lb)
    mb = m*lf/(lf+lb)
    cf_twin = 2*tire_front.cornering_stiffness_NperRad
    cb_twin = 2*tire_rear.cornering_stiffness_NperRad
    steer_ratio = car.steer_ratio_before_over_after_rack

    steer_grad = + mf/cf_twin - mb/cb_twin
    slip_grad = - mb/cb_twin

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
                    velocity_m_s=x[1],
                    single_track_on=single_track_on,
                    small_angles_on=small_angles_on)
            
            R = input["radius_of_turn_m"]
            slip = input["side_slip_rad"]
            steer = (lf+lb)/R + steer_grad/slip_grad*(slip - lb/R)
            vel = np.sqrt((R*slip - lb)/slip_grad)
            
            x0 = (steer, vel)

        case ("radius_of_turn_m", "steering_wheel_input_rad"):
            objective_function = \
                lambda x: motion_equations(
                    car = car,
                    tire_front=tire_front,
                    tire_rear=tire_rear,
                    radius_of_turn_m=input["radius_of_turn_m"],
                    side_slip_rad=x[0],
                    steering_wheel_input_rad=input["steering_wheel_input_rad"],
                    velocity_m_s=x[1],
                    single_track_on=single_track_on,
                    small_angles_on=small_angles_on)

            R = input["radius_of_turn_m"]
            s = input["steering_wheel_input_rad"]/steer_ratio
            slip = lb/R + slip_grad/steer_grad*(s - (lf+lb)/R)
            vel = np.sqrt((R*s - (lf+lb))/steer_grad)

            x0 = (slip, vel)

        case ("radius_of_turn_m", "velocity_m_s"):
            objective_function = \
                lambda x: motion_equations(
                    car = car,
                    tire_front=tire_front,
                    tire_rear=tire_rear,
                    radius_of_turn_m=input["radius_of_turn_m"],
                    side_slip_rad=x[0],
                    steering_wheel_input_rad=x[1],
                    velocity_m_s=input["velocity_m_s"],
                    single_track_on=single_track_on,
                    small_angles_on=small_angles_on)

            R = input["radius_of_turn_m"]
            v = input["velocity_m_s"]
            steer = ((lf+lb) + steer_grad*v**2)/R
            slip = (lb + slip_grad*v**2)/R
            
            x0 = (slip, steer)

        case ("side_slip_rad", "steering_wheel_input_rad"):
            objective_function = \
                lambda x: motion_equations(
                    car = car,
                    tire_front=tire_front,
                    tire_rear=tire_rear,
                    radius_of_turn_m=x[0],
                    side_slip_rad=input["side_slip_rad"],
                    steering_wheel_input_rad=input["steering_wheel_input_rad"],
                    velocity_m_s=x[1],
                    single_track_on=single_track_on,
                    small_angles_on=small_angles_on)

            slip = input["side_slip_rad"]
            s = input["steering_wheel_input_rad"]/steer_ratio
            radius = ((lf+lb) + steer_grad*v**2)/s
            vel = np.sqrt((s*lb - slip*(lf+lb))/(slip*steer_grad - s*slip_grad))
            
            x0 = (radius, vel)

        case ("side_slip_rad", "velocity_m_s"):
            objective_function = \
                lambda x: motion_equations(
                    car = car,
                    tire_front=tire_front,
                    tire_rear=tire_rear,
                    radius_of_turn_m=x[0],
                    side_slip_rad=input["side_slip_rad"],
                    steering_wheel_input_rad=x[1],
                    velocity_m_s=input["velocity_m_s"],
                    single_track_on=single_track_on,
                    small_angles_on=small_angles_on)

            slip = input["side_slip_rad"]
            v = input["velocity_m_s"]
            radius = (lb + slip_grad*v**2)/slip
            steer = slip*((lf+lb) + steer_grad*v**2)/(lb + slip_grad*v**2)
            
            x0 = (radius, steer)

        case ("steering_wheel_input_rad", "velocity_m_s"):
            objective_function = \
                lambda x: motion_equations(
                    car = car,
                    tire_front=tire_front,
                    tire_rear=tire_rear,
                    radius_of_turn_m=x[0],
                    side_slip_rad=x[1],
                    steering_wheel_input_rad=input["steering_wheel_input_rad"],
                    velocity_m_s=input["velocity_m_s"],
                    single_track_on=single_track_on,
                    small_angles_on=small_angles_on)

            s = input["steering_wheel_input_rad"]/steer_ratio
            v = input["velocity_m_s"]
            radius = ((lf+lb) + steer_grad*v**2)/s
            slip = s*((lb + slip_grad*v**2)/((lf+lb) + steer_grad*v**2))
            
            x0 = (radius, slip)
            
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
                np.nan.
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
                np.nan,
                np.nan,
                np.nan)

    #endregion

    #region - Definition of result_set to be returned

    # Motion state parameters given steady state condition
    motion_state.dyaw_dt = motion_state.velocity/motion_state.radius_of_turn
    motion_state.d2yaw_dt2 = 0
    motion_state.dside_slip_dt = 0
    
    wheel_angles = calc_wheel_angles(
        car,
        steering_wheel_input_rad=motion_state.steering_wheel_input,
        side_slip_rad=motion_state.side_slip,
        dyaw_dt_rad_s=motion_state.dyaw_dt,
        velocity_m_s=motion_state.velocity,
        single_track_on=single_track_on,
        small_angles_on=small_angles_on)
    
    wheel_forces = calc_wheel_forces(
        car,
        tire_front=tire_front,
        tire_rear=tire_rear,
        wheel_angles=wheel_angles,
        side_slip_rad=motion_state.side_slip,
        single_track_on=single_track_on,
        small_angles_on=small_angles_on)
    
    wheel_moments = calc_wheel_moments(
        car,
        wheel_forces=wheel_forces,
        single_track_on=single_track_on)
    
    motion_state.Fcent = wheel_forces.Fn_f + wheel_forces.Fn_b
    motion_state.Myaw = wheel_moments.Myaw_f + wheel_moments.Myaw_b

    result_set = ResultSet(
        car=car,
        tire_front=tire_front,
        tire_rear=tire_rear,
        motion_state=motion_state,
        wheel_angles=wheel_angles,
        wheel_forces=wheel_forces,
        wheel_moments=wheel_moments,
        time=np.nan,
        single_track_on=single_track_on,
        small_angles_on=small_angles_on)

    #endregion

    return result_set, solution

def calc_slip_ang_trans(
        slip_ang_static_rad: float,
        velocity_m_s: float,
        relaxation_length_m: float,
        slip_angle_previous_t_step: float,
        time_delta_s: float
        ) -> float:
    
    dalpha_dt = ((velocity_m_s / relaxation_length_m) * 
                 (slip_ang_static_rad - slip_angle_previous_t_step))
    
    slip_angle_trans_rad = (slip_angle_previous_t_step + 
                            dalpha_dt * time_delta_s)

    return slip_angle_trans_rad

def create_step_function(t_total: float, dt: float, t_rise: float,
                         t_start: float, height: float = 1) -> npt.NDArray:
    
    time = np.arange(0, t_total, dt)
    step = np.zeros_like(time)

    for i, t in enumerate(time):
        if t < t_start:
            step[i] = 0.0
        elif t < t_start + t_rise:
            step[i] = height * (t - t_start) / t_rise
        else:
            step[i] = height

    return step

@dataclass
class InitialConditions:
    side_slip_0: float
    dyaw_dt_0: float
    dside_slip_dt_0: float
    d2yaw_dt2_0: float

def transient_maneuver(
        car: Vehicle,
        *,
        tire_front: Tire,
        tire_rear: Tire,
        time_end_s: float,
        time_step_s: float,
        initial_conditions_SI: InitialConditions = InitialConditions(0,0,0,0),
        velocity_m_s: float,
        steering_wheel_input_f_of_t_rad: npt.NDArray[np.float64],
        single_track_on: bool = False,
        small_angles_on: bool = False
        ) -> ResultSet:

    time = np.arange(0, time_end_s, time_step_s)

    if steering_wheel_input_f_of_t_rad.shape != time.shape:
        raise ValueError(
            f"Input size mismatch! Expected 'steering_wheel_input_f_of_t_rad'"
            f"to be of shape {time.shape} due to the given time input, but got"
            f"{steering_wheel_input_f_of_t_rad.shape} instead."
        )

    #region - Preallocation of arrays

    np.full(len(time), np.nan)

    side_slip = np.full(len(time), np.nan)
    dyaw_dt = np.full(len(time), np.nan)
    dside_slip_dt = np.full(len(time), np.nan)
    d2yaw_dt2 = np.full(len(time), np.nan)
    radius_of_turn = np.full(len(time), np.nan)

    side_slip[0] = initial_conditions_SI.side_slip_0
    dyaw_dt[0] = initial_conditions_SI.dyaw_dt_0
    dside_slip_dt[0] = initial_conditions_SI.dside_slip_dt_0
    d2yaw_dt2[0] = initial_conditions_SI.d2yaw_dt2_0
    radius_of_turn[0] = velocity_m_s / (dyaw_dt[0] + side_slip[0])

    steer = np.full(len(time), np.nan)
    steer_fl = np.full(len(time), np.nan)
    steer_fr = np.full(len(time), np.nan)

    alpha_static_f = np.full(len(time), np.nan)
    alpha_static_b = np.full(len(time), np.nan)
    alpha_static_fl = np.full(len(time), np.nan)
    alpha_static_fr = np.full(len(time), np.nan)
    alpha_static_bl = np.full(len(time), np.nan)
    alpha_static_br = np.full(len(time), np.nan)
    
    alpha_trans_f = np.full(len(time), np.nan)
    alpha_trans_b = np.full(len(time), np.nan)
    alpha_trans_fl = np.full(len(time), np.nan)
    alpha_trans_fr = np.full(len(time), np.nan)
    alpha_trans_bl = np.full(len(time), np.nan)
    alpha_trans_br = np.full(len(time), np.nan)

    Fx_f = np.full(len(time), np.nan)
    Fy_f = np.full(len(time), np.nan)
    Fx_b = np.full(len(time), np.nan)
    Fy_b = np.full(len(time), np.nan)

    Fx_fl = np.full(len(time), np.nan)
    Fy_fl = np.full(len(time), np.nan)
    Fx_fr = np.full(len(time), np.nan)
    Fy_fr = np.full(len(time), np.nan)
    Fx_bl = np.full(len(time), np.nan)
    Fy_bl = np.full(len(time), np.nan)
    Fx_br = np.full(len(time), np.nan)
    Fy_br = np.full(len(time), np.nan)
    
    Fn_f = np.full(len(time), np.nan)
    Ft_f = np.full(len(time), np.nan)
    Fn_b = np.full(len(time), np.nan)
    Ft_b = np.full(len(time), np.nan)

    Fn_fl = np.full(len(time), np.nan)
    Ft_fl = np.full(len(time), np.nan)
    Fn_fr = np.full(len(time), np.nan)
    Ft_fr = np.full(len(time), np.nan)
    Fn_bl = np.full(len(time), np.nan)
    Ft_bl = np.full(len(time), np.nan)
    Fn_br = np.full(len(time), np.nan)
    Ft_br = np.full(len(time), np.nan)

    Myaw_f = np.full(len(time), np.nan)
    Myaw_b = np.full(len(time), np.nan)
    
    Myaw_fl = np.full(len(time), np.nan)
    Myaw_fr = np.full(len(time), np.nan)
    Myaw_bl = np.full(len(time), np.nan)
    Myaw_br = np.full(len(time), np.nan)

    Fcent = np.full(len(time), np.nan)
    Myaw = np.full(len(time), np.nan)

    #endregion

    #region - Time loop

    for t_idx in range(len(time[:-1])):
        wheel_angles_static = calc_wheel_angles(
            car,
            steering_wheel_input_rad=steering_wheel_input_f_of_t_rad[t_idx],
            side_slip_rad=side_slip[t_idx],
            dyaw_dt_rad_s=dyaw_dt[t_idx],
            velocity_m_s=velocity_m_s,
            single_track_on=single_track_on,
            small_angles_on=small_angles_on)
        
        steer[t_idx] = wheel_angles_static.steer
        steer_fl[t_idx] = wheel_angles_static.steer_fl
        steer_fr[t_idx] = wheel_angles_static.steer_fr
        
        alpha_static_f[t_idx] = wheel_angles_static.alpha_f
        alpha_static_b[t_idx] = wheel_angles_static.alpha_b
        alpha_static_fl[t_idx] = wheel_angles_static.alpha_fl
        alpha_static_fr[t_idx] = wheel_angles_static.alpha_fr
        alpha_static_bl[t_idx] = wheel_angles_static.alpha_bl
        alpha_static_br[t_idx] = wheel_angles_static.alpha_br

        if t_idx == 0:
            alpha_trans_f[0] = alpha_static_f[0]
            alpha_trans_b[0] = alpha_static_b[0]
            alpha_trans_fl[0] = alpha_static_fl[0]
            alpha_trans_fr[0] = alpha_static_fr[0]
            alpha_trans_bl[0] = alpha_static_bl[0]
            alpha_trans_br[0] = alpha_static_br[0]
        
        if single_track_on:
            alpha_trans_f[t_idx+1] = calc_slip_ang_trans(
            alpha_static_f[t_idx],
            velocity_m_s,
            tire_front.relaxation_length_m,
            alpha_trans_f[t_idx],
            time_step_s)

            alpha_trans_b[t_idx+1] = calc_slip_ang_trans(
            alpha_static_b[t_idx],
            velocity_m_s,
            tire_front.relaxation_length_m,
            alpha_trans_b[t_idx],
            time_step_s)

        elif not single_track_on:
            alpha_trans_fl[t_idx+1] = calc_slip_ang_trans(
                alpha_static_fl[t_idx],
                velocity_m_s,
                tire_front.relaxation_length_m,
                alpha_trans_fl[t_idx],
                time_step_s)
            
            alpha_trans_fr[t_idx+1] = calc_slip_ang_trans(
                alpha_static_fr[t_idx],
                velocity_m_s,
                tire_front.relaxation_length_m,
                alpha_trans_fr[t_idx],
                time_step_s)
            
            alpha_trans_bl[t_idx+1] = calc_slip_ang_trans(
                alpha_static_bl[t_idx],
                velocity_m_s,
                tire_front.relaxation_length_m,
                alpha_trans_bl[t_idx],
                time_step_s)
            
            alpha_trans_br[t_idx+1] = calc_slip_ang_trans(
                alpha_static_br[t_idx],
                velocity_m_s,
                tire_front.relaxation_length_m,
                alpha_trans_br[t_idx],
                time_step_s)
        
        wheel_angles_trans = WheelAngles(
            steer=wheel_angles_static.steer,
            steer_fl=wheel_angles_static.steer_fl,
            steer_fr=wheel_angles_static.steer_fr,
            alpha_f=alpha_trans_f[t_idx],
            alpha_fl=alpha_trans_fl[t_idx],
            alpha_fr=alpha_trans_fr[t_idx],
            alpha_b=alpha_trans_b[t_idx],
            alpha_bl=alpha_trans_bl[t_idx],
            alpha_br=alpha_trans_br[t_idx])
        
        wheel_forces = calc_wheel_forces(car,
                                         tire_front=tire_front,
                                         tire_rear=tire_rear,
                                         wheel_angles=wheel_angles_trans,
                                         side_slip_rad=side_slip[t_idx],
                                         single_track_on=single_track_on,
                                         small_angles_on=small_angles_on)
        
        
        Fx_f[t_idx] = wheel_forces.Fx_f
        Fy_f[t_idx] = wheel_forces.Fy_f
        Fx_b[t_idx] = wheel_forces.Fx_b
        Fy_b[t_idx] = wheel_forces.Fy_b
        
        Fx_fl[t_idx] = wheel_forces.Fx_fl
        Fy_fl[t_idx] = wheel_forces.Fy_fl
        Fx_fr[t_idx] = wheel_forces.Fx_fr
        Fy_fr[t_idx] = wheel_forces.Fy_fr
        Fx_bl[t_idx] = wheel_forces.Fx_bl
        Fy_bl[t_idx] = wheel_forces.Fy_bl
        Fx_br[t_idx] = wheel_forces.Fx_br
        Fy_br[t_idx] = wheel_forces.Fy_br
        
        Fn_f[t_idx] = wheel_forces.Fn_f
        Ft_f[t_idx] = wheel_forces.Ft_f
        Fn_b[t_idx] = wheel_forces.Fn_b
        Ft_b[t_idx] = wheel_forces.Ft_b
        
        Fn_fl[t_idx] = wheel_forces.Fn_fl
        Ft_fl[t_idx] = wheel_forces.Ft_fl
        Fn_fr[t_idx] = wheel_forces.Fn_fr
        Ft_fr[t_idx] = wheel_forces.Ft_fr
        Fn_bl[t_idx] = wheel_forces.Fn_bl
        Ft_bl[t_idx] = wheel_forces.Ft_bl
        Fn_br[t_idx] = wheel_forces.Fn_br
        Ft_br[t_idx] = wheel_forces.Ft_br
        
        Fcent[t_idx] = Fn_f[t_idx] + Fn_b[t_idx]

        wheel_moments = calc_wheel_moments(car,
                                           wheel_forces=wheel_forces,
                                           single_track_on=single_track_on)

        Myaw_f[t_idx] = wheel_moments.Myaw_f
        Myaw_b[t_idx] = wheel_moments.Myaw_b
        Myaw_fl[t_idx] = wheel_moments.Myaw_fl
        Myaw_fr[t_idx] = wheel_moments.Myaw_fr
        Myaw_bl[t_idx] = wheel_moments.Myaw_bl
        Myaw_br[t_idx] = wheel_moments.Myaw_br
        Myaw[t_idx] = Myaw_f[t_idx] + Myaw_b[t_idx]

        side_slip[t_idx+1] = (
            side_slip[t_idx] + (dside_slip_dt[t_idx] * time_step_s))
        
        dside_slip_dt[t_idx+1] = (
            -Fcent[t_idx] / (car.mass_kg*velocity_m_s) - dyaw_dt[t_idx])
        
        dyaw_dt[t_idx+1] = dyaw_dt[t_idx] + (d2yaw_dt2[t_idx] * time_step_s)
        d2yaw_dt2[t_idx+1] = Myaw[t_idx]/car.yaw_inertia_kgm2

        radius_of_turn[t_idx+1] = (velocity_m_s /
                                   (dyaw_dt[t_idx+1] + dside_slip_dt[t_idx+1]))

    #endregion
    
    #region - Definition of output

    motion_state = MotionState(
        radius_of_turn=radius_of_turn,
        side_slip=side_slip,
        steering_wheel_input=steering_wheel_input_f_of_t_rad,
        velocity=velocity_m_s,
        dyaw_dt=dyaw_dt,
        dside_slip_dt=dside_slip_dt,
        d2yaw_dt2=d2yaw_dt2,
        Fcent=Fcent,
        Myaw=Myaw)
    
    wheel_angles = WheelAngles(
        steer=steer,
        steer_fl=steer_fl,
        steer_fr=steer_fr,
        alpha_f=alpha_trans_f,
        alpha_fl=alpha_trans_fl,
        alpha_fr=alpha_trans_fr,
        alpha_b=alpha_trans_b,
        alpha_bl=alpha_trans_bl,
        alpha_br=alpha_trans_br)

    wheel_forces = WheelForces(Fx_f=Fx_f, Fy_f=Fy_f,
                               Fx_b=Fx_b, Fy_b=Fy_b,
                               Fn_f=Fn_f, Ft_f=Ft_f,
                               Fn_b=Fn_b, Ft_b=Ft_b,
                               Fx_fl=Fx_fl, Fy_fl=Fy_fl,
                               Fx_fr=Fx_fr, Fy_fr=Fy_fl,
                               Fx_bl=Fx_bl, Fy_bl=Fy_br,
                               Fx_br=Fx_br, Fy_br=Fy_br,
                               Fn_fl=Fn_fl, Ft_fl=Ft_fl,
                               Fn_fr=Fn_fr, Ft_fr=Ft_fr,
                               Fn_bl=Fn_bl, Ft_bl=Ft_bl,
                               Fn_br=Fn_br, Ft_br=Ft_br)
    
    wheel_moments = WheelMoments(
        Myaw_f=Myaw_f,
        Myaw_fl=Myaw_fl,
        Myaw_fr=Myaw_fr,
        Myaw_b=Myaw_b,
        Myaw_bl=Myaw_bl,
        Myaw_br=Myaw_br)
    
    result_set = ResultSet(
        car=car,
        tire_front=tire_front,
        tire_rear=tire_rear,
        motion_state=motion_state,
        wheel_angles=wheel_angles,
        wheel_forces=wheel_forces,
        wheel_moments=wheel_moments,
        time=time,
        single_track_on=single_track_on,
        small_angles_on=small_angles_on)

    #endregion

    return result_set

def draw_moment(line: Line2D, pos_x: float, pos_y: float, moment: float,
                scale_factor: float = 5e-5, color = 'red', linewidth = 1,
                markersize=5):
    
    radius = abs(moment)*scale_factor

    if moment >= 0:
        theta = np.linspace(np.radians(-45), np.radians(225), 100)
    elif moment < 0:
        theta = np.linspace(np.radians(225), np.radians(-45), 100)

    x = pos_x + radius * np.cos(theta)
    y = pos_y + radius * np.sin(theta)

    x_tip = x[-1]
    y_tip = y[-1]
    dx_tip = x[-2] - x[-1]
    dy_tip = y[-2] - y[-1]

    tip_angle_deg = np.degrees(np.arctan2(dy_tip, dx_tip))
    num_sides = 3
    rotated_triangle = (num_sides, 0, tip_angle_deg + 90)
    
    x = np.append(x, x_tip)
    y = np.append(y, y_tip)
    idx_tip = -1
    line.set_data(x, y)
    line.set_color(color)
    line.set_linewidth(linewidth)
    line.set_marker(rotated_triangle)
    line.set_markersize(markersize)
    line.set_markevery([idx_tip])

@dataclass
class Visualization:
    line_body: Line2D
    line_wheels: list[Line2D]
    quiver_vel_cg: Quiver
    quiver_vel_wheels: list[Quiver]
    line_path_cg: Line2D
    line_path_wheels: list[Line2D]
    quiver_wheel_forces: list[Quiver]
    line_wheel_moments: list[Line2D]
    quiver_force_result: Quiver
    quiver_force_res_norm: Quiver
    quiver_force_res_tang: Quiver
    line_moment_result: Line2D

def visu_vehicle_initialize() -> tuple[Figure, Axes, Visualization]:
    
    fig, ax = plt.subplots()
    line_body = ax.plot([], [], color='silver')[0]

    line_wheels = list()
    for _ in range(4):
        line_wheels.append(ax.plot([], [], color='black')[0])
    
    quiver_vel_cg = ax.quiver([], [], [], [], scale=10, scale_units='x',
                              width=0.001, headlength=3, headaxislength=3,
                              color='black')
    
    quiver_vel_wheels = list()
    for _ in range(4):
        quiver_vel_wheels.append(ax.quiver([], [], [], [], scale=10,
                                           scale_units='x', width=0.001,
                                           headlength=3, headaxislength=3,
                                           color='black'))
        
    line_path_cg = ax.plot([], [], color='grey', marker='x',
                           linestyle='dashed', linewidth=1)[0]
    
    line_path_wheels = list()
    for _ in range(4):
        line_path_wheels.append(ax.plot([], [], color='grey',
                                        linestyle='dashed', linewidth=1)[0])
        
    quiver_wheel_forces = list()
    for _ in range(4):
        quiver_wheel_forces.append(ax.quiver([], [], [], [], scale=1000,
                                             scale_units='x', width=0.001,
                                             headlength=3, headaxislength=3,
                                             color='red'))
        
    line_wheel_moments = list()
    for _ in range(4):
        line_wheel_moments.append(ax.plot([],[])[0])

    quiver_force_result = ax.quiver([], [], [], [], scale=1000, scale_units='x',
                                    width=0.001, headlength=3, headaxislength=3,
                                    color='royalblue')
    
    quiver_force_res_norm = ax.quiver([], [], [], [], scale=1000, 
                                          scale_units='x', width=0.001,
                                          headlength=3, headaxislength=3,
                                          color='turquoise')
    
    quiver_force_res_tang = ax.quiver([], [], [], [], scale=1000, 
                                          scale_units='x', width=0.001,
                                          headlength=3, headaxislength=3,
                                          color='turquoise')
    
    line_moment_result = ax.plot([], [])[0]

    vis = Visualization(line_body=line_body,
                        line_wheels=line_wheels,
                        quiver_vel_cg=quiver_vel_cg,
                        quiver_vel_wheels=quiver_vel_wheels,
                        line_path_cg=line_path_cg,
                        line_path_wheels=line_path_wheels,
                        quiver_wheel_forces=quiver_wheel_forces,
                        line_wheel_moments=line_wheel_moments,
                        quiver_force_result=quiver_force_result,
                        quiver_force_res_norm=quiver_force_res_norm,
                        quiver_force_res_tang=quiver_force_res_tang,
                        line_moment_result=line_moment_result)

    return fig, ax, vis

def visu_vehicle_update(
        vis: Visualization,
        result_set: ResultSet) -> list[Artist]:

    #region - Parameters used in multiple following sections

    rot_x_pts = lambda x,y,ang: x*np.cos(ang) - y*np.sin(ang)
    rot_y_pts = lambda x,y,ang: x*np.sin(ang) + y*np.cos(ang)
    
    lf = result_set.car.cg_to_front_m
    lb = result_set.car.cg_to_back_m

    if result_set.single_track_on:
        wheel_pos_x_m = np.array([+lf,-lb])
        wheel_pos_y_m = np.array([0,0])
    
    elif not result_set.single_track_on:
        wf = result_set.car.track_front_m
        wb = result_set.car.track_back_m
        wheel_pos_x_m = np.array([+lf,+lf,-lb,-lb])
        wheel_pos_y_m = np.array([+wf/2,-wf/2,+wb/2,-wb/2])
    
    #endregion

    #region - Vehicle body visualization

    if result_set.single_track_on:
        body_lines_x = np.array([+lf,-lb])
        body_lines_y = np.array([0,0])
    
    elif not result_set.single_track_on:
        body_lines_x = np.array([+lf,+lf,+lf,-lb,-lb,-lb])
        body_lines_y = np.array([+wf/2,-wf/2,0,0,+wb/2,-wb/2])
    
    rot_global = np.pi/2 - result_set.motion_state.side_slip

    body_lines_x_glob_ref = \
                rot_x_pts(body_lines_x,
                          body_lines_y,
                          rot_global)
    
    body_lines_y_glob_ref = \
                rot_y_pts(body_lines_x,
                          body_lines_y,
                          rot_global)
    
    vis.line_body.set_data(body_lines_x_glob_ref, body_lines_y_glob_ref)
    
    #endregion

    #region - Vehicle wheels visualization
    
    wheel_lines_x_m = np.array([+0.5, +0.5, -0.5, -0.5])
    wheel_lines_y_m = np.array([+0.2, -0.2, -0.2, +0.2])

    if result_set.single_track_on:
        num_wheels = 2
        steer = result_set.wheel_angles.steer
        wheel_angle_rad = [steer, 0]
    
    elif not result_set.single_track_on:
        num_wheels = 4
        st_fl = result_set.wheel_angles.steer_fl
        st_fr = result_set.wheel_angles.steer_fr
        toe_f = np.deg2rad(car.toe_out_front_deg)
        toe_b = np.deg2rad(car.toe_out_back_deg)

        wheel_angle_rad = [
            st_fl - np.sign(st_fl)*toe_f,
            st_fr + np.sign(st_fr)*toe_f,
            +np.sign(st_fl)*toe_b,
            -np.sign(st_fr)*toe_b]
    
    for i, x_shift, y_shift, wheel_ang in zip(range(num_wheels),
                                              wheel_pos_x_m,
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
        
        vis.line_wheels[i].set_data(inputs[0], inputs[1])

    #endregion

    #region - Velocity of CG visualization

    vel_cg = result_set.motion_state.velocity
    side_slip = result_set.motion_state.side_slip
    
    vel_cg_x = vel_cg * np.cos(side_slip)
    vel_cg_y = vel_cg * np.sin(side_slip)
    
    vel_cg_x_glob_ref = rot_x_pts(vel_cg_x, vel_cg_y, rot_global)
    vel_cg_y_glob_ref = rot_y_pts(vel_cg_x, vel_cg_y, rot_global)
    
    vis.quiver_vel_cg.set_offsets([0, 0])
    vis.quiver_vel_cg.set_UVC(vel_cg_x_glob_ref, vel_cg_y_glob_ref)
    
    #endregion

    #region - Velocity of individual wheels visualization
    
    # The yaw velocity component is the cross product between yaw rate
    # vector and wheel position with respect to the vehicle CG:
    dyaw_dt = result_set.motion_state.dyaw_dt
    vel_yaw_x = -dyaw_dt*np.array(wheel_pos_y_m)
    vel_yaw_y = +dyaw_dt*np.array(wheel_pos_x_m)
    
    vel_wheel_x = vel_cg_x + vel_yaw_x
    vel_wheel_y = vel_cg_y + vel_yaw_y
    
    for i, wheel_x, wheel_y, vel_x, vel_y in zip(range(num_wheels),
                                   wheel_pos_x_m, wheel_pos_y_m,
                                   vel_wheel_x, vel_wheel_y):
        
        vel_wheel_x_glob_ref = \
            rot_x_pts(vel_x, vel_y, rot_global)
        vel_wheel_y_glob_ref = \
            rot_y_pts(vel_x, vel_y, rot_global)
        
        wheel_x_glob_ref = \
            rot_x_pts(wheel_x, wheel_y, rot_global)
        wheel_y_glob_ref = \
            rot_y_pts(wheel_x, wheel_y, rot_global)
        
        vis.quiver_vel_wheels[i].set_offsets(
            [wheel_x_glob_ref, wheel_y_glob_ref])
        
        vis.quiver_vel_wheels[i].set_UVC(vel_wheel_x_glob_ref,
                                         vel_wheel_y_glob_ref)

    #endregion

    #region - Path radius of CG visualization
    
    unit_normal_to_vel_cg_x = -vel_cg_y/vel_cg
    unit_normal_to_vel_cg_y = +vel_cg_x/vel_cg
    
    radius_cg = vel_cg/dyaw_dt

    inst_cent_cg_x = radius_cg*unit_normal_to_vel_cg_x
    inst_cent_cg_y = radius_cg*unit_normal_to_vel_cg_y

    inst_cent_cg_glob_ref_x = \
        rot_x_pts(inst_cent_cg_x, inst_cent_cg_y, rot_global)
    inst_cent_cg_glob_ref_y = \
        rot_y_pts(inst_cent_cg_x, inst_cent_cg_y, rot_global)

    vis.line_path_cg.set_data([0, inst_cent_cg_glob_ref_x],
                              [0, inst_cent_cg_glob_ref_y])
    
    #endregion

    #region - Path radii of individual wheels
    
    for i, vel_x, vel_y, wheel_x, wheel_y in zip(range(num_wheels),
                                                  vel_wheel_x, vel_wheel_y, 
                                                  wheel_pos_x_m,
                                                  wheel_pos_y_m):
        
        vel_mag = np.sqrt(vel_x**2 + vel_y**2)
        
        unit_normal_to_vel_x = -vel_y/vel_mag
        unit_normal_to_vel_y = +vel_x/vel_mag

        radius = vel_mag/dyaw_dt
        
        inst_center_x = wheel_x + radius*unit_normal_to_vel_x
        inst_center_y = wheel_y + radius*unit_normal_to_vel_y

        inst_center_glob_ref_x = (
            rot_x_pts(inst_center_x, inst_center_y, rot_global))
        
        inst_center_glob_ref_y = (
            rot_y_pts(inst_center_x, inst_center_y, rot_global))

        wheel_glob_ref_x = rot_x_pts(wheel_x, wheel_y, rot_global)
        wheel_glob_ref_y = rot_y_pts(wheel_x, wheel_y, rot_global)
    
        path_radius_x = [wheel_glob_ref_x, inst_center_glob_ref_x]
        path_radius_y = [wheel_glob_ref_y, inst_center_glob_ref_y]
        vis.line_path_wheels[i].set_data(path_radius_x, path_radius_y)    
        
    #endregion

    #region - Force of individual wheels visualization

    if result_set.single_track_on:
        wheel_forcesX = [result_set.wheel_forces.Fx_f,
                         result_set.wheel_forces.Fx_b]
        
        wheel_forcesY = [result_set.wheel_forces.Fy_f,
                         result_set.wheel_forces.Fy_b]

    elif not result_set.single_track_on:
        wheel_forcesX = [result_set.wheel_forces.Fx_fl,
                         result_set.wheel_forces.Fx_fr,
                         result_set.wheel_forces.Fx_bl,
                         result_set.wheel_forces.Fx_br]

        wheel_forcesY = [result_set.wheel_forces.Fy_fl,
                         result_set.wheel_forces.Fy_fr,
                         result_set.wheel_forces.Fy_bl,
                         result_set.wheel_forces.Fy_br]
    
    for i, forceX, forceY, pos_x, pos_y in zip(range(num_wheels),
                                               wheel_forcesX,
                                               wheel_forcesY,
                                               wheel_pos_x_m,
                                               wheel_pos_y_m):

        pos_x_glob_ref = rot_x_pts(pos_x, pos_y, rot_global)
        pos_y_glob_ref = rot_y_pts(pos_x, pos_y, rot_global)
        
        forceX_glob_ref = rot_x_pts(forceX, forceY, rot_global)
        forceY_glob_ref = rot_y_pts(forceX, forceY, rot_global)

        vis.quiver_wheel_forces[i].set_offsets([pos_x_glob_ref, pos_y_glob_ref])
        vis.quiver_wheel_forces[i].set_UVC(forceX_glob_ref, forceY_glob_ref)

    #endregion

    #region - Moment of individual wheels visualization

    if result_set.single_track_on:
        wheel_moment_list = [result_set.wheel_moments.Myaw_f,
                             result_set.wheel_moments.Myaw_b]
    
    elif not result_set.single_track_on:
        wheel_moment_list = [result_set.wheel_moments.Myaw_fl,
                             result_set.wheel_moments.Myaw_fr,
                             result_set.wheel_moments.Myaw_bl,
                             result_set.wheel_moments.Myaw_br]
    
    for i, moment, pos_x, pos_y in zip(range(num_wheels),
                                       wheel_moment_list,
                                       wheel_pos_x_m,
                                       wheel_pos_y_m):

        pos_x_glob_ref = rot_x_pts(pos_x, pos_y, rot_global)
        pos_y_glob_ref = rot_y_pts(pos_x, pos_y, rot_global)

        draw_moment(vis.line_wheel_moments[i], pos_x=pos_x_glob_ref,
                    pos_y=pos_y_glob_ref, moment=moment, color='red')

    #endregion

    #region - Resultant force visualization

    force_res_x = result_set.wheel_forces.Fx_f + result_set.wheel_forces.Fx_b
    
    force_res_y = result_set.wheel_forces.Fy_f + result_set.wheel_forces.Fy_b
    
    force_res_x_glob_ref = rot_x_pts(force_res_x, force_res_y, rot_global)
    force_res_y_glob_ref = rot_y_pts(force_res_x, force_res_y, rot_global)

    vis.quiver_force_result.set_offsets([0, 0])
    vis.quiver_force_result.set_UVC(force_res_x_glob_ref, force_res_y_glob_ref)
    
    force_res_norm = (result_set.wheel_forces.Fn_f +
                      result_set.wheel_forces.Fn_b)
    
    force_res_norm_x = +force_res_norm*np.sin(side_slip)
    force_res_norm_y = -force_res_norm*np.cos(side_slip)
    
    force_res_norm_x_glob_ref = (
        rot_x_pts(force_res_norm_x, force_res_norm_y, rot_global))
    
    force_res_norm_y_glob_ref = (
        rot_y_pts(force_res_norm_x, force_res_norm_y, rot_global))
    
    vis.quiver_force_res_norm.set_offsets([0, 0])
    vis.quiver_force_res_norm.set_UVC(force_res_norm_x_glob_ref,
                                      force_res_norm_y_glob_ref)

    force_res_tang = (result_set.wheel_forces.Ft_f +
                      result_set.wheel_forces.Ft_b)

    force_res_tang_x = force_res_tang*np.cos(side_slip)
    force_res_tang_y = force_res_tang*np.sin(side_slip)

    force_res_tang_x_glob_ref = (
        rot_x_pts(force_res_tang_x, force_res_tang_y, rot_global))
    
    force_res_tang_y_glob_ref = (
        rot_y_pts(force_res_tang_x, force_res_tang_y, rot_global))
    
    vis.quiver_force_res_tang.set_offsets([0, 0])
    vis.quiver_force_res_tang.set_UVC(force_res_tang_x_glob_ref,
                                      force_res_tang_y_glob_ref)

    #endregion

    #region - Resultant moment visualization

    moment_res = (result_set.wheel_moments.Myaw_f +
                  result_set.wheel_moments.Myaw_b)

    draw_moment(vis.line_moment_result, pos_x=0, pos_y=0,
                moment=moment_res, color='royalblue')

    #endregion

    return [vis.line_body,
            *vis.line_wheels,
            vis.quiver_vel_cg,
            *vis.quiver_vel_wheels,
            vis.line_path_cg,
            *vis.line_path_wheels,
            *vis.quiver_wheel_forces,
            *vis.line_wheel_moments,
            vis.quiver_force_result,
            vis.quiver_force_res_norm,
            vis.quiver_force_res_tang,
            vis.line_moment_result]

def visu_generate_video(filename: str, result_set_array: ResultSet,
                        ax_xlim = None, ax_ylim = None, fig_size = None):
    
    fig, ax, vis = visu_vehicle_initialize()
    fig.set_size_inches(fig_size)
    ax.set_aspect('equal')
    ax.set_xlim(ax_xlim)
    ax.set_ylim(ax_ylim)
    total_frames = len(result_set_array.time)

    update = lambda result_set_scalar: visu_vehicle_update(vis,
                                                           result_set_scalar)
    
    ani = FuncAnimation(fig, update, frames=result_set_array, blit=True)
    
    ani.save(filename=filename, writer='ffmpeg', fps=30, dpi=150)

def plot_time_series(result_set: ResultSet) -> tuple[Figure, Axes]:
    fig, axs = plt.subplots(2, 3)
    fig.subplots_adjust(wspace=0.45)
    fig.set_size_inches(18,8)

    t = result_set.time
    axs[0,0].plot(t, result_set.motion_state.dyaw_dt,
                  label="$\\dot{\\psi}$")
    axs[0,0].plot(t, result_set.motion_state.dside_slip_dt,
                  label="$\\dot{\\beta}$")
    axs[0,0].set_ylabel('Angular Velocity (rad/s)')
    ax_twin = axs[0,0].twinx()
    ax_twin.set_ylim(np.degrees(axs[0,0].get_ylim()))
    ax_twin.set_ylabel('Angular Velocity (deg/s)')
    
    
    axs[1,0].plot(t, result_set.motion_state.d2yaw_dt2)
    axs[1,0].set_ylabel('Yaw Acceleration (rad/s²)')
    ax_twin = axs[1,0].twinx()
    ax_twin.set_ylim(np.degrees(axs[1,0].get_ylim()))
    ax_twin.set_ylabel('Yaw Acceleration (deg/s²)')
    
    axs[0,1].plot(t, result_set.wheel_angles.alpha_fl, label="$\\alpha_{fl}$")
    axs[0,1].plot(t, result_set.wheel_angles.alpha_fr, label="$\\alpha_{fr}$")
    axs[0,1].plot(t, result_set.wheel_angles.steer_fl, label="$\\delta_{fl}$")
    axs[0,1].plot(t, result_set.wheel_angles.steer_fr, label="$\\delta_{fr}$")
    axs[0,1].set_ylabel('Angle (rad)')
    ax_twin = axs[0,1].twinx()
    ax_twin.set_ylim(np.degrees(axs[0,1].get_ylim()))
    ax_twin.set_ylabel('Angle (deg)')
    
    axs[1,1].plot(t, result_set.wheel_angles.alpha_bl, label="$\\alpha_{bl}$")
    axs[1,1].plot(t, result_set.wheel_angles.alpha_br, label="$\\alpha_{br}$")
    axs[1,1].plot(t, result_set.motion_state.side_slip, label="$\\beta$")
    axs[1,1].set_ylabel('Angle (rad)')
    ax_twin = axs[1,1].twinx()
    ax_twin.set_ylim(np.degrees(axs[1,1].get_ylim()))
    ax_twin.set_ylabel('Angle (deg)')
    
    axs[0,2].plot(t, 1/result_set.motion_state.radius_of_turn)
    axs[0,2].set_ylabel('Curvature (1/m)')
    ax_twin = axs[0,2].twinx()
    ymin, ymax = axs[0,2].get_ylim()
    ax_twin.set_ylim(ymin, ymax)
    fig.canvas.draw()
    ticks_raw = axs[0,2].get_yticks()
    ticks_curv = [tick for tick in ticks_raw if ymin <= tick <= ymax]
    radius_labels = []
    for tick in ticks_curv:
        if abs(tick) < 1e-5:
            radius_labels.append(r"$\infty$")
        else:
            value = 1.0 / tick
            radius_labels.append(f"{value:.2f}")
    ax_twin.set_yticks(ticks_curv)
    ax_twin.set_yticklabels(radius_labels)
    ax_twin.set_ylabel('Radius of Curvature (m)')

    acc_lat = (result_set.motion_state.velocity**2 /
               result_set.motion_state.radius_of_turn)
    
    axs[1,2].plot(t, acc_lat)
    axs[1,2].set_ylabel('Lateral Acceleration (m/s²)')
    ax_twin = axs[1,2].twinx()
    ax_lim = np.array(axs[1,2].get_ylim())/9.81
    ax_twin.set_ylim(ax_lim)
    ax_twin.set_ylabel('Lateral Acceleration (g)')
    
    for ax in axs.flatten():
        ax.grid(True)
        ax.set_xlabel('Time (s)')
        ax.legend(fontsize='large')

    return fig, axs

def plot_result_comparison(result_sets: list[ResultSet]) -> tuple[Figure, Axes]:
    
    fig, axs = plt.subplots(3, 3)
    fig.subplots_adjust(wspace=0.45, hspace=0.45)
    fig.set_size_inches(18,8)
    fig.suptitle(
        f'Vehicle Velocity = {result_sets[0].motion_state.velocity*3.6} km/h',
        fontsize='xx-large')

    label_mapping = {(False, False): 'DT,LA',
                     (False, True): 'DT,SA',
                     (True, False): 'ST,LA',
                     (True, True): 'ST,SA'}

    for result_set in result_sets:
        t = result_set.time
        axs[0,0].plot(t, result_set.motion_state.dyaw_dt,
                      label=label_mapping[(result_set.single_track_on,
                                           result_set.small_angles_on)])
    axs[0,0].set_title('Yaw Velocity')
    axs[0,0].set_ylabel('Angular Velocity (rad/s)')
    ax_twin = axs[0,0].twinx()
    ax_twin.set_ylim(np.degrees(axs[0,0].get_ylim()))
    ax_twin.set_ylabel('Angular Velocity (deg/s)')

    for result_set in result_sets:
        t = result_set.time
        axs[0,1].plot(t, result_set.motion_state.side_slip,
                      label=label_mapping[(result_set.single_track_on,
                                           result_set.small_angles_on)])
    axs[0,1].set_title('Sideslip Angle')
    axs[0,1].set_ylabel('Angle (rad)')
    ax_twin = axs[0,1].twinx()
    ax_twin.set_ylim(np.degrees(axs[0,1].get_ylim()))
    ax_twin.set_ylabel('Angle (deg)')
    
    for result_set in result_sets:
        t = result_set.time
        axs[0,2].plot(t, 1/result_set.motion_state.radius_of_turn,
                      label=label_mapping[(result_set.single_track_on,
                                           result_set.small_angles_on)])
    axs[0,2].set_title('Inst. Turning Radius')
    axs[0,2].set_ylabel('Curvature (1/m)')
    ax_twin = axs[0,2].twinx()
    ymin, ymax = axs[0,2].get_ylim()
    ax_twin.set_ylim(ymin, ymax)
    fig.canvas.draw()
    ticks_raw = axs[0,2].get_yticks()
    ticks_curv = [tick for tick in ticks_raw if ymin <= tick <= ymax]
    radius_labels = []
    for tick in ticks_curv:
        if abs(tick) < 1e-5:
            radius_labels.append(r"$\infty$")
        else:
            value = 1.0 / tick
            radius_labels.append(f"{value:.2f}")
    ax_twin.set_yticks(ticks_curv)
    ax_twin.set_yticklabels(radius_labels)
    ax_twin.set_ylabel('Radius of Curvature (m)')
    
    for result_set in result_sets:
        t = result_set.time
        axs[1,0].plot(t, result_set.motion_state.d2yaw_dt2,
                      label=label_mapping[(result_set.single_track_on,
                                           result_set.small_angles_on)])
    axs[1,0].set_title('Yaw Acceleration')
    axs[1,0].set_ylabel('Angular Acceleration (rad/s²)')
    ax_twin = axs[1,0].twinx()
    ax_twin.set_ylim(np.degrees(axs[1,0].get_ylim()))
    ax_twin.set_ylabel('Angular Acceleration (deg/s²)')
    
    for result_set in result_sets:
        t = result_set.time
        axs[1,1].plot(t, result_set.motion_state.dside_slip_dt,
                      label=label_mapping[(result_set.single_track_on,
                                           result_set.small_angles_on)])
    axs[1,1].set_title('Sideslip Velocity')
    axs[1,1].set_ylabel('Angular Velocity (rad/s)')
    ax_twin = axs[1,1].twinx()
    ax_twin.set_ylim(np.degrees(axs[1,1].get_ylim()))
    ax_twin.set_ylabel('Angular Velocity (deg/s)')

    for result_set in result_sets:
        acc_lat = (result_set.motion_state.velocity**2 / 
                   result_set.motion_state.radius_of_turn)
        t = result_set.time
        axs[1,2].plot(t, acc_lat,
                      label=label_mapping[(result_set.single_track_on,
                                           result_set.small_angles_on)])
    axs[1,2].set_title('Centripetal Acceleration')
    axs[1,2].set_ylabel('Acceleration (m/s²)')
    ax_twin = axs[1,2].twinx()
    ax_lim = np.array(axs[1,2].get_ylim())/9.81
    ax_twin.set_ylim(ax_lim)
    ax_twin.set_ylabel('Acceleration (g)')

    for result_set in result_sets:
        t = result_set.time
        axs[2,0].plot(t, result_set.wheel_angles.steer,
                      label=label_mapping[(result_set.single_track_on,
                                           result_set.small_angles_on)])
    axs[2,0].set_title('Steer Angle')
    axs[2,0].set_ylabel('Angle (rad)')
    ax_twin = axs[2,0].twinx()
    ax_twin.set_ylim(np.degrees(axs[2,0].get_ylim()))
    ax_twin.set_ylabel('Angle (deg)')

    for result_set in result_sets:
        t = result_set.time
        if result_set.single_track_on:
            axs[2,1].plot(t, result_set.wheel_angles.alpha_f,
                      label=(r"$\alpha_{f}$" + " (" + 
                             label_mapping[(result_set.single_track_on,
                                            result_set.small_angles_on)] + ")"))
        elif not result_set.single_track_on:
            axs[2,1].plot(t, result_set.wheel_angles.alpha_fl,
                      label=(r"$\alpha_{fl}$" + " (" + 
                             label_mapping[(result_set.single_track_on,
                                            result_set.small_angles_on)] + ")"))
            axs[2,1].plot(t, result_set.wheel_angles.alpha_fr,
                      label=(r"$\alpha_{fr}$" + " (" + 
                             label_mapping[(result_set.single_track_on,
                                            result_set.small_angles_on)] + ")"))
    axs[2,1].set_title('Front Axle Slip Angle')
    axs[2,1].set_ylabel('Angle (rad)')
    ax_twin = axs[2,1].twinx()
    ax_twin.set_ylim(np.degrees(axs[2,1].get_ylim()))
    ax_twin.set_ylabel('Angle (deg)')

    for result_set in result_sets:
        t = result_set.time
        if result_set.single_track_on:
            axs[2,2].plot(t, result_set.wheel_angles.alpha_f,
                      label=(r"$\alpha_{b}$" + " (" + 
                             label_mapping[(result_set.single_track_on,
                                            result_set.small_angles_on)] + ")"))
        elif not result_set.single_track_on:
            axs[2,2].plot(t, result_set.wheel_angles.alpha_fl,
                      label=(r"$\alpha_{bl}$" + " (" + 
                             label_mapping[(result_set.single_track_on,
                                            result_set.small_angles_on)] + ")"))
            axs[2,2].plot(t, result_set.wheel_angles.alpha_fr,
                      label=(r"$\alpha_{br}$" + " (" + 
                             label_mapping[(result_set.single_track_on,
                                            result_set.small_angles_on)] + ")"))
    axs[2,2].set_title('Back Axle Slip Angle')
    axs[2,2].set_ylabel('Angle (rad)')
    ax_twin = axs[2,2].twinx()
    ax_twin.set_ylim(np.degrees(axs[2,2].get_ylim()))
    ax_twin.set_ylabel('Angle (deg)')

    for ax in axs.flatten():
        ax.grid(True)
        ax.set_xlabel('Time (s)')
        ax.legend(fontsize='small')

    return fig, axs

if __name__ == "__main__":
    car = Vehicle(mass_kg=1500,
                    cg_to_front_m = 2,
                    cg_to_back_m = 2,
                    toe_out_front_deg = -0.5,
                    toe_out_back_deg = 0.0,
                    track_front_m=1.8,
                    track_back_m=2,
                    ackermann_factor=1,
                    steer_ratio_before_over_after_rack= 13,
                    yaw_inertia_multiplier=1)

    car.yaw_inertia_kgm2 = car.yaw_inertia_estimation_kgm2()
    
    tire_front = Tire(cornering_stiffness_NperRad=20000,
                      relaxation_length_m=0.3)
    tire_rear = Tire(cornering_stiffness_NperRad=25000,
                     relaxation_length_m=0.3)

    # input = {"radius": 12, "velocity": 20/3.6}
    # result_set,_ = steady_state_cornering(car,
    #                        tire_front=tire_front,
    #                        tire_rear=tire_rear,
    #                        input=input)
    
    # fig, ax, vis = visu_vehicle_initialize()
    # ax.set_aspect('equal')
    # ax.set_xlim(-8,8)
    # ax.set_ylim(-4,4)
    # visu_vehicle_update(vis=vis, result_set=result_set)
    # fig.show()
    # print()

    steer_step = create_step_function(t_total=2.0, dt=0.01, t_rise=0.10,
                                      t_start=0.01, height=np.radians(90))
    
    input = {"tire_front": tire_front, "tire_rear": tire_rear,
             "time_end_s": 2.0, "time_step_s": 0.01, "velocity_m_s": 100/3.6,
             "steering_wheel_input_f_of_t_rad": steer_step}
    
    result_set_ST_SA = []
    input_ST_SA = [(False,False), (False,True), (True,False), (True,True)]
    for input_ST, input_SA in input_ST_SA:
        input_complete = {**input,
                          'single_track_on': input_ST,
                          'small_angles_on': input_SA}
        result_set_ST_SA.append(transient_maneuver(car, **input_complete))
    
    fig_ST_SA, axs_ST_SA = plot_result_comparison(result_set_ST_SA)
    fig_ST_SA.show()
    
    result_set_array = transient_maneuver(car, **input)

    fig_ts, axs_ts = plot_time_series(result_set_array)
    fig_ts.show()

    fig_vv, ax_vv, vis = visu_vehicle_initialize()
    fig_vv.set_size_inches(18,8)
    ax_vv.set_aspect('equal')
    ax_vv.set_xlim(-8,8)
    ax_vv.set_ylim(-4,4)
    
    # visu_generate_video('/home/mgriebeler/Videos/lateral_dynamics/test.mp4',
    #                     result_set_array=result_set_array,
    #                     ax_xlim=(-8,8), ax_ylim=(-4,4), fig_size=(18,8))

    # for i in range(len(result_set_array.wheel_angles.alpha_bl)):
    #     visu_vehicle_update(vis=vis, result_set=result_set_array[i])
    #     fig_vv.show()
    #     print()