import numpy as np
import numpy.typing as npt

from scipy.optimize import root
from scipy.integrate import solve_ivp

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.quiver import Quiver

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
    steer_fl: npt.ArrayLike
    steer_fr: npt.ArrayLike
    alpha_fl: npt.ArrayLike
    alpha_fr: npt.ArrayLike
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
        np.atan2(np.tan(steer), 1 - ack*(wf/2)*np.tan(steer)/(lf+lb)) \
            + toe_f*np.sign(steer)

    steer_fr_rad = \
        np.atan2(np.tan(steer), 1 + ack*(wf/2)*np.tan(steer)/(lf+lb)) \
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
    Myaw_fl: npt.ArrayLike
    Myaw_fr: npt.ArrayLike
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
    radius_of_turn: npt.ArrayLike
    side_slip: npt.ArrayLike
    steering_wheel_input: npt.ArrayLike
    velocity: npt.ArrayLike
    dyaw_dt: npt.ArrayLike
    d2yaw_dt2: npt.ArrayLike
    dside_slip_dt: npt.ArrayLike
    Fcent: npt.ArrayLike
    Myaw_f: npt.ArrayLike
    Myaw_b: npt.ArrayLike
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
                              time_sliced)

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

    alpha_f = slip + lf/R - s
    alpha_b = slip - lb/R
    
    Ff = cf_twin * alpha_f
    Fb = cb_twin * alpha_b
    Fcent = Ff + Fb
    
    Myaw_f = Ff * lf
    Myaw_b = Fb * lb
    Myaw = Myaw_f + Myaw_b
    
    # Definition of output and return statement

    wheel_angles = WheelAngles(steer_fl=s,
                               steer_fr=s,
                               alpha_fl=alpha_f,
                               alpha_fr=alpha_f,
                               alpha_bl=alpha_b,
                               alpha_br=alpha_b)
    
    wheel_forces = WheelForces(Fx_fl=0, Fy_fl=Ff/2,
                               Fx_fr=0, Fy_fr=Ff/2,
                               Fx_bl=0, Fy_bl=Fb/2,
                               Fx_br=0, Fy_br=Fb/2,
                               Fn_fl=Ff/2, Ft_fl=0,
                               Fn_fr=Ff/2, Ft_fr=0,
                               Fn_bl=Fb/2, Ft_bl=0,
                               Fn_br=Fb/2, Ft_br=0)
    
    wheel_moments = WheelMoments(Myaw_fl=Myaw_f/2, Myaw_fr=Myaw_f/2,
                                 Myaw_bl=Myaw_b/2, Myaw_br=Myaw_b/2)
    
    motion_state = MotionState(radius_of_turn=R,
                               side_slip=slip,
                               steering_wheel_input=s*steer_ratio,
                               velocity=v,
                               dyaw_dt=v/R,
                               d2yaw_dt2=0,
                               dside_slip_dt=0,
                               Fcent=Fcent,
                               Myaw_f=Myaw_f,
                               Myaw_b=Myaw_b,
                               Myaw=Myaw)
    
    result_set = ResultSet(car, tire_front, tire_rear, motion_state,
                           wheel_angles, wheel_forces, wheel_moments,
                           time=np.nan)
    
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
        velocity_m_s=motion_state.velocity)
    
    wheel_forces = calc_wheel_forces(
        car,
        tire_front=tire_front,
        tire_rear=tire_rear,
        wheel_angles=wheel_angles,
        side_slip_rad=motion_state.side_slip)
    
    wheel_moments = calc_wheel_moments(
        car,
        wheel_forces=wheel_forces)
    
    motion_state.Fcent = (wheel_forces.Fn_fl + wheel_forces.Fn_fr +
                          wheel_forces.Fn_bl + wheel_forces.Fn_br)
    
    motion_state.Myaw_f = wheel_moments.Myaw_fl + wheel_moments.Myaw_fr
    motion_state.Myaw_b = wheel_moments.Myaw_bl + wheel_moments.Myaw_br
    motion_state.Myaw = motion_state.Myaw_f + motion_state.Myaw_b

    result_set = ResultSet(
        car=car,
        tire_front=tire_front,
        tire_rear=tire_rear,
        motion_state=motion_state,
        wheel_angles=wheel_angles,
        wheel_forces=wheel_forces,
        wheel_moments=wheel_moments,
        time=np.nan)

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

def tm_single_track(
        car: Vehicle,
        *,
        tire_front: Tire,
        tire_rear: Tire,
        time_end_s: float,
        time_step_s: float,
        initial_conditions_SI: InitialConditions = InitialConditions(0,0,0,0),
        velocity_m_s: float,
        steering_wheel_input_f_of_t_rad: npt.NDArray[np.float64]
        ) -> ResultSet:

    time = np.arange(0, time_end_s, time_step_s)

    if steering_wheel_input_f_of_t_rad.shape != time.shape:
        raise ValueError(
            f"Input size mismatch! Expected 'steering_wheel_input_f_of_t_rad'"
            f"to be of shape {time.shape} due to the given time input, but got"
            f"{steering_wheel_input_f_of_t_rad.shape} instead."
        )
    
    # Shorthand for the necessary parameters
    lf = car.cg_to_front_m
    lb = car.cg_to_back_m
    cf_twin = 2*tire_front.cornering_stiffness_NperRad
    cb_twin = 2*tire_rear.cornering_stiffness_NperRad
    steer_ratio = car.steer_ratio_before_over_after_rack

    steer_at_wheel = steering_wheel_input_f_of_t_rad/steer_ratio
    
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

    steer_fl = np.full(len(time), np.nan)
    steer_fr = np.full(len(time), np.nan)

    alpha_static_fl = np.full(len(time), np.nan)
    alpha_static_fr = np.full(len(time), np.nan)
    alpha_static_bl = np.full(len(time), np.nan)
    alpha_static_br = np.full(len(time), np.nan)
    alpha_trans_fl = np.full(len(time), np.nan)
    alpha_trans_fr = np.full(len(time), np.nan)
    alpha_trans_bl = np.full(len(time), np.nan)
    alpha_trans_br = np.full(len(time), np.nan)

    Fx_fl = np.full(len(time), np.nan)
    Fy_fl = np.full(len(time), np.nan)
    Fx_fr = np.full(len(time), np.nan)
    Fy_fr = np.full(len(time), np.nan)
    Fx_bl = np.full(len(time), np.nan)
    Fy_bl = np.full(len(time), np.nan)
    Fx_br = np.full(len(time), np.nan)
    Fy_br = np.full(len(time), np.nan)
    Fn_fl = np.full(len(time), np.nan)
    Ft_fl = np.full(len(time), np.nan)
    Fn_fr = np.full(len(time), np.nan)
    Ft_fr = np.full(len(time), np.nan)
    Fn_bl = np.full(len(time), np.nan)
    Ft_bl = np.full(len(time), np.nan)
    Fn_br = np.full(len(time), np.nan)
    Ft_br = np.full(len(time), np.nan)

    Myaw_fl = np.full(len(time), np.nan)
    Myaw_fr = np.full(len(time), np.nan)
    Myaw_bl = np.full(len(time), np.nan)
    Myaw_br = np.full(len(time), np.nan)

    Fcent = np.full(len(time), np.nan)
    Myaw_f = np.full(len(time), np.nan)
    Myaw_b = np.full(len(time), np.nan)
    Myaw = np.full(len(time), np.nan)

    #endregion

    #region - Time loop

    for t_idx in range(len(time[:-1])):
        steer_fl[t_idx] = steer_at_wheel[t_idx]
        steer_fr[t_idx] = steer_at_wheel[t_idx]
        
        v = velocity_m_s
        s = steer_at_wheel[t_idx]

        alpha_f = side_slip[t_idx] + lf*dyaw_dt[t_idx]/v - s
        alpha_b = side_slip[t_idx] - lb*dyaw_dt[t_idx]/v

        alpha_static_fl[t_idx] = alpha_f
        alpha_static_fr[t_idx] = alpha_f
        alpha_static_bl[t_idx] = alpha_b
        alpha_static_br[t_idx] = alpha_b

        if t_idx == 0:
            alpha_trans_fl[0] = alpha_static_fl[0]
            alpha_trans_fr[0] = alpha_static_fr[0]
            alpha_trans_bl[0] = alpha_static_bl[0]
            alpha_trans_br[0] = alpha_static_br[0]
        
        alpha_trans_fl[t_idx+1] = calc_slip_ang_trans(
            alpha_static_fl[t_idx],
            velocity_m_s,
            tire_front.relaxation_length_m,
            alpha_trans_fl[t_idx],
            time_step_s)
        
        alpha_trans_fr[t_idx+1] = alpha_trans_fl[t_idx+1]
        
        alpha_trans_bl[t_idx+1] = calc_slip_ang_trans(
            alpha_static_bl[t_idx],
            velocity_m_s,
            tire_front.relaxation_length_m,
            alpha_trans_bl[t_idx],
            time_step_s)
        
        alpha_trans_br[t_idx+1] = alpha_trans_bl[t_idx+1]
        
        Ff = -cf_twin * alpha_trans_fl[t_idx]
        Fb = -cb_twin * alpha_trans_bl[t_idx]

        Fx_fl[t_idx] = 0
        Fy_fl[t_idx] = Ff/2
        Fx_fr[t_idx] = 0
        Fy_fr[t_idx] = Ff/2
        Fx_bl[t_idx] = 0
        Fy_bl[t_idx] = Fb/2
        Fx_br[t_idx] = 0
        Fy_br[t_idx] = Fb/2
        Fn_fl[t_idx] = Ff/2
        Ft_fl[t_idx] = 0
        Fn_fr[t_idx] = Ff/2
        Ft_fr[t_idx] = 0
        Fn_bl[t_idx] = Fb/2
        Ft_bl[t_idx] = 0
        Fn_br[t_idx] = Fb/2
        Ft_br[t_idx] = 0
        Fcent[t_idx] = (Fn_fl[t_idx] + Fn_fr[t_idx] + 
                        Fn_bl[t_idx] + Fn_br[t_idx])
        
        Myaw_f[t_idx] = Ff * lf
        Myaw_b[t_idx] = -Fb * lb
        Myaw[t_idx] = Myaw_f[t_idx] + Myaw_b[t_idx]

        Myaw_fl[t_idx] = Myaw_f[t_idx]/2
        Myaw_fr[t_idx] = Myaw_f[t_idx]/2
        Myaw_bl[t_idx] = Myaw_b[t_idx]/2
        Myaw_br[t_idx] = Myaw_b[t_idx]/2

        side_slip[t_idx+1] = side_slip[t_idx] + (dside_slip_dt[t_idx] * time_step_s)
        dside_slip_dt[t_idx+1] = Fcent[t_idx] / (car.mass_kg*velocity_m_s) - dyaw_dt[t_idx]

        dyaw_dt[t_idx+1] = dyaw_dt[t_idx] + (d2yaw_dt2[t_idx] * time_step_s)
        d2yaw_dt2[t_idx+1] = Myaw[t_idx]/car.yaw_inertia_kgm2

        radius_of_turn[t_idx+1] = (car.mass_kg*velocity_m_s**2) / Fcent[t_idx]

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
        Myaw_f=Myaw_f,
        Myaw_b=Myaw_b,
        Myaw=Myaw)
    
    wheel_angles = WheelAngles(
        steer_fl=steer_fl,
        steer_fr=steer_fr,
        alpha_fl=alpha_trans_fl,
        alpha_fr=alpha_trans_fr,
        alpha_bl=alpha_trans_bl,
        alpha_br=alpha_trans_br)
    
    wheel_forces = WheelForces(
        Fx_fl=Fx_fl, Fy_fl=Fy_fl,
        Fx_fr=Fx_fr, Fy_fr=Fy_fr,
        Fx_bl=Fx_bl, Fy_bl=Fy_bl,
        Fx_br=Fx_br, Fy_br=Fy_br,
        Fn_fl=Fn_fl, Ft_fl=Ft_fl,
        Fn_fr=Fn_fr, Ft_fr=Ft_fr,
        Fn_bl=Fn_bl, Ft_bl=Ft_bl,
        Fn_br=Fn_br, Ft_br=Ft_br)
    
    wheel_moments = WheelMoments(
        Myaw_fl=Myaw_fl,
        Myaw_fr=Myaw_fr,
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
        time=time)

    #endregion

    return result_set

def transient_maneuver(
        car: Vehicle,
        *,
        tire_front: Tire,
        tire_rear: Tire,
        time_end_s: float,
        time_step_s: float,
        initial_conditions_SI: InitialConditions = InitialConditions(0,0,0,0),
        velocity_m_s: float,
        steering_wheel_input_f_of_t_rad: npt.NDArray[np.float64]
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

    steer_fl = np.full(len(time), np.nan)
    steer_fr = np.full(len(time), np.nan)

    alpha_static_fl = np.full(len(time), np.nan)
    alpha_static_fr = np.full(len(time), np.nan)
    alpha_static_bl = np.full(len(time), np.nan)
    alpha_static_br = np.full(len(time), np.nan)
    alpha_trans_fl = np.full(len(time), np.nan)
    alpha_trans_fr = np.full(len(time), np.nan)
    alpha_trans_bl = np.full(len(time), np.nan)
    alpha_trans_br = np.full(len(time), np.nan)

    Fx_fl = np.full(len(time), np.nan)
    Fy_fl = np.full(len(time), np.nan)
    Fx_fr = np.full(len(time), np.nan)
    Fy_fr = np.full(len(time), np.nan)
    Fx_bl = np.full(len(time), np.nan)
    Fy_bl = np.full(len(time), np.nan)
    Fx_br = np.full(len(time), np.nan)
    Fy_br = np.full(len(time), np.nan)
    Fn_fl = np.full(len(time), np.nan)
    Ft_fl = np.full(len(time), np.nan)
    Fn_fr = np.full(len(time), np.nan)
    Ft_fr = np.full(len(time), np.nan)
    Fn_bl = np.full(len(time), np.nan)
    Ft_bl = np.full(len(time), np.nan)
    Fn_br = np.full(len(time), np.nan)
    Ft_br = np.full(len(time), np.nan)

    Myaw_fl = np.full(len(time), np.nan)
    Myaw_fr = np.full(len(time), np.nan)
    Myaw_bl = np.full(len(time), np.nan)
    Myaw_br = np.full(len(time), np.nan)

    Fcent = np.full(len(time), np.nan)
    Myaw_f = np.full(len(time), np.nan)
    Myaw_b = np.full(len(time), np.nan)
    Myaw = np.full(len(time), np.nan)

    #endregion

    #region - Time loop

    for t_idx in range(len(time[:-1])):
        wheel_angles_static = calc_wheel_angles(
            car,
            steering_wheel_input_rad=steering_wheel_input_f_of_t_rad[t_idx],
            side_slip_rad=side_slip[t_idx],
            dyaw_dt_rad_s=dyaw_dt[t_idx],
            velocity_m_s=velocity_m_s
            )
        
        steer_fl[t_idx] = wheel_angles_static.steer_fl
        steer_fr[t_idx] = wheel_angles_static.steer_fr
        alpha_static_fl[t_idx] = wheel_angles_static.alpha_fl
        alpha_static_fr[t_idx] = wheel_angles_static.alpha_fr
        alpha_static_bl[t_idx] = wheel_angles_static.alpha_bl
        alpha_static_br[t_idx] = wheel_angles_static.alpha_br

        if t_idx == 0:
            alpha_trans_fl[0] = alpha_static_fl[0]
            alpha_trans_fr[0] = alpha_static_fr[0]
            alpha_trans_bl[0] = alpha_static_bl[0]
            alpha_trans_br[0] = alpha_static_br[0]
        
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
            wheel_angles_static.steer_fl,
            wheel_angles_static.steer_fr,
            alpha_trans_fl[t_idx],
            alpha_trans_fr[t_idx],
            alpha_trans_bl[t_idx],
            alpha_trans_br[t_idx])
        
        wheel_forces = calc_wheel_forces(car,
                                         tire_front=tire_front,
                                         tire_rear=tire_rear,
                                         wheel_angles=wheel_angles_trans,
                                         side_slip_rad=side_slip[t_idx])
        
        Fx_fl[t_idx] = wheel_forces.Fx_fl
        Fy_fl[t_idx] = wheel_forces.Fy_fl
        Fx_fr[t_idx] = wheel_forces.Fx_fr
        Fy_fr[t_idx] = wheel_forces.Fy_fr
        Fx_bl[t_idx] = wheel_forces.Fx_bl
        Fy_bl[t_idx] = wheel_forces.Fy_bl
        Fx_br[t_idx] = wheel_forces.Fx_br
        Fy_br[t_idx] = wheel_forces.Fy_br
        Fn_fl[t_idx] = wheel_forces.Fn_fl
        Ft_fl[t_idx] = wheel_forces.Ft_fl
        Fn_fr[t_idx] = wheel_forces.Fn_fr
        Ft_fr[t_idx] = wheel_forces.Ft_fr
        Fn_bl[t_idx] = wheel_forces.Fn_bl
        Ft_bl[t_idx] = wheel_forces.Ft_bl
        Fn_br[t_idx] = wheel_forces.Fn_br
        Ft_br[t_idx] = wheel_forces.Ft_br
        Fcent[t_idx] = (Fn_fl[t_idx] + Fn_fr[t_idx] + 
                        Fn_bl[t_idx] + Fn_br[t_idx])

        wheel_moments = calc_wheel_moments(car, wheel_forces)

        Myaw_fl[t_idx] = wheel_moments.Myaw_fl
        Myaw_fr[t_idx] = wheel_moments.Myaw_fr
        Myaw_bl[t_idx] = wheel_moments.Myaw_bl
        Myaw_br[t_idx] = wheel_moments.Myaw_br
        Myaw_f[t_idx] = Myaw_fl[t_idx] + Myaw_fr[t_idx]
        Myaw_b[t_idx] = Myaw_bl[t_idx] + Myaw_br[t_idx]
        Myaw[t_idx] = Myaw_f[t_idx] + Myaw_b[t_idx]

        side_slip[t_idx+1] = side_slip[t_idx] + (dside_slip_dt[t_idx] * time_step_s)
        dside_slip_dt[t_idx+1] = -Fcent[t_idx] / (car.mass_kg*velocity_m_s) - dyaw_dt[t_idx]
        
        dyaw_dt[t_idx+1] = dyaw_dt[t_idx] + (d2yaw_dt2[t_idx] * time_step_s)
        d2yaw_dt2[t_idx+1] = Myaw[t_idx]/car.yaw_inertia_kgm2

        radius_of_turn[t_idx+1] = velocity_m_s / (dyaw_dt[t_idx+1] + dside_slip_dt[t_idx+1])

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
        Myaw_f=Myaw_f,
        Myaw_b=Myaw_b,
        Myaw=Myaw)
    
    wheel_angles = WheelAngles(
        steer_fl=steer_fl,
        steer_fr=steer_fr,
        alpha_fl=alpha_trans_fl,
        alpha_fr=alpha_trans_fr,
        alpha_bl=alpha_trans_bl,
        alpha_br=alpha_trans_br)

    wheel_forces = WheelForces(
        Fx_fl=Fx_fl, Fy_fl=Fy_fl,
        Fx_fr=Fx_fr, Fy_fr=Fy_fr,
        Fx_bl=Fx_bl, Fy_bl=Fy_bl,
        Fx_br=Fx_br, Fy_br=Fy_br,
        Fn_fl=Fn_fl, Ft_fl=Ft_fl,
        Fn_fr=Fn_fr, Ft_fr=Ft_fr,
        Fn_bl=Fn_bl, Ft_bl=Ft_bl,
        Fn_br=Fn_br, Ft_br=Ft_br)
    
    wheel_moments = WheelMoments(
        Myaw_fl=Myaw_fl,
        Myaw_fr=Myaw_fr,
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
        time=time)

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
        result_set: ResultSet):

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
    
    for i, x_shift, y_shift, wheel_ang in zip(range(4),
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

    vel_cg_x = motion_state.velocity*np.cos(result_set.motion_state.side_slip)
    vel_cg_y = motion_state.velocity*np.sin(result_set.motion_state.side_slip)
    
    vel_cg_x_glob_ref = rot_x_pts(vel_cg_x, vel_cg_y, rot_global)
    vel_cg_y_glob_ref = rot_y_pts(vel_cg_x, vel_cg_y, rot_global)
    
    vis.quiver_vel_cg.set_offsets([0, 0])
    vis.quiver_vel_cg.set_UVC(vel_cg_x_glob_ref, vel_cg_y_glob_ref)
    
    #endregion

    #region - Velocity of individual wheels visualization
    
    for i, wheel_x, wheel_y in zip(range(4), wheel_pos_x_m, wheel_pos_y_m):
        
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
        
        vis.quiver_vel_wheels[i].set_offsets(
            [wheel_x_glob_ref, wheel_y_glob_ref])
        
        vis.quiver_vel_wheels[i].set_UVC(vel_wheel_x_glob_ref,
                                         vel_wheel_y_glob_ref)
        
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

    vis.line_path_cg.set_data([0, path_radius_cg_glob_ref_x],
                              [0, path_radius_cg_glob_ref_y])
    
    #endregion

    #region - Path radii of individual wheels
    
    path_radius_wheel_graphics = list()
    for i, wheel_x, wheel_y in zip(range(4), wheel_pos_x_m, wheel_pos_y_m):
        
        wheel_glob_ref_x = \
            rot_x_pts(wheel_x, wheel_y, rot_global)
        wheel_glob_ref_y = \
            rot_y_pts(wheel_x, wheel_y, rot_global)
        
        vis.line_path_wheels[i].set_data(
            [wheel_glob_ref_x, path_radius_cg_glob_ref_x],
            [wheel_glob_ref_y, path_radius_cg_glob_ref_y])
        
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
    
    for i, forceX, forceY, pos_x, pos_y in zip(range(4),
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
    
    wheel_moments = calc_wheel_moments(car=result_set.car, 
                                       wheel_forces=wheel_forces)

    wheel_moment_list = [wheel_moments.Myaw_fl, wheel_moments.Myaw_fr,
                          wheel_moments.Myaw_bl, wheel_moments.Myaw_br]
    
    for i, moment, pos_x, pos_y in zip(range(4),
                                       wheel_moment_list,
                                       wheel_pos_x_m,
                                       wheel_pos_y_m):

        pos_x_glob_ref = rot_x_pts(pos_x, pos_y, rot_global)
        pos_y_glob_ref = rot_y_pts(pos_x, pos_y, rot_global)

        draw_moment(vis.line_wheel_moments[i], pos_x=pos_x_glob_ref,
                    pos_y=pos_y_glob_ref, moment=moment, color='red')

    #endregion

    #region - Resultant force visualization

    force_res_x = (wheel_forces.Fx_fl + wheel_forces.Fx_fr + 
                   wheel_forces.Fx_bl + wheel_forces.Fx_br)
    
    force_res_y = (wheel_forces.Fy_fl + wheel_forces.Fy_fr + 
                   wheel_forces.Fy_bl + wheel_forces.Fy_br)
    
    force_res_x_glob_ref = rot_x_pts(force_res_x, force_res_y, rot_global)
    force_res_y_glob_ref = rot_y_pts(force_res_x, force_res_y, rot_global)

    vis.quiver_force_result.set_offsets([0, 0])
    vis.quiver_force_result.set_UVC(force_res_x_glob_ref, force_res_y_glob_ref)
    
    force_res_norm = (wheel_forces.Fn_fl + wheel_forces.Fn_fr +
                      wheel_forces.Fn_bl + wheel_forces.Fn_br)
    
    force_res_norm_x = +force_res_norm*np.sin(result_set.motion_state.side_slip)
    force_res_norm_y = -force_res_norm*np.cos(result_set.motion_state.side_slip)
    
    force_res_norm_x_glob_ref = (
        rot_x_pts(force_res_norm_x, force_res_norm_y, rot_global))
    
    force_res_norm_y_glob_ref = (
        rot_y_pts(force_res_norm_x, force_res_norm_y, rot_global))
    
    vis.quiver_force_res_norm.set_offsets([0, 0])
    vis.quiver_force_res_norm.set_UVC(force_res_norm_x_glob_ref,
                                      force_res_norm_y_glob_ref)

    force_res_tang = (wheel_forces.Ft_fl + wheel_forces.Ft_fr +
                      wheel_forces.Ft_bl + wheel_forces.Ft_br)

    force_res_tang_x = force_res_tang*np.cos(result_set.motion_state.side_slip)
    force_res_tang_y = force_res_tang*np.sin(result_set.motion_state.side_slip)

    force_res_tang_x_glob_ref = (
        rot_x_pts(force_res_tang_x, force_res_tang_y, rot_global))
    
    force_res_tang_y_glob_ref = (
        rot_y_pts(force_res_tang_x, force_res_tang_y, rot_global))
    
    vis.quiver_force_res_tang.set_offsets([0, 0])
    vis.quiver_force_res_tang.set_UVC(force_res_tang_x_glob_ref,
                                      force_res_tang_y_glob_ref)

    #endregion

    #region - Resultant moment visualization

    moment_res = (wheel_moments.Myaw_fl + wheel_moments.Myaw_fr +
                  wheel_moments.Myaw_bl + wheel_moments.Myaw_br)

    draw_moment(vis.line_moment_result, pos_x=0, pos_y=0, moment=moment_res, color='royalblue')

    #endregion

def plot_time_series(result_set: ResultSet) -> tuple[Figure, Axes]:
    fig, axs = plt.subplots(2, 3)
    fig.subplots_adjust(wspace=0.45)

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
            radius_labels.append("$\infty$")
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

if __name__ == "__main__":
    car = Vehicle(mass_kg=1500,
                    cg_to_front_m = 2,
                    cg_to_back_m = 2,
                    toe_out_front_deg = 0.0,
                    toe_out_back_deg = 0.0,
                    track_front_m=1.8,
                    track_back_m=2,
                    ackermann_factor=1,
                    steer_ratio_before_over_after_rack= 13,
                    yaw_inertia_multiplier=1)

    car.yaw_inertia_kgm2 = car.yaw_inertia_estimation_kgm2()
    
    tire_front = Tire(cornering_stiffness_NperRad=20000,
                      relaxation_length_m=0.3)
    tire_rear = Tire(cornering_stiffness_NperRad=15000,
                     relaxation_length_m=0.3)

    # input = {"radius": 12, "velocity": 20/3.6}
    # result_set,_ = steady_state_cornering(car,
    #                        tire_front=tire_front,
    #                        tire_rear=tire_rear,
    #                        input=input)
    
    # fig, ax, vis = visu_vehicle_initialize()
    # ax.set_aspect('equal')
    # visu_vehicle_update(vis=vis, result_set=result_set)
    # fig.show()
    # print()
    
    # result_set = ssc_single_track(car,
    #                        tire_front=tire_front,
    #                        tire_rear=tire_rear,
    #                        input=input)

    steer_step = create_step_function(t_total=5.0, dt=0.01, t_rise=0.10,
                                      t_start=0.01, height=np.radians(90))
    
    input = {"tire_front": tire_front, "tire_rear": tire_rear,
             "time_end_s": 5.0, "time_step_s": 0.01, "velocity_m_s": 50/3.6,
             "steering_wheel_input_f_of_t_rad": steer_step}
    
    
    result_set_f_of_t = transient_maneuver(car, **input)
    # result_set_f_of_t = tm_single_track(car, **input)
    
    fig_ts, axs_ts = plot_time_series(result_set_f_of_t)
    fig_ts.show()

    fig_vv, ax_vv, vis = visu_vehicle_initialize()
    ax_vv.set_aspect('equal')
    ax_vv.set_xlim(-8,8)
    ax_vv.set_ylim(-4,4)
    for i in range(len(result_set_f_of_t.wheel_angles.alpha_bl)):
        visu_vehicle_update(vis=vis, result_set=result_set_f_of_t[i])
        fig_vv.show()
        print()