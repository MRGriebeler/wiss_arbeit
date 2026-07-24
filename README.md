# Lateral Dynamics Simulation

## Project Description

This project uses Python to solve the movement equations governing lateral movement of a two-axle, front-axle-steered road vehicle using pneumatic tires.

## Reference Frames

### **Illustration** (Reference Frames)

![Illustration of the used reference frames](./images/reference_frames.png)

### Angle Measurement

All angles are defined as positive in the counter-clockwise direction and are measured **from vehicle/wheel** axis **to velocity vector**.

### Vehicle Reference Frame $(ê_x, ê_y)$

The vehicle reference frame has its **origin** at the vehicle **center of gravity (CG)**. The **x-axis** is **parallel** to the **vehicle longitudinal axis**, with **positive direction** pointing **towards the direction of travel**. The **y-axis** points to the **left** of the vehicle.

### CG-Velocity Reference Frame $(ê_n, ê_t)$

The CG-velocity reference frame is a **normal-tangential** reference with respect to the **CG-velocity**. It also has its **origin** at the vehicle **CG**. The orientation difference between the two reference frames is the vehicle **sideslip angle** ($\beta$), as it can be seen in the [illustration](#illustration-reference-frames).

## Model Inputs

### Vehicle Properties

Vehicle physical characteristics are contained within the class **Vehicle**.

```python
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
```

The _**yaw_inertia_kgm2**_ parameter is has no default value, but an estimation can be done after the vehicle instantiation by using the method _**yaw_inertia_estimation_kgm2()**_. This method estimates the yaw inertia based on the vehicle's mean track width and wheelbase using the expression for the polar moment of inertia of a filled rectangle:

$$ I_{rect} = \frac{1}{12}m(b^2+h^2) $$

The product of the previous expression is multiplied by _**yaw_inertia_multiplier**_ and returned by the method.

### Tire Properties

Tire physical characteristics are contained within the class **Tire**.

```python
@dataclass
class Tire:
    cornering_stiffness_NperRad: float = 30000.0
    relaxation_length_m: float = 0.3
```

## Calculation of Wheel Angles

### Steering Angles

#### **Output** (Steering Angles)

- $\delta_l$ : Steering angle of the **left** wheel
- $\delta_r$ : Steering angle of the **right** wheel

#### **Input** (Steering Angles)

- $\delta$ : Steering angle without compensation for the vehicle's track width
- $w_f$ : Front axle track width
- $w_b$ : Back (rear) axle track width
- $l_f$ : Distance between CG and front axle
- $l_b$ : Distance between CG and back axle
- $R$ : Turning radius
- $k_{ack}$ : Ackermann geometry factor

#### **Illustration** (Steering Angles)

The geometric construction during a left-hand turn is shown below:

![Steering Angle](images/steering_angle.png)

#### **Equations** (Steering Angles)

From the geometry, it follows that the steering angle without track width compensation is

$$ \tan\delta = \frac{l_f+l_b}{\sqrt{R^2-l_b^2}} $$

and the compensated steering angles for left and right wheels are

$$ \tan\delta_l = \tan\delta \bigg/ \left(1 - \frac{w_f}{2}\frac{k_{ack}}{\sqrt{R^2-l_b^2}} \right) $$

$$ \tan\delta_r = \tan\delta \bigg/ \left(1 + \frac{w_f}{2}\frac{k_{ack}}{\sqrt{R^2-l_b^2}} \right) $$

If $\delta$ is assumed to be small, then $R >> l_b$ and the equations become

$$ \delta = \frac{l_f+l_b}{R} $$

$$ \delta_l = \delta \bigg/ \left(1 - \frac{w_f}{2}\frac{k_{ack}}{R} \right) $$

$$ \delta_r = \delta \bigg/ \left(1 + \frac{w_f}{2}\frac{k_{ack}}{R} \right) $$

For a single-track model, $w_f = 0$ and thus

$$ \delta = \delta_l = \delta_r $$

Additionally, the previous equality is also true if the Ackermann factor $k_{ack}$ is set to 0, which means there's no Ackermann steering compensation. If $k_{ack} > 0$, the steering angle of the wheel internal to the curve will be higher than the steering angle of the outer wheel. For $k_{ack} = 1$, the steering follows the Ackermann geometry. Values higher than 1 (overcompensation) and lower than 0 (inverted compensation) are also possible.

### Static Tire Slip Angles

The static tire slip angles are the geometrically defined slip angles that determine the steady state slip angle values.

#### **Output** (Static Tire Slip Angles)

If the **single-track model** option is selected:

- $\alpha_f$ : Slip angle of the front tire
- $\alpha_b$ : Slip angle of the back tire

If the **double-track model** option is selected:

- $\alpha_{fl}$ : Slip angle of the front left tire
- $\alpha_{fr}$ : Slip angle of the front right tire
- $\alpha_{bl}$ : Slip angle of the back left tire
- $\alpha_{br}$ : Slip angle of the back right tire

#### **Input** (Static Tire Slip Angles)

- $l_f$ : Distance between CG and front axle
- $l_b$ : Distance between CG and back axle
- $\beta$ : Sideslip angle
- $v$ : Velocity magnitude at the CG
- $\dot\psi$ : Yaw velocity

If the **single-track model** option is selected:

- $\delta$ : Steering angle

If the **double-track model** option is selected:

- $w_f$ : Front axle track width
- $w_b$ : Back axle track width
- $\delta_l$ : Steering angle of the **left** wheel
- $\delta_r$ : Steering angle of the **right** wheel

#### **Illustration** (Static Tire Slip Angles)

The velocity at the front left wheel and its components is depicted below:

![Slip Angle](images/slip_angle.png)

#### **Equations** (StaticTire Slip Angles)

The velocity at any wheel is given by the sum of the CG velocity and the velocity due to the yaw velocity:

$$ \vec v_{wheel} = \vec v_{CG} + \vec v_{\psi} $$

In the vehicle reference frame, the CG velocity is

$$ \vec v_{CG} = v(\cos\beta ê_x + \sin\beta ê_y) $$

The velocity component due to the yaw velocity is given by

$$ \vec v_{\psi} = \frac{d\vec\psi}{dt} \times \vec r $$

where $\vec r$ is the position of the wheel with respect to the vehicle CG. For the [depicted case](#illustration-tire-slip-angles), the position is

$$ \vec r = l_f ê_x + \frac{w_f}{2} ê_y $$

The yaw angular velocity vector is

$$ \frac{d\vec\psi}{dt} = \dot\psi ê_z $$

Substituting the two previous expressions into the expression for $\vec v_\psi$:

$$ \vec v_\psi = (\dot\psi ê_z) \times (l_f ê_x + \frac{w_f}{2} ê_y) $$

$$ \vec v_\psi = \dot\psi \left( l_f (ê_z \times ê_x) + \frac{w_f}{2}(ê_z \times ê_y) \right) $$

and thus the velocity component due to the yaw rotation is

$$ \vec v_\psi = \dot\psi \left(-\frac{w_f}{2}ê_x + l_f ê_y \right) $$

Substituting the expressions for $\vec v_{CG}$ and $\vec v_\psi$ into the expression for $\vec v_{wheel}$ (for the [depicted case](#illustration-tire-slip-angles), $\vec v_{wheel} = \vec v_{fl}$, the front left wheel velocity):

$$ v_{fl,x} = v \cos\beta - \dot\psi \frac{w_f}{2} $$

$$ v_{fl,y} = v \sin\beta + \dot\psi l_f $$

The slip angle $\alpha_{fl}$ is the difference between the wheel velocity vector direction and the steering angle $\delta_{fl}$:

$$ \alpha_{fl} = \arctan \left(\frac{v_{fl,y}}{v_{fl,x}} \right) - \delta_{fl} $$

and thus

$$ \alpha_{fl} = \arctan \left(\frac{v \sin\beta + \dot\psi l_f}{v \cos\beta - \dot\psi \frac{w_f}{2}} \right) - \delta_{fl} $$

which can also be expressed as

$$ \alpha_{fl} = \arctan \left(\frac{\sin\beta + \frac{\dot\psi}{v} l_f}{ \cos\beta - \frac{\dot\psi}{v} \frac{w_f}{2}} \right) - \delta_{fl} $$

Under the assumption of small $(\alpha_{fl} + \delta_{fl})$ and small $\beta$, the expression becomes

$$ \alpha_{fl} = \frac{\beta + \frac{\dot\psi}{v} l_f}{ 1 - \frac{\dot\psi}{v} \frac{w_f}{2}} - \delta_{fl} $$

Furthermore, considering the case of a single track model ($w_f=0$), the expression (now for $\alpha_f$) simplifies to

$$ \alpha_f = \beta + \frac{\dot\psi}{v} l_f - \delta_{fl} $$

### Implementation (Calculation of Wheel Angles)

The previously detailed equations for steering and slip angles are implemented via the function `calc_wheel_angles()`, which takes the following inputs:

```python
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
```

The function returns a `WheelAngles` object, which is a container for the wheel angle related information. It is defined as

```python
@dataclass
class WheelAngles:
    steer_f:    numpy.typing.ArrayLike
    steer_fl:   numpy.typing.ArrayLike
    steer_fr:   numpy.typing.ArrayLike
    steer_bl:   numpy.typing.ArrayLike
    steer_br:   numpy.typing.ArrayLike
    alpha_f:    numpy.typing.ArrayLike
    alpha_fl:   numpy.typing.ArrayLike
    alpha_fr:   numpy.typing.ArrayLike
    alpha_b:    numpy.typing.ArrayLike
    alpha_bl:   numpy.typing.ArrayLike
    alpha_br:   numpy.typing.ArrayLike
```

### Transient Tire Slip Angles

The transient tire slip angles are the slip angle values between two slip angle steady states. During the process of slip angle change, the slip angle gradually progresses between the steady states based on the tire **relaxation length** parameter.

#### **Output** (Transient Tire Slip Angles)

- $\alpha_t$ : Transient slip angle

#### **Input** (Transient Tire Slip Angles)

- $\alpha$ : Current static slip angle
- $v$ : Vehicle CG velocity magnitude
- $\lambda$ : Tire relaxation length

#### **Equations** (Transient Tire Slip Angles)

The transient tire slip angle is modelled as a first order system:

$$ \frac{d\alpha_t}{dt} = \frac{v}{\lambda}(\alpha_t - \alpha) $$

#### **Implementation** (Transient Tire Slip Angles)

The previously described equation is implemented via the function `calc_slip_ang_trans()`, which takes the following inputs:

```python
def calc_slip_ang_trans(
        slip_ang_static_rad: float,
        velocity_m_s: float,
        relaxation_length_m: float,
        slip_angle_previous_t_step: float,
        time_delta_s: float
        ) -> float:
```

## Calculation of Wheel Forces

### Input (Calculation of Wheel Forces)

- $c_f$ : Front-tire cornering stiffness
- $c_b$ : Back-tire cornering stiffness

If the **single-track model** option is selected:

- $\alpha_f$ : Slip angle of the front tire
- $\alpha_b$ : Slip angle of the back tire

If the **double-track model** option is selected:

- $\alpha_{fl}$ : Slip angle of the front left tire
- $\alpha_{fr}$ : Slip angle of the front right tire
- $\alpha_{bl}$ : Slip angle of the back left tire
- $\alpha_{br}$ : Slip angle of the back right tire

Only necessary for the calculation of forces with respect to the **CG-Velocity** [reference frame](#cg-velocity-reference-frame):

- $\beta$ : Sideslip angle

### Wheel Forces in Vehicle Reference Frame

#### **Output** (Wheel Forces in Vehicle Reference Frame)

- $F_{x,f}$ : Front-axle force in the x-direction
- $F_{y,f}$ : Front-axle force in the y-direction
- $F_{x,b}$ : Back-axle force in the x-direction
- $F_{x,b}$ : Back-axle force in the y-direction

Additionally, if the **double-track model** option is selected:

- $F_{x,fl}$ : Front-left wheel force in the x-direction
- $F_{y,fl}$ : Front-left wheel force in the y-direction
- $F_{x,fr}$ : Front-right wheel force in the x-direction
- $F_{y,fr}$ : Front-right wheel force in the y-direction
- $F_{x,bl}$ : Back-left wheel force in the x-direction
- $F_{y,bl}$ : Back-left wheel force in the y-direction
- $F_{x,br}$ : Back-right wheel force in the x-direction
- $F_{y,br}$ : Back-right wheel force in the y-direction

#### **Equations** (Wheel Forces in Vehicle Reference Frame)

Refer to the [illustration](#illustration-reference-frames) for the visualization of force direction.

If the **single-track model** option is selected:

$$ F_{x,f} = +2c_f \alpha_{f} \sin\delta_{f} $$
$$ F_{y,f} = -2c_f \alpha_{f} \cos\delta_{f} $$
$$ F_{x,b} = +2c_b \alpha_{b} \sin\delta_{b} $$
$$ F_{y,b} = -2c_b \alpha_{b} \cos\delta_{b} $$

If the **double-track model** option is selected:

For the front axle:

$$ F_{x,fl} = +c_f \alpha_{fl} \sin\delta_{fl} $$
$$ F_{y,fl} = -c_f \alpha_{fl} \cos\delta_{fl} $$
$$ F_{x,fr} = +c_f \alpha_{fr} \sin\delta_{fr} $$
$$ F_{y,fr} = -c_f \alpha_{fr} \cos\delta_{fr} $$
$$ F_{x,f} = F_{x,fl} + F_{x,fr} $$
$$ F_{y,f} = F_{y,fl} + F_{y,fr} $$

For the back axle:

$$ F_{x,bl} = +c_b \alpha_{bl} \sin\delta_{bl} $$
$$ F_{y,bl} = -c_b \alpha_{bl} \cos\delta_{bl} $$
$$ F_{x,br} = +c_b \alpha_{br} \sin\delta_{br} $$
$$ F_{y,br} = -c_b \alpha_{br} \cos\delta_{br} $$
$$ F_{x,b} = F_{x,bl} + F_{x,br} $$
$$ F_{y,b} = F_{y,bl} + F_{y,br} $$

### Wheel Forces in CG-Velocity Reference Frame

#### **Output** (Wheel Forces in CG-Velocity Reference Frame)

- $F_{n,f}$ : Front-axle force in the normal-direction
- $F_{t,f}$ : Front-axle force in the tangential-direction
- $F_{n,b}$ : Back-axle force in the normal-direction
- $F_{t,b}$ : Back-axle force in the tangential-direction

Additionally, if the **double-track model** option is selected:

- $F_{n,fl}$ : Front-left wheel force in the normal-direction
- $F_{t,fl}$ : Front-left wheel force in the tangential-direction
- $F_{n,fr}$ : Front-right wheel force in the normal-direction
- $F_{t,fr}$ : Front-right wheel force in the tangential-direction
- $F_{n,bl}$ : Back-left wheel force in the normal-direction
- $F_{t,bl}$ : Back-left wheel force in the tangential-direction
- $F_{n,br}$ : Back-right wheel force in the normal-direction
- $F_{t,br}$ : Back-right wheel force in the tangential-direction

#### **Equations** (Wheel Forces in CG-Velocity Reference Frame)

If the **single-track model** option is selected:

$$ F_{t,f} = +2c_f \alpha_f \sin(\beta - \delta_f) $$
$$ F_{n,f} = -2c_f \alpha_f \cos(\beta - \delta_f) $$
$$ F_{t,b} = +2c_b \alpha_b \sin(\beta - \delta_b) $$
$$ F_{n,b} = -2c_b \alpha_b \cos(\beta - \delta_b) $$

If the **double-track model** option is selected:

For the front axle:

$$ F_{t,fl} = +c_f \alpha_{fl} \sin(\beta - \delta_{fl}) $$
$$ F_{n,fl} = -c_f \alpha_{fl} \cos(\beta - \delta_{fl}) $$
$$ F_{t,fr} = +c_f \alpha_{fr} \sin(\beta - \delta_{fr}) $$
$$ F_{n,fr} = -c_f \alpha_{fr} \cos(\beta - \delta_{fr}) $$
$$ F_{t,f} = F_{t,fl} + F_{t,fr} $$
$$ F_{n,f} = F_{n,fl} + F_{n,fr} $$

For the back axle:

$$ F_{t,bl} = +c_b \alpha_{bl} \sin(\beta - \delta_{bl}) $$
$$ F_{n,bl} = -c_b \alpha_{bl} \cos(\beta - \delta_{bl}) $$
$$ F_{t,br} = +c_b \alpha_{br} \sin(\beta - \delta_{br}) $$
$$ F_{n,br} = -c_b \alpha_{br} \cos(\beta - \delta_{br}) $$
$$ F_{t,b} = F_{t,bl} + F_{t,br} $$
$$ F_{n,b} = F_{n,bl} + F_{n,br} $$

### Implementation (Calculation of Wheel Forces)

The previously detailed equations for wheel forces are implemented via the function `calc_wheel_forces()`, which takes the following inputs:

```python
def calc_wheel_forces(
        *,
        tire_front: Tire,
        tire_rear: Tire,
        wheel_angles: WheelAngles,
        side_slip_rad: float,
        single_track_on: bool = False,
        small_angles_on: bool = False
        ) -> WheelForces:
```

The function returns a `WheelForces` object, which is a container for the wheel force related information. It is defined as

```python
@dataclass
class WheelForces:
    Fx_f:       numpy.typing.ArrayLike
    Fy_f:       numpy.typing.ArrayLike
    Fx_b:       numpy.typing.ArrayLike
    Fy_b:       numpy.typing.ArrayLike
    Fn_f:       numpy.typing.ArrayLike
    Ft_f:       numpy.typing.ArrayLike
    Fn_b:       numpy.typing.ArrayLike
    Ft_b:       numpy.typing.ArrayLike
    Fx_fl:      numpy.typing.ArrayLike
    Fy_fl:      numpy.typing.ArrayLike
    Fx_fr:      numpy.typing.ArrayLike
    Fy_fr:      numpy.typing.ArrayLike
    Fx_bl:      numpy.typing.ArrayLike
    Fy_bl:      numpy.typing.ArrayLike
    Fx_br:      numpy.typing.ArrayLike
    Fy_br:      numpy.typing.ArrayLike
    Fn_fl:      numpy.typing.ArrayLike
    Ft_fl:      numpy.typing.ArrayLike
    Fn_fr:      numpy.typing.ArrayLike
    Ft_fr:      numpy.typing.ArrayLike
    Fn_bl:      numpy.typing.ArrayLike
    Ft_bl:      numpy.typing.ArrayLike
    Fn_br:      numpy.typing.ArrayLike
    Ft_br:      numpy.typing.ArrayLike
```

## Calculation of Wheel Moments

### Input (Calculation of Wheel Moments)

- $w_f$ : Front axle track width
- $w_b$ : Back (rear) axle track width
- $l_f$ : Distance between CG and front axle
- $l_b$ : Distance between CG and back axle
- $F_{x,f}$ : Front-axle force in the x-direction
- $F_{y,f}$ : Front-axle force in the y-direction
- $F_{x,b}$ : Back-axle force in the x-direction
- $F_{x,b}$ : Back-axle force in the y-direction

### Output (Calculation of Wheel Moments)

- $M_{z,f}$ : Front-axle yaw moment
- $M_{z,b}$ : Back-axle yaw moment

Additionally, if the **double-track model** option is selected:

- $M_{z,fl}$ : Front-left wheel yaw moment
- $M_{z,fr}$ : Front-left wheel yaw moment
- $M_{z,bl}$ : Front-right wheel yaw moment
- $M_{z,br}$ : Front-right wheel yaw moment

### Equations (Calculation of Wheel Moments)

If the **single-track model** option is selected:

$$ M_{z,f} = +l_f F_{y,f} $$
$$ M_{z,b} = -l_b F_{y,b} $$

If the **double-track model** option is selected:

$$ M_{z,fl} = +l_f F_{y,fl} - \frac{w_f}{2} F_{x,fl} $$
$$ M_{z,fr} = +l_f F_{y,fr} + \frac{w_f}{2} F_{x,fr} $$
$$ M_{z,bl} = -l_b F_{y,bl} - \frac{w_b}{2} F_{x,bl} $$
$$ M_{z,br} = -l_b F_{y,br} + \frac{w_b}{2} F_{x,br} $$

$$ M_{z,f} = M_{z,fl} + M_{z,fr} $$
$$ M_{z,b} = M_{z,bl} + M_{z,br} $$

### Implementation (Calculation of Wheel Moments)

The previously detailed equations for wheel moments are implemented via the function `calc_wheel_moments()`, which takes the following inputs:

```python
def calc_wheel_moments(
        car: Vehicle,
        wheel_forces: WheelForces,
        single_track_on: bool = False
        ) -> WheelMoments:
```

The function returns a `WheelMoments` object, which is a container for the wheel moment related information. It is defined as

```python
@dataclass
class WheelMoments:
    Myaw_f:         numpy.typing.ArrayLike
    Myaw_fl:        numpy.typing.ArrayLike
    Myaw_fr:        numpy.typing.ArrayLike
    Myaw_b:         numpy.typing.ArrayLike
    Myaw_bl:        numpy.typing.ArrayLike
    Myaw_br:        numpy.typing.ArrayLike
```

## Vehicle Motion State

The container object `MotionState` groups the necessary information to characterize the instantaneous motion state of a vehicle. The attributes of `MotionState` are:

- $R$ : Instantaneous turning radius described by the vehicle CG
- $\beta$ : Sideslip angle
- $\delta_{sw}$ : Steering wheel angle - **not** steering angle at the wheel ($\delta$)
- $v$ : Velocity magnitude of the vehicle's CG
- $\dot\psi$ : Yaw angular velocity
- $\ddot\psi$ : Yaw angular acceleration
- $\dot\beta$ : Sideslip angular velocity
- $F_{cent}$ : Total centripetal force generated by the vehicle
- $M_z$ : Yaw moment generated by the vehicle

The implementation is shown next

```python
@dataclass
class MotionState:
    radius_of_turn:         numpy.typing.ArrayLike
    side_slip:              numpy.typing.ArrayLike
    steering_wheel_input:   numpy.typing.ArrayLike
    velocity:               numpy.typing.ArrayLike
    dyaw_dt:                numpy.typing.ArrayLike
    d2yaw_dt2:              numpy.typing.ArrayLike
    dside_slip_dt:          numpy.typing.ArrayLike
    Fcent:                  numpy.typing.ArrayLike
    Myaw:                   numpy.typing.ArrayLike
```

## Simulation Result Set

The container `ResultSet` groups the information provided as result of a simulation. The attributes of `ResultSet` are:

- `Vehicle` : container with [vehicle properties](#vehicle-properties)
- `Tire` Front Axle: container with [tire properties](#tire-properties)
- `Tire` Back Axle: container with [tire properties](#tire-properties)
- `MotionState`: container with [vehicle motion state](#vehicle-motion-state)
- `WheelAngles`: container with [wheel angles](#calculation-of-wheel-angles)
- `WheelForces`: container with [wheel forces](#calculation-of-wheel-forces)
- `WheelMoments`: container with [wheel moments](#calculation-of-wheel-moments)
- **Time**: array with the calculated discrete simulation time steps
- **Single-Track-On**: boolean parameter for double- or single-track-model
- **Small-Angles-On**: boolean parameter for small angle assumption

The implementation is shown next

```python
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
```

## Steady State Cornering

### Description (Steady State Cornering)

The simplest lateral dynamics maneuver is the **steady state cornering**. The vehicle **motion state** during a specified steady state cornering maneuver must **simultaneously** satisfy the two following conditions:

$$\begin{cases}

F_{cent} = mv²/R \\
M_z = 0

\end{cases}$$

In the previously stated equations, $F_{cent}$ and $M_z$ are calculated based on the [tire forces](#calculation-of-wheel-forces) and [moments](#calculation-of-wheel-moments) as follows:

$$ F_{cent} = F_{n,f} + F_{n,b} $$

$$ M_z = M_{z,f} + M_{z,b} $$

Due to the steady state condition, some of the parameter of the `MotionState` [container](#vehicle-motion-state) are zero:

- $\ddot\psi = 0$ (Yaw acceleration)
- $\dot\beta = 0$ (Sideslip angular velocity)
- $M_z = 0$ (Yaw moment)

Additionally, $F_{cent}$ and $\dot\psi$ can be derived from the remaining `MotionState` parameters and vehicle:

$$ F_{cent} = mv²/R $$

$$ \dot\psi = v/R $$

Four `MotionState` parameters remain:

- $R$ : Instantaneous turning radius described by the vehicle CG
- $\beta$ : Sideslip angle
- $\delta_{sw}$ : Steering wheel angle - **not** steering angle at the wheel ($\delta$)
- $v$ : Velocity magnitude of the vehicle's CG

The two conditions $(F_{cent} = mv^2/R$ and $Mz=0)$ can be used to determine two of the four parameters. As such, the other two parameters must be informed for the system of equations to be solvable.

### Output (Steady State Cornering)

The **Steady State Cornering** simulation returns a `ResultSet` [container](#simulation-result-set).

### Input (Steady State Cornering)

- `Vehicle` : container with [vehicle properties](#vehicle-properties)
- `Tire` Front Axle: container with [tire properties](#tire-properties)
- `Tire` Back Axle: container with [tire properties](#tire-properties)
- **Input**: Dictionary `{'state1': value1, 'state2': value2}` informing two of the following states:
    - `"radius_of_turn_m"`
    - `"side_slip_rad"`
    - `"steering_wheel_input_rad"`
    - `"velocity_m_s"`
- **Single-Track-On**: boolean parameter for double- or single-track-model
- **Small-Angles-On**: boolean parameter for small angle assumption

### Implementation (Steady State Cornering)

The previously described steady state cornering simulation is implemented as follows

```python
def steady_state_cornering(
        car: Vehicle,
        *,
        tire_front: Tire,
        tire_rear: Tire,
        input: dict,
        single_track_on: bool = False,
        small_angles_on: bool = False):
```

Inside the `steady_state_cornering()` function, the mathematical problem to be solved is defined used the `motion_equations()` function:

```python
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

    wheel_angles_rad = calc_wheel_angles(...)

    wheel_forces_N = calc_wheel_forces(...)

    force_centripetal_N = wheel_forces_N.Fn_f + wheel_forces_N.Fn_b

    force_centrifugal_N = car.mass_kg*velocity_m_s**2/radius_of_turn_m

    force_diff_N = force_centripetal_N + force_centrifugal_N

    wheel_moments_Nm = calc_wheel_moments(...)

    moment_yaw_Nm = wheel_moments_Nm.Myaw_f + wheel_moments_Nm.Myaw_b

    return (force_diff_N, moment_yaw_Nm)
```

The `motion_equations()` function is then passed to `scipy.optmize.root()` to solve the non-linear system of equations. The initial guess for the two unknown states are calculated based on the equations for the single-track-model with small-angle assumption. This calculation uses the following parameters:

- $m$ : Vehicle mass
- $l_f$ : Distance from front-axle to CG
- $l_b$ : Distance from back-axle to CG
- $c_f$ : Front tire cornering stiffness
- $c_b$ : Back tire cornering stiffness

Defining:

$$ m_f = m\left( \frac{l_b}{lf+lb}\right) $$

$$ m_b = m\left( \frac{l_f}{lf+lb}\right) $$

The steering gradient $(k_\delta)$ and sideslip gradient $(k_\beta)$ are:

$$ k_\delta = \frac{m_f}{c_f} - \frac{m_b}{c_b} $$

$$ k_\beta = - \frac{m_b}{c_b} $$

The initial guesses for the four motion states are:

- If $(R, \beta)$ are informed:
    $$ \delta_0 = \frac{l_f+l_b}{R} + \frac{k_\delta}{k_\beta}(\beta - \frac{l_b}{R}) $$
    $$ v_0 = \sqrt{\frac{R\beta - l_b}{k_\beta}} $$

- If $(R, \delta)$ are informed:
    $$ \beta_0 = \frac{l_b}{R} + \frac{k_\beta}{k_\delta}(\delta - \frac{l_f + l_b}{R}) $$
    $$ v_0 = \sqrt{\frac{R\delta - (l_f+l_b)}{k_\delta}} $$

- If $(R, v)$ are informed:
    $$ \delta_0 = \frac{l_f+l_b}{R} + k_\delta \frac{v^2}{R} $$
    $$ \beta_0 = \frac{l_b}{R} + k_\beta\frac{v^2}{R} $$

- If $(\beta, \delta)$ are informed:

    $$ R_0 = \frac{l_f+l_b}{\delta} + k_\delta \frac{v^2}{\delta} $$
    $$ v_0 = \sqrt{\frac{\delta l_b - \beta (l_f+l_b)}{\beta k_\delta - \delta k_\beta}} $$

- If $(\beta, v)$ are informed:
    $$ R_0 = \frac{lb}{\beta} + k_\beta\frac{v^2}{\beta} $$
    $$ \delta_0 = \beta\frac{(l_f+l_b) + k_\delta v^2}{l_b + k_\beta v^2} $$

- If $(\delta, v)$ are informed:
    $$ R_0 = \frac{l_f+l_b}{\delta} + k_\delta \frac{v^2}{\delta} $$
    $$ \beta_0 = \delta \frac{l_b + k_\beta v^2}{(l_f+l_b) + k_\delta v^2} $$

## Transient Maneuver

### Description (Transient Maneuver)

The **transient maneuver** allows for the simulation of a time-dependent steer input. During the simulation, a system of two coupled differential equations is solved. These are:

$$ \dot\beta = -\frac{F_{cent}}{mv} - \dot\psi $$

$$ \ddot\psi = \frac{M_z}{J_z} $$

where $J_z$ is the vehicle's yaw inertia.

Differently than the steady [state cornering simulation](#steady-state-cornering), the transient maneuver simulation allows for the motion state of the vehicle to be specified uniquely as steering wheel angle $(\delta_{sw}(t))$ and CG-velocity $(v)$. The steering wheel angle can depend on time, but the velocity is constant.

### Output (Transient Maneuver)

The **transient maneuver** simulation returns a `ResultSet` [container](#simulation-result-set).

### Input (Transient Maneuver)

- `Vehicle` : container with [vehicle properties](#vehicle-properties)
- `Tire` Front Axle: container with [tire properties](#tire-properties)
- `Tire` Back Axle: container with [tire properties](#tire-properties)
- $t_{end}$ : Simulation end time
- $\Delta t$ : Simulation time step
- $v$ : CG-Velocity magnitude (constant)
- $\delta_{sw}(t)$ : Steering wheel angle as time array
- **Single-Track-On**: boolean parameter for double- or single-track-model
- **Small-Angles-On**: boolean parameter for small angle assumption