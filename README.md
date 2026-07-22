# Lateral Dynamics Simulation

## Project Description

This project uses Python to solve the movement equations governing lateral movement of a two-axle, front-axle-steered road vehicle using pneumatic tires.

## Reference Frames

### Reference Frames Illustration

![Illustration of the used reference frames](./images/reference_frames.png)

### Angle Measurement

All angles are defined as positive in the counter-clockwise direction and are measured **from vehicle/wheel** axis **to velocity vector**.

### Vehicle Reference Frame $(ê_x, ê_y)$

The vehicle reference frame has its **origin** at the vehicle **center of gravity (CG)**. The **x-axis** is **parallel** to the **vehicle longitudinal axis**, with **positive direction** pointing **towards the direction of travel**. The **y-axis** points to the **left** of the vehicle.

### CG-Velocity Reference Frame $(ê_n, ê_t)$

The CG-velocity reference frame is a **normal-tangential** reference with respect to the **CG-velocity**. It also has its **origin** at the vehicle **CG**. The orientation difference between the two reference frames is the vehicle **sideslip angle** ($\beta$), as it can be seen in the [illustration](#reference-frames).

## Model Inputs

### Vehicle Properties

Vehicle physical characteristics are contained within the class **Vehicle**.

```python
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
class Tire:
    cornering_stiffness_NperRad: float = 30000.0
    relaxation_length_m: float = 0.3
```

## Calculation of Wheel Angles

### Steering Angle

#### Output

- $\delta_l$ : Steering angle of the **left** wheel
- $\delta_r$ : Steering angle of the **right** wheel

#### Input

- $\delta$ : Steering angle without compensation for the vehicle's track width
- $w_f$ : Front axle track width
- $w_b$ : Back (rear) axle track width
- $l_f$ : Distance between CG and front axle
- $l_b$ : Distance between CG and back axle
- $R$ : Turning radius
- $k_{ack}$ : Ackermann geometry factor

#### Steering Angle Illustration

The geometric construction during a left-hand turn is shown below:

![Steering Angle](images/steering_angle.png)

#### Steering Angle Equations

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

Additionally, the previous equality is also true if the Ackermann factor $k_{ack}$ is set to 0, which means there's no Ackermann steering compensation.
