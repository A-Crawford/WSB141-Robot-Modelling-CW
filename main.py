import RPPRRR_Manipulator_RTB 
import roboticstoolbox as rtbf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from spatialmath import SE3
import numpy as np
import sympy as sy

from rpprrr_manipulator import RPPRRRManipulator

IK_TRANSFORM_1 = np.array(
    [
        [1, 0, 0, -0.3],
        [0, -1, 0, -0.25],
        [0, 0, -1, 0.2],
        [0, 0, 0, 1]
    ]
)

IK_TRANSFORM_2 = np.array(
    [
        [-0.8419, 0, 0.5396, 0.7684],
        [0, -1, 0, -0.25],
        [0.5396, 0, -0.8419, 1.925],
        [0, 0, 0, 1]
    ]
)

IK_TRANSFORM_3 = np.array(
    [
        [-0.0023, -1, 0.1, -0.257],
        [-0.002, 1, 0.2, -0.299],
        [-1, 0, -1, 3.34],
        [0, 0, 0, 1]
    ]
)

JOINT_VELOCITIES = np.array(
    [0, 15, 0, 0.1, -30, 15, 10, 0]
)

STATIC_FORCE_TRANSFORM = np.array(
    [
        [0.6791, -0.6403, 0.359, -0.4475],
        [-0.6403, 0.2775, -0.7162, -0.335],
        [0.359, 0.7162, 0.5985, 0.599],
        [0, 0, 0, 1]
    ]
)

if __name__ == "__main__":

    manipulator = RPPRRRManipulator()
    sympy_manipulator = RPPRRRManipulator.RPPRRRManipulatorSympy()

    manipulator_fk = manipulator.forward_kinematics([0, 0.5, 0, 0, 0, 0])
    print(manipulator_fk)
    
    jacobian, linear_velocities, angular_velocities = manipulator.joint_velcoities(joint_angles=[0, -np.radians(60), 0.40, 0.10, np.radians(90), np.radians(180), np.radians(90), 0], joint_velocities=JOINT_VELOCITIES)
    
    print("Linear Velocities [X, Y, Z]:", linear_velocities)
    print("Angular Velocities [X, Y, Z]:",angular_velocities)
    
    manipulator.static_torques(mass=0.2, g=9.8, transform=STATIC_FORCE_TRANSFORM)
    
    # print(np.round(np.array(sy.simplify(sympy_manipulator.TB_T_FK)).astype(float), 2))
    

    # TB_T = np.array(sy.simplify(sympy_manipulator.TB_T_FK)).astype(float)
    # print(TB_T)

    # manipulator.plot([0, 0, 0.5, 0, 0, 0, 0, 0], block=True)

    # ik_sol_1 = manipulator.inverse_kinematics(IK_TRANSFORM_1, display=True)
    # ik_sol_2 = manipulator.inverse_kinematics(IK_TRANSFORM_2, display=True)
    # ik_sol_3 = manipulator.inverse_kinematics(IK_TRANSFORM_3, display=True)

