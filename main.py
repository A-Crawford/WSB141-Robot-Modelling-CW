import RPPRRR_Manipulator_RTB 
import roboticstoolbox as rtbf
from spatialmath import SE3
import numpy as np

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

if __name__ == "__main__":

    manipulator = RPPRRRManipulator()
    manipulator_fk = manipulator.forward_kinematics([0, 0.5, 0, 0, 0, 0])
    
    manipulator.plot([0, 0.5, 0, 0, 0, 0], block=True)

    ik_sol_1 = manipulator.inverse_kinematics(IK_TRANSFORM_1, display=True)
    ik_sol_2 = manipulator.inverse_kinematics(IK_TRANSFORM_2, display=True)
    ik_sol_3 = manipulator.inverse_kinematics(IK_TRANSFORM_3, display=True)

