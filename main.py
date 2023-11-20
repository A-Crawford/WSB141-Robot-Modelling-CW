import RPPRRR_Manipulator_RTB 
import roboticstoolbox as rtb
import matplotlib as plt
import numpy as np

from rpprrr_manipulator import RPPRRRManipulator

IK_Transform_1 = np.array(
    [
        [1, 0, 0, -0.3],
        [0, -1, 0, -0.25],
        [0, 0, -1, 0.2],
        [0, 0, 0, 1]
    ]
)

IK_Transform_2 = np.array(
    [
        [-0.8419, 0, 0.5396, 0.7684],
        [0, -1, 0, -0.25],
        [0.5396, 0, -0.8419, 1.925],
        [0, 0, 0, 1]
    ]
)

IK_Transform_3 = np.array(
    [
        [-0.0023, -1, 0.1, -0.257],
        [-0.002, 1, 0.2, -0.299],
        [-1, 0, -1, 3.34],
        [0, 0, 0, 1]
    ]
)

if __name__ == "__main__":

    manipulator = RPPRRRManipulator()
    manipualtor_fk = manipulator.forward_kinematics([0, 0.5, 0, 0, 0, 0])


    # RBTManipulator = RPPRRR_Manipulator_RTB.RPPRRRManipulator().model

    # print(RBTManipulator)
    # print(RBTManipulator.q)

    # FK_Transform = RBTManipulator.fkine([0, 0, 0.5, 0, 0, 0, 0, 0])
    # print('Forward Kinematics: \n', FK_Transform)

    # IK_Sol = RBTManipulator.ikine_LM(FK_Transform)
    # print('Inverse Kinematics of Foraward Kinematics: \n', IK_Sol)

    # IK_1_Sol = RBTManipulator.ikine_LM(IK_Transform_1)
    # print(IK_1_Sol)

    # IK_2_Sol = RBTManipulator.ikine_LM(IK_Transform_2)
    # print(IK_2_Sol)

    # IK_3_Sol = RBTManipulator.ikine_LM(IK_Transform_3)
    # print(IK_3_Sol)


    # print(np.round(np.degrees(IK_Sol.q[:])))

    # print('Error: ', np.linalg.norm(FK_Transform - RBTManipulator.fkine(IK_Sol.q)))

    # # RBTManipulator.plot([0, 0, 0.5, 0, 0, 0, 0, 0], block=True)
    # # RBTManipulator.plot(IK_1_Sol.q, block=True)