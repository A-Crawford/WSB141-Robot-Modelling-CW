import RPPRRR_Manipulator_RTB 
import roboticstoolbox as rtb
import matplotlib as plt
import numpy as np

if __name__ == "__main__":
    RBTManipulator = RPPRRR_Manipulator_RTB.RPPRRRManipulator().model

    print(RBTManipulator)
    print(RBTManipulator.q)

    FK_Transform = RBTManipulator.fkine([0, 0, 0.5, 0, 0, 0, 0, 0])
    print('Forward Kinematics: \n', FK_Transform)

    IK_Sol = RBTManipulator.ikine_LM(FK_Transform)
    print('Inverse Kinematics of Foraward Kinematics: \n', IK_Sol)

    print(np.round(np.degrees(IK_Sol.q[:])))

    print('Error: ', np.linalg.norm(FK_Transform - RBTManipulator.fkine(IK_Sol.q)))

    RBTManipulator.plot([0, 0, 0.5, 0, 0, 0, 0, 0], block=True)