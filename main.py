import RPPRRR_Manipulator_RTB 
import roboticstoolbox as rtb
import matplotlib as plt
import numpy as np

if __name__ == "__main__":
    RBTmanipulator = RPPRRR_Manipulator_RTB.RPPRRRManipulator().model

    print(RBTmanipulator)
    RBTmanipulator.plot([0, 0, 0.5, 0, 0, 0, 0, 0], block=True)

    FKTransform = RBTmanipulator.fkine([0, 0, 0.5, 0, 0, 0, 0, 0]) #Base=0, 1, 2, 3, 4, 5, 6, Tool=0 
    print("Forward Kinematics with joint angles [0, 0.5, 0, 0, 0]: \n", FKTransform)

    RBTmanipulator.plot(RBTmanipulator.q)

    input('Press any button to continue...')
