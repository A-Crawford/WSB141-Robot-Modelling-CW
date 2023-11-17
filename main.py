import RPPRRR_Manipulator_RTB 
import roboticstoolbox as rtb
import matplotlib as plt
import numpy as np

if __name__ == "__main__":
    RBTmanipulator = RPPRRR_Manipulator_RTB.RPPRRRManipulator().model

    print(RBTmanipulator)
    
    RBTmanipulator.plot([0, 0, 0.5, 0, 0, 0, 0, 0], block=True)

    

    input('Press any button to continue...')
