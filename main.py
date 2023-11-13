import RPPRRR_Manipulator_RTB 
import roboticstoolbox as rtb
import numpy as np

if __name__ == "__main__":
    manipulator = RPPRRR_Manipulator_RTB.RPPRRRManipulator().model

    Transformation_Matrix = manipulator.fkine([0, 0.5, 0, 0, 0, 0])
    print(Transformation_Matrix)

    manipulator.plot(manipulator.q)

    input('Press any button to continue...')

    

    
