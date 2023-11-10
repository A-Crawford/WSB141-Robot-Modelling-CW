import RPPRRR_Manipulator 
import roboticstoolbox as rtb
import numpy as np

if __name__ == "__main__":
    manipulator = RPPRRR_Manipulator.RPPRRRManipulator().model

    Transformation_Matrix = manipulator.fkine([0, 0.5, 0, 0, 0, 0])
    print(Transformation_Matrix)

    

    
