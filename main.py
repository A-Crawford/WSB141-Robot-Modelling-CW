import RPPRRR_Manipulator_RTB 
import RPPRRR_Manipulator_Numpy
import roboticstoolbox as rtb
import numpy as np

if __name__ == "__main__":
    RBTmanipulator = RPPRRR_Manipulator_RTB.RPPRRRManipulator().model
    NumpyManipulator = RPPRRR_Manipulator_Numpy.RPPRRRManipulator()

    Transformation_Matrix = RBTmanipulator.fkine([0, 0.5, 0, 0, 0, 0])
    RBTmanipulator.plot(RBTmanipulator.q)

    print('Numpy Ans: \n', NumpyManipulator.TBase_Tool)

    NumpyManipulator.plot_frames()

    input('Press any button to continue...')
