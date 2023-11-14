import roboticstoolbox as rtb
import numpy as np
import RPPRRR_Manipulator_Numpy


# Robotics Toolbox Solution
class RPPRRRManipulator():
    def __init__(self):
        self.L0 = 0.10
        self.L1 = 0.20
        self.L2 = 0.30
        self.L3 = 0.30
        self.L4 = 0.10
        self.L5 = 0.05

        self.theta1 = 0
        self.theta4 = 0
        self.theta5 = 0
        self.theta6 = 0

        self.d2 = 0.5
        self.d3 = 0

        self.DH_Table = self.create_DH_table()
        self.model = self.create_RTB_robot()
        T0_2 = self.Base_Tool_Transforms()

    def create_DH_table(self):
        # Default DH table is we use is Ln, Alpha, d, offset
        # Robotics Toolbox uses d, a==Ln, alpha, offset
        # Will be using Robotics Toolbox conventiion
        # Could craete instance of rbt.serialLink directly but this allows for adjustment of table indepenetly
        dh_table = [
            [0, 0, self.L0, 0],
            [0, 0, 0, self.theta1],
            [0, 0, self.d2, 0],
            [0, np.radians(90), self.d3, 0],
            [0, 0, self.L1, self.theta4],
            [self.L2, np.radians(90), self.L5, self.theta5],
            [0, 0, self.L3, self.theta6],
            [0, 0, self.L4, 0]
        ]
        return dh_table
    
    def create_RTB_robot(self):
        # R P P R R R 
        robot = rtb.DHRobot(
            [
                rtb.PrismaticDH(q=self.DH_Table[0][2]), # Base - Added for matricies, not an actual joint to be used

                rtb.RevoluteDH(a=self.DH_Table[1][0], alpha=self.DH_Table[1][1], d=self.DH_Table[1][2], offset=self.DH_Table[1][3]),

                rtb.PrismaticDH(a=self.DH_Table[2][0], alpha=self.DH_Table[2][1], q=self.DH_Table[2][2], offset=self.DH_Table[2][3]),
                rtb.PrismaticDH(a=self.DH_Table[3][0], alpha=self.DH_Table[3][1], q=self.DH_Table[3][2], offset=self.DH_Table[3][3]),

                rtb.RevoluteDH(a=self.DH_Table[4][0], alpha=self.DH_Table[4][1], d=self.DH_Table[4][2], offset=self.DH_Table[4][3]),
                rtb.RevoluteDH(a=self.DH_Table[5][0], alpha=self.DH_Table[5][1], d=self.DH_Table[5][2], offset=self.DH_Table[5][3]),
                rtb.RevoluteDH(a=self.DH_Table[6][0], alpha=self.DH_Table[6][1], d=self.DH_Table[6][2], offset=self.DH_Table[6][3]),

                rtb.PrismaticDH(q=self.DH_Table[7][2]) # Tool - Not to be used as an actual joint
            ],
            name="RPPRRR Manipulator"
        )
        return robot
    

    def Base_Tool_Transforms(self):

        Trasnforms = RPPRRR_Manipulator_Numpy.RPPRRRManipulator()

        T0_2 = Trasnforms.TB_1.dot(Trasnforms.T1_2)
        print(T0_2)
        return T0_2



    

    