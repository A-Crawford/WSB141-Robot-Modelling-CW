import roboticstoolbox as rtb
import numpy as np
import RPPRRR_Manipulator_Numpy
from spatialmath import SE3


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
        Rbase = SE3
        Rbase.Rx = 0
        Rbase.Ry = 0
        Rbase.Rz = 0
        # R P P R R R 
        robot = rtb.DHRobot(
            [
                rtb.RevoluteDH(d=self.DH_Table[0][2], qlim=np.array([0, 0])), # Base - Added for matricies, not an actual joint to be used Qlim = 0 so it wont be used

                rtb.RevoluteDH(),
                rtb.PrismaticDH(q=self.d2, qlim=np.array([0, 0.5])),
                rtb.PrismaticDH(alpha=np.radians(90), q=self.d3, qlim=np.array([-0.1, 0.1])),
                rtb.RevoluteDH(d=self.L1),
                rtb.RevoluteDH(alpha=np.radians(90), a=self.L2, d=self.L5),
                rtb.RevoluteDH(d=self.L3),

                rtb.RevoluteDH(qlim=np.array([0, 0])) # Tool - Not to be used as an actual joint, Qlim is 0 so it wont be used
            ],
            name="RPPRRR Manipulator",
            base=Rbase,
        )
        return robot
    
        
    
