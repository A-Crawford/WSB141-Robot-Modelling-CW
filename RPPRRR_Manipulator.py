import roboticstoolbox as rtb
import numpy as np

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
            [0, 0, 0, self.theta1],
            [0, 0, self.d2, 0],
            [0, 90, self.d3, 0],
            [0, 0, self.L1, self.theta4],
            [self.L2, 90, self.L5, self.theta5],
            [0, 0, self.L3, self.theta6],
        ]
        return dh_table
    
    def create_RTB_robot(self):
        # R P P R R R 
        robot = rtb.DHRobot(
            [
                rtb.RevoluteDH(d=self.DH_Table[0][0], a=self.DH_Table[0][1], alpha=self.DH_Table[0][2], offset=self.DH_Table[0][3]),
                rtb.PrismaticDH(q=self.DH_Table[1][0], a=self.DH_Table[1][1], alpha=self.DH_Table[1][2], offset=self.DH_Table[1][3]),
                rtb.PrismaticDH(q=self.DH_Table[2][0], a=self.DH_Table[2][1], alpha=self.DH_Table[2][2], offset=self.DH_Table[2][3]),
                rtb.RevoluteDH(d=self.DH_Table[3][0], a=self.DH_Table[3][1], alpha=self.DH_Table[3][2], offset=self.DH_Table[3][3]),
                rtb.RevoluteDH(d=self.DH_Table[4][0], a=self.DH_Table[4][1], alpha=self.DH_Table[4][2], offset=self.DH_Table[4][3]),
                rtb.RevoluteDH(d=self.DH_Table[5][0], a=self.DH_Table[5][1], alpha=self.DH_Table[5][2], offset=self.DH_Table[5][3]),
            ],
            base=[0, 0.1, 0, 0],
            tool=[0, 0.1, 0, 0],
            name="RPPRRR Manipulator"
        )
        print(robot)
        return robot
    
    

    