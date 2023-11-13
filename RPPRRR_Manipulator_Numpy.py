import numpy as np

#Numpy Solution
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

    def create_DH_table(self):
        self.DH_Table = np.array([
            #Alpha, a, d, theta
            [0, 0, self.L1, 0],
            [0, 0, 0, self.theta1],
            [0, 0, self.d2, 0],
            [np.radians(90), 0, self.d3, 0],
            [0, 0, self.L1, self.theta4],
            [np.radians(90), self.L2, self.L5, self.theta5],
            [0, 0, self.L3, self.theta6],
            [0, 0, self.L4, 0]
        ])

    def define_transforms(self):
        self.TB_1 = np.array([
            [np.cos(self.DH_Table[0,3]), -np.sin(self.DH_Table[0, 3]), 0, self.DH_Table[0, 1]],
            [(np.sin(self.DH_Table[0, 3])*np.cos(self.DH_Table[0, 0])), (np.cos(self.DH_Table[0, 3])*np.cos(self.DH_Table[0, 0])), -np.sin(self.DH_Table[0,0]), -(np.sin(self.DH_Table[0, 0])*self.DH_Table[0, 2])],
            [(np.sin(self.DH_Table[0, 3])*np.sin(self.DH_Table[0, 0])), (np.cos(self.DH_Table[0, 3])*np.sin(self.DH_Table[0, 0])), np.cos(self.DH_Table[0, 0]), (np.cos(self.DH_Table[0, 3])*self.DH_Table(0, 2))],
            [0, 0, 0, 1]
        ])

        self.T1_2 = np.array([
            [np.cos(self.DH_Table[1,3]), -np.sin(self.DH_Table[1, 3]), 0, self.DH_Table[1, 1]],
            [(np.sin(self.DH_Table[1, 3])*np.cos(self.DH_Table[1, 0])), (np.cos(self.DH_Table[1, 3])*np.cos(self.DH_Table[1, 0])), -np.sin(self.DH_Table[1,0]), -(np.sin(self.DH_Table[1, 0])*self.DH_Table[1, 2])],
            [(np.sin(self.DH_Table[1, 3])*np.sin(self.DH_Table[1, 0])), (np.cos(self.DH_Table[1, 3])*np.sin(self.DH_Table[1, 0])), np.cos(self.DH_Table[1, 0]), (np.cos(self.DH_Table[1, 3])*self.DH_Table(1, 2))],
            [0, 0, 0, 1]
        ])

        self.T2_3 = np.array([
            [np.cos(self.DH_Table[2,3]), -np.sin(self.DH_Table[2, 3]), 0, self.DH_Table[2, 1]],
            [(np.sin(self.DH_Table[2, 3])*np.cos(self.DH_Table[2, 0])), (np.cos(self.DH_Table[2, 3])*np.cos(self.DH_Table[2, 0])), -np.sin(self.DH_Table[2,0]), -(np.sin(self.DH_Table[2, 0])*self.DH_Table[2, 2])],
            [(np.sin(self.DH_Table[2, 3])*np.sin(self.DH_Table[2, 0])), (np.cos(self.DH_Table[2, 3])*np.sin(self.DH_Table[2, 0])), np.cos(self.DH_Table[2, 0]), (np.cos(self.DH_Table[2, 3])*self.DH_Table(2, 2))],
            [0, 0, 0, 1]
        ])

        self.T3_4 = np.array([
            [np.cos(self.DH_Table[3,3]), -np.sin(self.DH_Table[3, 3]), 0, self.DH_Table[3, 1]],
            [(np.sin(self.DH_Table[3, 3])*np.cos(self.DH_Table[3, 0])), (np.cos(self.DH_Table[3, 3])*np.cos(self.DH_Table[3, 0])), -np.sin(self.DH_Table[3,0]), -(np.sin(self.DH_Table[3, 0])*self.DH_Table[3, 2])],
            [(np.sin(self.DH_Table[3, 3])*np.sin(self.DH_Table[3, 0])), (np.cos(self.DH_Table[3, 3])*np.sin(self.DH_Table[3, 0])), np.cos(self.DH_Table[3, 0]), (np.cos(self.DH_Table[3, 3])*self.DH_Table(3, 2))],
            [0, 0, 0, 1]
        ])

        self.T4_5 = np.array([
            [np.cos(self.DH_Table[4,3]), -np.sin(self.DH_Table[4, 3]), 0, self.DH_Table[4, 1]],
            [(np.sin(self.DH_Table[4, 3])*np.cos(self.DH_Table[4, 0])), (np.cos(self.DH_Table[4, 3])*np.cos(self.DH_Table[4, 0])), -np.sin(self.DH_Table[4,0]), -(np.sin(self.DH_Table[4, 0])*self.DH_Table[4, 2])],
            [(np.sin(self.DH_Table[4, 3])*np.sin(self.DH_Table[4, 0])), (np.cos(self.DH_Table[4, 3])*np.sin(self.DH_Table[4, 0])), np.cos(self.DH_Table[4, 0]), (np.cos(self.DH_Table[4, 3])*self.DH_Table(4, 2))],
            [0, 0, 0, 1]
        ])

        self.T5_6 = np.array([
            [np.cos(self.DH_Table[5,3]), -np.sin(self.DH_Table[5, 3]), 0, self.DH_Table[5, 1]],
            [(np.sin(self.DH_Table[5, 3])*np.cos(self.DH_Table[5, 0])), (np.cos(self.DH_Table[5, 3])*np.cos(self.DH_Table[5, 0])), -np.sin(self.DH_Table[5,0]), -(np.sin(self.DH_Table[5, 0])*self.DH_Table[5, 2])],
            [(np.sin(self.DH_Table[5, 3])*np.sin(self.DH_Table[5, 0])), (np.cos(self.DH_Table[5, 3])*np.sin(self.DH_Table[5, 0])), np.cos(self.DH_Table[5, 0]), (np.cos(self.DH_Table[5, 3])*self.DH_Table(5, 2))],
            [0, 0, 0, 1]
        ])

        self.T6_7 = np.array([
            [np.cos(self.DH_Table[6,3]), -np.sin(self.DH_Table[6, 3]), 0, self.DH_Table[6, 1]],
            [(np.sin(self.DH_Table[6, 3])*np.cos(self.DH_Table[6, 0])), (np.cos(self.DH_Table[6, 3])*np.cos(self.DH_Table[6, 0])), -np.sin(self.DH_Table[6,0]), -(np.sin(self.DH_Table[6, 0])*self.DH_Table[6, 2])],
            [(np.sin(self.DH_Table[6, 3])*np.sin(self.DH_Table[6, 0])), (np.cos(self.DH_Table[6, 3])*np.sin(self.DH_Table[6, 0])), np.cos(self.DH_Table[6, 0]), (np.cos(self.DH_Table[6, 3])*self.DH_Table(6, 2))],
            [0, 0, 0, 1]
        ])

        self.T7_T = np.array([
            [np.cos(self.DH_Table[7,3]), -np.sin(self.DH_Table[7, 3]), 0, self.DH_Table[7, 1]],
            [(np.sin(self.DH_Table[7, 3])*np.cos(self.DH_Table[7, 0])), (np.cos(self.DH_Table[7, 3])*np.cos(self.DH_Table[7, 0])), -np.sin(self.DH_Table[7,0]), -(np.sin(self.DH_Table[7, 0])*self.DH_Table[7, 2])],
            [(np.sin(self.DH_Table[7, 3])*np.sin(self.DH_Table[7, 0])), (np.cos(self.DH_Table[7, 3])*np.sin(self.DH_Table[7, 0])), np.cos(self.DH_Table[7, 0]), (np.cos(self.DH_Table[7, 3])*self.DH_Table(7, 2))],
            [0, 0, 0, 1]
        ])
