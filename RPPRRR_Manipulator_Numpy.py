import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

        self.create_DH_table()
        self.define_transforms()
        self.calculate_transform()

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
            [(np.sin(self.DH_Table[0, 3])*np.sin(self.DH_Table[0, 0])), (np.cos(self.DH_Table[0, 3])*np.sin(self.DH_Table[0, 0])), np.cos(self.DH_Table[0, 0]), (np.cos(self.DH_Table[0, 3])*self.DH_Table[0, 2])],
            [0, 0, 0, 1]
        ])

        self.T1_2 = np.array([
            [np.cos(self.DH_Table[1,3]), -np.sin(self.DH_Table[1, 3]), 0, self.DH_Table[1, 1]],
            [(np.sin(self.DH_Table[1, 3])*np.cos(self.DH_Table[1, 0])), (np.cos(self.DH_Table[1, 3])*np.cos(self.DH_Table[1, 0])), -np.sin(self.DH_Table[1,0]), -(np.sin(self.DH_Table[1, 0])*self.DH_Table[1, 2])],
            [(np.sin(self.DH_Table[1, 3])*np.sin(self.DH_Table[1, 0])), (np.cos(self.DH_Table[1, 3])*np.sin(self.DH_Table[1, 0])), np.cos(self.DH_Table[1, 0]), (np.cos(self.DH_Table[1, 3])*self.DH_Table[1, 2])],
            [0, 0, 0, 1]
        ])

        self.T2_3 = np.array([
            [np.cos(self.DH_Table[2,3]), -np.sin(self.DH_Table[2, 3]), 0, self.DH_Table[2, 1]],
            [(np.sin(self.DH_Table[2, 3])*np.cos(self.DH_Table[2, 0])), (np.cos(self.DH_Table[2, 3])*np.cos(self.DH_Table[2, 0])), -np.sin(self.DH_Table[2,0]), -(np.sin(self.DH_Table[2, 0])*self.DH_Table[2, 2])],
            [(np.sin(self.DH_Table[2, 3])*np.sin(self.DH_Table[2, 0])), (np.cos(self.DH_Table[2, 3])*np.sin(self.DH_Table[2, 0])), np.cos(self.DH_Table[2, 0]), (np.cos(self.DH_Table[2, 3])*self.DH_Table[2, 2])],
            [0, 0, 0, 1]
        ])

        self.T3_4 = np.array([
            [np.cos(self.DH_Table[3,3]), -np.sin(self.DH_Table[3, 3]), 0, self.DH_Table[3, 1]],
            [(np.sin(self.DH_Table[3, 3])*np.cos(self.DH_Table[3, 0])), (np.cos(self.DH_Table[3, 3])*np.cos(self.DH_Table[3, 0])), -np.sin(self.DH_Table[3,0]), -(np.sin(self.DH_Table[3, 0])*self.DH_Table[3, 2])],
            [(np.sin(self.DH_Table[3, 3])*np.sin(self.DH_Table[3, 0])), (np.cos(self.DH_Table[3, 3])*np.sin(self.DH_Table[3, 0])), np.cos(self.DH_Table[3, 0]), (np.cos(self.DH_Table[3, 3])*self.DH_Table[3, 2])],
            [0, 0, 0, 1]
        ])

        self.T4_5 = np.array([
            [np.cos(self.DH_Table[4,3]), -np.sin(self.DH_Table[4, 3]), 0, self.DH_Table[4, 1]],
            [(np.sin(self.DH_Table[4, 3])*np.cos(self.DH_Table[4, 0])), (np.cos(self.DH_Table[4, 3])*np.cos(self.DH_Table[4, 0])), -np.sin(self.DH_Table[4,0]), -(np.sin(self.DH_Table[4, 0])*self.DH_Table[4, 2])],
            [(np.sin(self.DH_Table[4, 3])*np.sin(self.DH_Table[4, 0])), (np.cos(self.DH_Table[4, 3])*np.sin(self.DH_Table[4, 0])), np.cos(self.DH_Table[4, 0]), (np.cos(self.DH_Table[4, 3])*self.DH_Table[4, 2])],
            [0, 0, 0, 1]
        ])

        self.T5_6 = np.array([
            [np.cos(self.DH_Table[5,3]), -np.sin(self.DH_Table[5, 3]), 0, self.DH_Table[5, 1]],
            [(np.sin(self.DH_Table[5, 3])*np.cos(self.DH_Table[5, 0])), (np.cos(self.DH_Table[5, 3])*np.cos(self.DH_Table[5, 0])), -np.sin(self.DH_Table[5,0]), -(np.sin(self.DH_Table[5, 0])*self.DH_Table[5, 2])],
            [(np.sin(self.DH_Table[5, 3])*np.sin(self.DH_Table[5, 0])), (np.cos(self.DH_Table[5, 3])*np.sin(self.DH_Table[5, 0])), np.cos(self.DH_Table[5, 0]), (np.cos(self.DH_Table[5, 3])*self.DH_Table[5, 2])],
            [0, 0, 0, 1]
        ])

        self.T6_7 = np.array([
            [np.cos(self.DH_Table[6,3]), -np.sin(self.DH_Table[6, 3]), 0, self.DH_Table[6, 1]],
            [(np.sin(self.DH_Table[6, 3])*np.cos(self.DH_Table[6, 0])), (np.cos(self.DH_Table[6, 3])*np.cos(self.DH_Table[6, 0])), -np.sin(self.DH_Table[6,0]), -(np.sin(self.DH_Table[6, 0])*self.DH_Table[6, 2])],
            [(np.sin(self.DH_Table[6, 3])*np.sin(self.DH_Table[6, 0])), (np.cos(self.DH_Table[6, 3])*np.sin(self.DH_Table[6, 0])), np.cos(self.DH_Table[6, 0]), (np.cos(self.DH_Table[6, 3])*self.DH_Table[6, 2])],
            [0, 0, 0, 1]
        ])

        self.T7_T = np.array([
            [np.cos(self.DH_Table[7,3]), -np.sin(self.DH_Table[7, 3]), 0, self.DH_Table[7, 1]],
            [(np.sin(self.DH_Table[7, 3])*np.cos(self.DH_Table[7, 0])), (np.cos(self.DH_Table[7, 3])*np.cos(self.DH_Table[7, 0])), -np.sin(self.DH_Table[7,0]), -(np.sin(self.DH_Table[7, 0])*self.DH_Table[7, 2])],
            [(np.sin(self.DH_Table[7, 3])*np.sin(self.DH_Table[7, 0])), (np.cos(self.DH_Table[7, 3])*np.sin(self.DH_Table[7, 0])), np.cos(self.DH_Table[7, 0]), (np.cos(self.DH_Table[7, 3])*self.DH_Table[7, 2])],
            [0, 0, 0, 1]
        ])

        return self.TB_1, self.T1_2, self.T2_3, self.T3_4, self.T4_5, self.T5_6, self.T6_7, self.T7_T

    def calculate_transform(self):
        self.TBase_Tool = self.TB_1.dot(self.T1_2).dot(self.T2_3).dot(self.T3_4).dot(self.T4_5).dot(self.T5_6).dot(self.T6_7).dot(self.T7_T)

        return self.TBase_Tool
    
    def plot_frames(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim([-1,1])
        ax.set_ylim([-1,1])
        ax.set_zlim([-1,1])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')


        origin = np.zeros((4,4))
        origin_R = origin[:3, :3]
        translation_01 = self.TB_1[:3, 3]
        rotation_01 = self.TB_1[:3, :3].astype(float)

        #Plot Origin
        ax.quiver(origin[:3,0], origin[:3,1], origin[:3,2], origin_R[0],origin_R[1],origin_R[2],color='k')
        ax.quiver(translation_01[0], translation_01[1], translation_01[2], rotation_01[0, 0], rotation_01[1, 0], rotation_01[2,0], color='r')
        ax.quiver(translation_01[0], translation_01[1], translation_01[2], rotation_01[0, 1], rotation_01[1, 1], rotation_01[2,1], color='g')
        ax.quiver(translation_01[0], translation_01[1], translation_01[2], rotation_01[0, 2], rotation_01[1, 2], rotation_01[2, 2], color='b')
        ax.text(translation_01[0], translation_01[1], translation_01[2], "TB_1", color='k', fontsize=7)

        TB_2 = self.TB_1.dot(self.T1_2)
        translation_12 = TB_2[:3, 3]
        roation_12 = TB_2[:3, :3].astype(float)
        ax.quiver(translation_12[0], translation_12[1], translation_12[2], roation_12[0, 0], roation_12[1, 0], roation_12[2,0], color='r')
        ax.quiver(translation_12[0], translation_12[1], translation_12[2], roation_12[0, 1], roation_12[1, 1], roation_12[2,1], color='g')
        ax.quiver(translation_12[0], translation_12[1], translation_12[2], roation_12[0, 2], roation_12[1, 2], roation_12[2, 2], color='b')
        ax.text(translation_12[0], translation_12[1], translation_12[2], "T1_2", color='k', fontsize=7)

        TB_3 = TB_2.dot(self.T2_3)
        translation_23 = TB_3[:3, 3]
        roation_23 = TB_3[:3, :3].astype(float)
        ax.quiver(translation_23[0], translation_23[1], translation_23[2], roation_23[0, 0], roation_23[1, 0], roation_23[2,0], color='r')
        ax.quiver(translation_23[0], translation_23[1], translation_23[2], roation_23[0, 1], roation_23[1, 1], roation_23[2,1], color='g')
        ax.quiver(translation_23[0], translation_23[1], translation_23[2], roation_23[0, 2], roation_23[1, 2], roation_23[2, 2], color='b')
        ax.text(translation_23[0], translation_23[1], translation_23[2], "T2_3", color='k', fontsize=7)

        TB_4 = TB_3.dot(self.T3_4)
        translation_34 = TB_4[:3, 3]
        roation_34 = TB_4[:3, :3].astype(float)
        ax.quiver(translation_34[0], translation_34[1], translation_34[2], roation_34[0, 0], roation_34[1, 0], roation_34[2,0], color='r')
        ax.quiver(translation_34[0], translation_34[1], translation_34[2], roation_34[0, 1], roation_34[1, 1], roation_34[2,1], color='g')
        ax.quiver(translation_34[0], translation_34[1], translation_34[2], roation_34[0, 2], roation_34[1, 2], roation_34[2, 2], color='b')
        ax.text(translation_34[0], translation_34[1], translation_34[2], "T3_4", color='k', fontsize=7)

        TB_5 = TB_4.dot(self.T4_5)
        translation_45 = TB_5[:3, 3]
        roation_45 = TB_5[:3, :3].astype(float)
        ax.quiver(translation_45[0], translation_45[1], translation_45[2], roation_45[0, 0], roation_45[1, 0], roation_45[2,0], color='r')
        ax.quiver(translation_45[0], translation_45[1], translation_45[2], roation_45[0, 1], roation_45[1, 1], roation_45[2,1], color='g')
        ax.quiver(translation_45[0], translation_45[1], translation_45[2], roation_45[0, 2], roation_45[1, 2], roation_45[2, 2], color='b')
        ax.text(translation_45[0], translation_45[1], translation_45[2], "T4_5", color='k', fontsize=7)

        TB_6 = TB_5.dot(self.T5_6)
        translation_56= TB_6[:3, 3]
        roation_56 = TB_6[:3, :3].astype(float)
        ax.quiver(translation_56[0], translation_56[1], translation_56[2], roation_56[0, 0], roation_56[1, 0], roation_56[2,0], color='r')
        ax.quiver(translation_56[0], translation_56[1], translation_56[2], roation_56[0, 1], roation_56[1, 1], roation_56[2,1], color='g')
        ax.quiver(translation_56[0], translation_56[1], translation_56[2], roation_56[0, 2], roation_56[1, 2], roation_56[2, 2], color='b')
        ax.text(translation_56[0], translation_56[1], translation_56[2], "T5_6", color='k', fontsize=7)


        plt.show()



