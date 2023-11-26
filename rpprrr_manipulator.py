#!/usr/bin/env python
"""
@author: Aidan Crawford
"""

import numpy as np
from roboticstoolbox import DHRobot, RevoluteDH, PrismaticDH, IKSolution
from spatialmath import SE3
import sympy as sy


class RPPRRRManipulator(DHRobot):
    """
    Class to model and interact with RPPRRR Manipulator as defined in 23WSB141 - Introduction To Robotics Coursework
    """

    def __init__(self):
        # Define Parameters in DH Table and specified in FK 'actual dimensions'
        self.L0 = 0.10
        self.L1 = 0.20
        self.L2 = 0.30
        self.L3 = 0.30
        self.L4 = 0.10
        self.L5 = 0.05
        self.THETA1 = 0
        self.THETA4 = 0
        self.THETA5 = 0
        self.THETA6 = 0
        self.D2 = 0.5
        self.D3 = 0

        self.DH_TABLE = np.array(
            [
                [0, 0, self.L0, 0],
                [0, 0, 0, self.THETA1],
                [0, 0, self.D2, 0],
                [np.radians(90), 0, self.D3, 0],
                [0, 0, self.L1, self.THETA4],
                [np.radians(90), self.L2, self.L5, self.THETA5],
                [0, 0, self.L3, self.THETA6],
                [0, 0, self.L4, 0]
            ]
        )

        links = [
            RevoluteDH(# Fake joint to mimic base frame 
                alpha=self.DH_TABLE[0][0], 
                a=self.DH_TABLE[0][1], 
                d=self.DH_TABLE[0][2], 
                offset=self.DH_TABLE[0][3], 
                qlim=np.array([0, 0.1])
                ), 

            RevoluteDH(
                alpha=self.DH_TABLE[1][0], 
                a=self.DH_TABLE[1][1], 
                d=self.DH_TABLE[1][2], 
                offset=self.DH_TABLE[1][3], 
                qlim=np.array([np.radians(-180), np.radians(180)])
                ), 

            PrismaticDH(
                alpha=self.DH_TABLE[2][0], 
                a=self.DH_TABLE[2][1], 
                q=self.DH_TABLE[2][2], 
                offset=self.DH_TABLE[2][3], 
                qlim=np.array([0.0, 0.5])
                ),
            PrismaticDH(
                alpha=self.DH_TABLE[3][0], 
                a=self.DH_TABLE[3][1], 
                q=self.DH_TABLE[3][2], 
                offset=self.DH_TABLE[3][3], 
                qlim=np.array([-0.1, 0.1])
                ),

            RevoluteDH(
                alpha=self.DH_TABLE[4][0],
                a=self.DH_TABLE[4][1], 
                d=self.DH_TABLE[4][2], 
                offset=self.DH_TABLE[4][3], 
                qlim=np.array([np.radians(-90), np.radians(90)])
                ),
            RevoluteDH(
                alpha=self.DH_TABLE[5][0], 
                a=self.DH_TABLE[5][1], 
                d=self.DH_TABLE[5][2], 
                offset=self.DH_TABLE[5][3], 
                qlim=np.array([np.radians(-180), np.radians(180)])
                ),
            RevoluteDH(
                alpha=self.DH_TABLE[6][0], 
                a=self.DH_TABLE[6][1], 
                d=self.DH_TABLE[6][2], 
                offset=self.DH_TABLE[6][3], 
                qlim=np.array([np.radians(-90), np.radians(90)])
                ),

            RevoluteDH( #Fake joint to mimic tool frame
                alpha=self.DH_TABLE[7][0], 
                a=self.DH_TABLE[7][1], 
                d=self.DH_TABLE[7][2], 
                offset=self.DH_TABLE[7][3], qlim=np.array([0, 0])
                ) 
        ]


        super().__init__(
            links, 
            name="RPPRRR Manipulator",
        )

    def forward_kinematics(self, joint_angles: list) -> SE3:
        '''
        Calculates forward kinematics from five (5) joint angles

        :param joint_angles: joint angles in radians
        :type joint_angles: List

        :return FK_Sol: Transformation matrix of end effectors forward kinematics
        :type FK_sol: SE3

        .. note::
            - Angles are in radians
            - Automatically adds a zero to the start and end of `joint_angles` to account for the placeholder joints used in the model to represent the base and end effector
            - If a solution cannot be found None will be returned
        '''
        if type(joint_angles) is not list:
            raise TypeError(f"{type(joint_angles)} is not valid. {list} expected.")

        try:
            if len(joint_angles) == 6:
                joint_angles.insert(0, 0)
                joint_angles.append(0)
            else:
                raise Exception("Incorrect array size. 6 joint angles are required")
        except Exception as e:
            print("An error occured: ", e)

        try: 
            fk_sol = self.fkine(joint_angles)
        except Exception as e:
            print("An error occured while calculating forawrd kinematics: ", e)
            return False

        return fk_sol
    
    def inverse_kinematics(self, transform: np.ndarray, display=False: bool) -> IKSolution:
        '''
        Given a desired positon P and Orientation R, in the form of a SE3 transformation matrix, will return whether the manipulator can reach and the joint angles to do so.

        :param transform: Transformation matrix in a compatiable format
        :type transform: transform matrix
        '''
        try:
            ik_solution = self.ikine_LM(transform)
            if display:
                print(
                    f'''
                    Inverse Kinematic solvable: {ik_solution.success} '''
                )
                if ik_solution.success is not True:
                    print(f'''
                        Closest possible solution: {ik_solution.q.T}  
                        ''')
                else:
                    print(f'''
                        Solution: {ik_solution.q.T}
                    ''')
            return ik_solution
        except Exception as e:
            print("An error occured while calculating inverse kinematics: ", e)
            
    def joint_velcoities(self, joint_angles: list, joint_velocities: list):
        '''
        Given a set of joint angles and velocitiies, will return a jacobian matrix of linear and angular velocities
        '''
        self.jacobian_matrix = self.jacob0(joint_angles)
        
        velocites = self.jacobian_matrix @ joint_velocities
        linear_velocities = velocites[:3]
        angular_velocities = velocites[3:]
        
        return self.jacobian_matrix, linear_velocities, angular_velocities
    
    def static_torques(self, mass, g, transform):
        '''
        Give a point mass load, applied at the origin of the tool frame, will calculate the static force and torque at each joint for a set transform
        '''
        joint_angles = self.inverse_kinematics(transform)
        
        
        

    class RPPRRRManipulatorSympy():
        def __init__(self):
            self.L0 = sy.symbols('L0')
            self.L1 = sy.symbols('L1')
            self.L2 = sy.symbols('L2')
            self.L3 = sy.symbols('L3')
            self.L4 = sy.symbols('L4')
            self.L5 = sy.symbols('L5')
            self.THETA1 = sy.symbols('THETA1')
            self.THETA4 = sy.symbols('THETA4')
            self.THETA5 = sy.symbols('THETA5')
            self.THETA6 = sy.symbols('THETA6')
            self.D2 = sy.symbols('D2')
            self.D3 = sy.symbols('D3')

            self.DH_TABLE = sy.Matrix(
            [   #alpha, A, D, theta
                [0, 0, self.L0, 0],
                [0, 0, 0, self.THETA1],
                [0, 0, self.D2, 0],
                [np.radians(90), 0, self.D3, 0],
                [0, 0, self.L1, self.THETA4],
                [np.radians(90), self.L2, self.L5, self.THETA5],
                [0, 0, self.L3, self.THETA6],
                [0, 0, self.L4, 0]
            ])
            
            
            self.TB_1 = sy.Matrix([
                [sy.cos(self.DH_TABLE[0, 3]), -sy.sin(self.DH_TABLE[0, 3]), 0, self.DH_TABLE[0, 1]],
                [(sy.sin(self.DH_TABLE[0, 3])*sy.cos(self.DH_TABLE[0, 0])), (sy.cos(self.DH_TABLE[0, 3])*sy.cos(self.DH_TABLE[0, 0])), -sy.sin(self.DH_TABLE[0, 0]), (-sy.sin(self.DH_TABLE[0, 0])*self.DH_TABLE[0, 2])],
                [(sy.sin(self.DH_TABLE[0, 3])*sy.sin(self.DH_TABLE[0, 0])), (sy.cos(self.DH_TABLE[0, 3])*sy.sin(self.DH_TABLE[0, 0])), sy.cos(self.DH_TABLE[0, 0]), (sy.cos(self.DH_TABLE[0, 0])*self.DH_TABLE[0, 2])],
                [0, 0, 0, 1]
            ])

            self.T1_2 = sy.Matrix([
                [sy.cos(self.DH_TABLE[1, 3]), -sy.sin(self.DH_TABLE[1, 3]), 0, self.DH_TABLE[1, 1]],
                [(sy.sin(self.DH_TABLE[1, 3])*sy.cos(self.DH_TABLE[1, 0])), (sy.cos(self.DH_TABLE[1, 3])*sy.cos(self.DH_TABLE[1, 0])), -sy.sin(self.DH_TABLE[1, 0]), (-sy.sin(self.DH_TABLE[1, 0])*self.DH_TABLE[1, 2])],
                [(sy.sin(self.DH_TABLE[1, 3])*sy.sin(self.DH_TABLE[1, 0])), (sy.cos(self.DH_TABLE[1, 3])*sy.sin(self.DH_TABLE[1, 0])), sy.cos(self.DH_TABLE[1, 0]), (sy.cos(self.DH_TABLE[1, 0])*self.DH_TABLE[1, 2])],
                [0, 0, 0, 1]
            ])

            self.T2_3 = sy.Matrix([
                [sy.cos(self.DH_TABLE[2, 3]), -sy.sin(self.DH_TABLE[2, 3]), 0, self.DH_TABLE[2, 1]],
                [(sy.sin(self.DH_TABLE[2, 3])*sy.cos(self.DH_TABLE[2, 0])), (sy.cos(self.DH_TABLE[2, 3])*sy.cos(self.DH_TABLE[2, 0])), -sy.sin(self.DH_TABLE[2, 0]), (-sy.sin(self.DH_TABLE[2, 0])*self.DH_TABLE[2, 2])],
                [(sy.sin(self.DH_TABLE[2, 3])*sy.sin(self.DH_TABLE[2, 0])), (sy.cos(self.DH_TABLE[2, 3])*sy.sin(self.DH_TABLE[2, 0])), sy.cos(self.DH_TABLE[2, 0]), (sy.cos(self.DH_TABLE[2, 0])*self.DH_TABLE[2, 2])],
                [0, 0, 0, 1]
            ])

            self.T3_4 = sy.Matrix([
                [sy.cos(self.DH_TABLE[3, 3]), -sy.sin(self.DH_TABLE[3, 3]), 0, self.DH_TABLE[3, 1]],
                [(sy.sin(self.DH_TABLE[3, 3])*sy.cos(self.DH_TABLE[3, 0])), (sy.cos(self.DH_TABLE[3, 3])*sy.cos(self.DH_TABLE[3, 0])), -sy.sin(self.DH_TABLE[3, 0]), (-sy.sin(self.DH_TABLE[3, 0])*self.DH_TABLE[3, 2])],
                [(sy.sin(self.DH_TABLE[3, 3])*sy.sin(self.DH_TABLE[3, 0])), (sy.cos(self.DH_TABLE[3, 3])*sy.sin(self.DH_TABLE[3, 0])), sy.cos(self.DH_TABLE[3, 0]), (sy.cos(self.DH_TABLE[3, 0])*self.DH_TABLE[3, 2])],
                [0, 0, 0, 1]
            ])

            self.T4_5 = sy.Matrix([
                [sy.cos(self.DH_TABLE[4, 3]), -sy.sin(self.DH_TABLE[4, 3]), 0, self.DH_TABLE[4, 1]],
                [(sy.sin(self.DH_TABLE[4, 3])*sy.cos(self.DH_TABLE[4, 0])), (sy.cos(self.DH_TABLE[4, 3])*sy.cos(self.DH_TABLE[4, 0])), -sy.sin(self.DH_TABLE[4, 0]), (-sy.sin(self.DH_TABLE[4, 0])*self.DH_TABLE[4, 2])],
                [(sy.sin(self.DH_TABLE[4, 3])*sy.sin(self.DH_TABLE[4, 0])), (sy.cos(self.DH_TABLE[4, 3])*sy.sin(self.DH_TABLE[4, 0])), sy.cos(self.DH_TABLE[4, 0]), (sy.cos(self.DH_TABLE[4, 0])*self.DH_TABLE[4, 2])],
                [0, 0, 0, 1]
            ])

            self.T5_6 = sy.Matrix([
                [sy.cos(self.DH_TABLE[5, 3]), -sy.sin(self.DH_TABLE[5, 3]), 0, self.DH_TABLE[5, 1]],
                [(sy.sin(self.DH_TABLE[5, 3])*sy.cos(self.DH_TABLE[5, 0])), (sy.cos(self.DH_TABLE[5, 3])*sy.cos(self.DH_TABLE[5, 0])), -sy.sin(self.DH_TABLE[5, 0]), (-sy.sin(self.DH_TABLE[5, 0])*self.DH_TABLE[5, 2])],
                [(sy.sin(self.DH_TABLE[5, 3])*sy.sin(self.DH_TABLE[5, 0])), (sy.cos(self.DH_TABLE[5, 3])*sy.sin(self.DH_TABLE[5, 0])), sy.cos(self.DH_TABLE[5, 0]), (sy.cos(self.DH_TABLE[5, 0])*self.DH_TABLE[5, 2])],
                [0, 0, 0, 1]
            ])

            self.T6_T = sy.Matrix([
                [sy.cos(self.DH_TABLE[6, 3]), -sy.sin(self.DH_TABLE[6, 3]), 0, self.DH_TABLE[6, 1]],
                [(sy.sin(self.DH_TABLE[6, 3])*sy.cos(self.DH_TABLE[6, 0])), (sy.cos(self.DH_TABLE[6, 3])*sy.cos(self.DH_TABLE[6, 0])), -sy.sin(self.DH_TABLE[6, 0]), (-sy.sin(self.DH_TABLE[6, 0])*self.DH_TABLE[6, 2])],
                [(sy.sin(self.DH_TABLE[6, 3])*sy.sin(self.DH_TABLE[6, 0])), (sy.cos(self.DH_TABLE[6, 3])*sy.sin(self.DH_TABLE[6, 0])), sy.cos(self.DH_TABLE[6, 0]), (sy.cos(self.DH_TABLE[6, 0])*self.DH_TABLE[6, 2])],
                [0, 0, 0, 1]
            ])
            
            self.TB_2 = self.TB_1*self.T1_2
            self.TB_3 = self.TB_2*self.T2_3
            self.TB_4 = self.TB_3*self.T3_4
            self.TB_5 = self.TB_4*self.T4_5
            self.TB_6 = self.TB_5*self.T5_6
            self.TB_T = self.TB_6*self.T6_T
            
            print(self.TB_2.subs({self.L0:0.1, self.L1:0.2, self.L2:0.3, self.L3:0.3, self.L4:0.1, self.L5:0.05, self.THETA1:0, self.D2:0.5, self.D3:0, self.THETA4:0, self.THETA5:0, self.THETA6:0}))
            print(self.TB_3.subs({self.L0:0.1, self.L1:0.2, self.L2:0.3, self.L3:0.3, self.L4:0.1, self.L5:0.05, self.THETA1:0, self.D2:0.5, self.D3:0, self.THETA4:0, self.THETA5:0, self.THETA6:0}))
            print(self.TB_4.subs({self.L0:0.1, self.L1:0.2, self.L2:0.3, self.L3:0.3, self.L4:0.1, self.L5:0.05, self.THETA1:0, self.D2:0.5, self.D3:0, self.THETA4:0, self.THETA5:0, self.THETA6:0}))
            print(self.TB_5.subs({self.L0:0.1, self.L1:0.2, self.L2:0.3, self.L3:0.3, self.L4:0.1, self.L5:0.05, self.THETA1:0, self.D2:0.5, self.D3:0, self.THETA4:0, self.THETA5:0, self.THETA6:0}))
            print(self.TB_6.subs({self.L0:0.1, self.L1:0.2, self.L2:0.3, self.L3:0.3, self.L4:0.1, self.L5:0.05, self.THETA1:0, self.D2:0.5, self.D3:0, self.THETA4:0, self.THETA5:0, self.THETA6:0}))
            print(self.TB_T.subs({self.L0:0.1, self.L1:0.2, self.L2:0.3, self.L3:0.3, self.L4:0.1, self.L5:0.05, self.THETA1:0, self.D2:0.5, self.D3:0, self.THETA4:0, self.THETA5:0, self.THETA6:0}))
            
            self.TB_T = self.TB_1*self.T1_2*self.T2_3*self.T3_4*self.T4_5*self.T5_6*self.T6_T
            self.TB_T_FK = self.TB_T.subs({self.L0:0.1, self.L1:0.2, self.L2:0.3, self.L3:0.3, self.L4:0.1, self.L5:0.05, self.THETA1:0, self.D2:0.5, self.D3:0, self.THETA4:0, self.THETA5:0, self.THETA6:0})
            