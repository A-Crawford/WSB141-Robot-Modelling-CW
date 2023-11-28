#!/usr/bin/env python
"""
@author: Aidan Crawford
"""

import numpy as np
from roboticstoolbox import SerialLink, RevoluteDH, PrismaticDH, IKSolution
from spatialmath import SE3
import sympy as sy


class RPPRRRManipulator(SerialLink):
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
            PrismaticDH(# Fake joint to mimic base frame 
                alpha=self.DH_TABLE[0][0], 
                a=self.DH_TABLE[0][1], 
                q=self.DH_TABLE[0][2], 
                offset=self.DH_TABLE[0][3], 
                qlim=np.array([0.1, 0.1])
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

            PrismaticDH( #Fake joint to mimic tool frame
                alpha=self.DH_TABLE[7][0], 
                a=self.DH_TABLE[7][1], 
                q=self.DH_TABLE[7][2], 
                offset=self.DH_TABLE[7][3], 
                qlim=np.array([0.1, 0.1])
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
        
        self.joint_type_check(joint_angles, list)
        joint_angles = self.correct_list_size(joint_angles)

        try: 
            fk_sol = self.fkine(joint_angles)
        except Exception as e:
            print("An error occured while calculating forawrd kinematics: ", e)
            return False

        return fk_sol
    
    def inverse_kinematics(self, transform: np.ndarray, display=False) -> IKSolution:
        '''
        Given a desired positon P and Orientation R, in the form of a SE3 transformation matrix, will return whether the manipulator can reach and the joint angles to do so.
        If the desired position and orientation are out of reach, method will return the closest solution

        :param transform: Transformation matrix as a numpy ndarray.
        :type transform: transform matrix
        
        :return ik_solution: IKSolution containing success, joint values, iterations and reason
        :type ik_solution: IKSolution
        '''
        return self.ikine_LM(transform)
        #self.transform_type_check(transform)
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
            
    def joint_velocities(self, joint_angles: list, joint_velocities: list):
        '''
        Given a set of joint angles and velocitiies, will return a jacobian matrix of linear and angular velocities
        
        :param joint_angles: list of joint angles at which the joint velocities will be calculated at
        :type joint_angles: list
        :param joint_velocities: list of joint velocities of which the tool velcoity can then be calculated with
        :type joint_velocities: list
        
        :return jacobian_matrix: Jacobian matrix operator
        :type jacobian_matrix: numpy.ndarray
        :return linear_velocities: Calculated linear velocities
        :type linear_velocities: list
        :return angular velocities: Calculated angular velocities
        :type angular_velocities: list
        '''

        # Type checking for joing_angles
        self.joint_type_check(joint_angles, list) # Will raise an exception if incorrect type
        joint_angles = self.correct_list_size(joint_angles) #Check list has sufficient joint variables, add 0s to start/end for 'fake' joints if not already present
        
        #Type checking for joint_velocities
        self.joint_type_check(joint_velocities, list) # Will raise an exception if incorrect type
        joint_velocities = self.correct_list_size(joint_velocities) #Check list has sufficient joint variables, add 0s to start/end for 'fake' joints if not already present
        
        jacobian_matrix = self.jacob0(joint_angles)
        velocites = jacobian_matrix @ joint_velocities
        linear_velocities = velocites[:3]
        angular_velocities = velocites[3:]
        
        return jacobian_matrix, linear_velocities, angular_velocities
    
    def static_torques(self, mass, g, transform):
        '''
        Given a point mass load, applied at the origin of the tool frame, will calculate the static force and torque at each joint for a set transform
        
        :param mass: mass applied at origin of tool frame
        :type mass: float
        :param g: force of gravity in m/s
        :type g: float
        :param transform: transform for static pose in which torques are to be calculated at
        :type transform: SE3 or numpy.ndarray
        
        :return torques: caculated torque values for each joint
        :type torques: list
        '''
        #Type cheeck transform
        self.transform_type_check(transform)
        
        # Calculate joint angles from transform using inverse_kinematics
        # Joint angles used to declare wrench variable 
        joint_angles = self.inverse_kinematics(transform).q
        jacobian_matrix = self.jacob0(joint_angles) # RBT function to compute jacobian operator
        wrench = np.array([0, mass*g, 0, 0, 0, 0]).T # X, Y, Z, Rx, Ry, Rz. Only gravity acting on the Y axis
        torques = self.pay(wrench, joint_angles, jacobian_matrix, 0) #RBT function to calculate torques from payload
        return torques # return calculated values
    
    def transform_type_check(self, var):
        '''
        Given a transform, will return a bool as to whether it is a valid type#
        
        :param var: The variable to check validity of type
        :type var: Any valid transform type, ndarray, SE3, etc.
        
        :return bool: validation of type
        '''
        
        if isinstance(var, np.ndarray) or isinstance(var, SE3):
            return var
        else:
            raise TypeError(f"{type(var)} is not valid. {np.ndarray}, or, {type(SE3)} expected.")
            return False
    
    def joint_type_check(self, var, output, alt_output=None):
        '''
        Given a var and desired type, will return Bool whether type is correct. ALternative output can be used when more than one type is accetable, eg. SE3 and ndarray
        
        :param var: The variable to check type of
        :type var: Any
        
        :param output: The desired type to check against
        :type output: Any
        
        :return bool: Boolean of whether type is correct
        '''
        
        if not isinstance(var, output):
            raise TypeError(f"{type(var)} is not valid. {output} expected.")
        else:
            return True
        
    def correct_list_size(self, input_list):
        '''
        Given a list of size six (6), will return an amended list with a leading and following 0 from the original list
        
        :param input_list: A list to correct to size eight (8) from six
        :type input_list: List
        
        :return input_list: input_list with a zero added to the start and end
        :type input_list: list
        '''
        
        try:
            if len(input_list) == 6:
                input_list.insert(0, 0)
                input_list.append(0)
            else:
                raise Exception("Incorrect array size. 6 joint angles are required")
        except Exception as e:
            print("An error occured: ", e)
        return input_list
        
    def ik_error(self, transform: SE3, ik_solution: IKSolution):
        '''
        Given a transform and IK Solution, will return the error
        
        :param transform: The transform the IK solution is based on
        :type transform: SE3 and numpy.ndarray
        
        :return ik_error: numerical error
        :type ik_error: float
        '''
        
        self.transform_type_check(transform)
        self.joint_type_check(ik_solution, IKSolution)
        ik_error = np.linalg.norm(transform - self.fkine(ik_solution.q))
        return ik_error
        
    

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

            # self.DH_TABLE = sy.Matrix(
            # [   #alpha, A, D, theta
            #     [0, 0, self.L0, 0],
            #     [0, 0, 0, self.THETA1],
            #     [0, 0, self.D2, 0],
            #     [np.radians(90), 0, self.D3, 0],
            #     [0, 0, self.L1, self.THETA4],
            #     [np.radians(90), self.L2, self.L5, self.THETA5],
            #     [0, 0, self.L3, self.THETA6],
            #     [0, 0, self.L4, 0]
            # ])
            
            self.DH_TABLE = sy.Matrix(
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
            
            self.TT_T = sy.Matrix([
                [sy.cos(self.DH_TABLE[7, 3]), -sy.sin(self.DH_TABLE[7, 3]), 0, self.DH_TABLE[7, 1]],
                [(sy.sin(self.DH_TABLE[7, 3])*sy.cos(self.DH_TABLE[7, 0])), (sy.cos(self.DH_TABLE[7, 3])*sy.cos(self.DH_TABLE[7, 0])), -sy.sin(self.DH_TABLE[7, 0]), (-sy.sin(self.DH_TABLE[7, 0])*self.DH_TABLE[7, 2])],
                [(sy.sin(self.DH_TABLE[7, 3])*sy.sin(self.DH_TABLE[7, 0])), (sy.cos(self.DH_TABLE[7, 3])*sy.sin(self.DH_TABLE[7, 0])), sy.cos(self.DH_TABLE[7, 0]), (sy.cos(self.DH_TABLE[7, 0])*self.DH_TABLE[7, 2])],
                [0, 0, 0, 1]
            ])
            
            
            self.TB_T = self.TB_1*self.T1_2*self.T2_3*self.T3_4*self.T4_5*self.T5_6*self.T6_T*self.TT_T
            self.TB_T_FK = self.TB_T.subs({
                self.L0: 0.10,
                self.L1: 0.20,
                self.L2: 0.30,
                self.L3: 0.30,
                self.L4: 0.10,
                self.L5: 0.05,
                self.THETA1: 0,
                self.D2: 0.5,
                self.D3: 0,
                self.THETA4: 0,
                self.THETA5: 0,
                self.THETA6: 0
            })
            
            
            