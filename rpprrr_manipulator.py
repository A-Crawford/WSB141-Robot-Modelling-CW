#!/usr/bin/env python
"""
@author: Aidan Crawford
"""

import numpy as np
from roboticstoolbox import DHRobot, RevoluteMDH, PrismaticMDH, IKSolution
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
                [0, self.L2, self.L5, self.THETA5],
                [np.radians(90), 0, self.L3, self.THETA6],
                [0, 0, self.L4, 0]
            ]
        )

        links = [
            RevoluteMDH(# Fake joint to mimic base frame 
                alpha=self.DH_TABLE[0][0], 
                a=self.DH_TABLE[0][1], 
                d=self.DH_TABLE[0][2], 
                offset=self.DH_TABLE[0][3], 
                qlim=np.array([0, 0])
                ), 

            RevoluteMDH(
                alpha=self.DH_TABLE[1][0], 
                a=self.DH_TABLE[1][1], 
                d=self.DH_TABLE[1][2], 
                offset=self.DH_TABLE[1][3], 
                qlim=np.array([np.radians(-180), np.radians(180)])
                ), 

            PrismaticMDH(
                alpha=self.DH_TABLE[2][0], 
                a=self.DH_TABLE[2][1], 
                q=self.DH_TABLE[2][2], 
                offset=self.DH_TABLE[2][3], 
                qlim=np.array([0.0, 0.5])
                ),
            PrismaticMDH(
                alpha=self.DH_TABLE[3][0], 
                a=self.DH_TABLE[3][1], 
                q=self.DH_TABLE[3][2], 
                offset=self.DH_TABLE[3][3], 
                qlim=np.array([-0.1, 0.1])
                ),

            RevoluteMDH(
                alpha=self.DH_TABLE[4][0],
                a=self.DH_TABLE[4][1], 
                d=self.DH_TABLE[4][2], 
                offset=self.DH_TABLE[4][3], 
                qlim=np.array([np.radians(-90), np.radians(90)])
                ),
            RevoluteMDH(
                alpha=self.DH_TABLE[5][0], 
                a=self.DH_TABLE[5][1], 
                d=self.DH_TABLE[5][2], 
                offset=self.DH_TABLE[5][3], 
                qlim=np.array([np.radians(-180), np.radians(180)])
                ),
            RevoluteMDH(
                alpha=self.DH_TABLE[6][0], 
                a=self.DH_TABLE[6][1], 
                d=self.DH_TABLE[6][2], 
                offset=self.DH_TABLE[6][3], 
                qlim=np.array([np.radians(-90), np.radians(90)])
                ),

            RevoluteMDH( #Fake joint to mimic tool frame
                alpha=self.DH_TABLE[7][0], 
                a=self.DH_TABLE[7][1], 
                d=self.DH_TABLE[7][2], 
                offset=self.DH_TABLE[7][3], 
                qlim=np.array([0, 0])
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
        Given a desired positon P and Orientation R, in the form of a SE3 transformation matrix, will return whether the manipulator can reach and all the solutions to do so. Has a 100 maximum search limit.
        If the desired position and orientation are out of reach, method will return the closest solution

        :param transform: Transformation matrix as a numpy ndarray.
        :type transform: transform matrix
        
        :return valid_solutions, invalid solutions: tuple of two lists of soltions which returned Ture and False for valid respectively 
        :type tuple<list<ik_solution>>: IKSolution
        ''' 
        
        print("\nCalculating Inverse Kinematics for the folllowing transformation matrix:\n", transform)
        
        self.transform_type_check(transform)
        self.solutions = []
        for x in range(0, 100):
            ik = self.ikine_LM(transform)
            self.solutions.append(ik)
        
        valid_solutions = []
        invalid_solutions = []
        
        for x in self.solutions:
            if x.success is True:
                valid_solutions.append(x)
            else:
                invalid_solutions.append(x)
                
        if len(valid_solutions) > len(invalid_solutions):          
            if len(valid_solutions) > 99:
                print("\nOver 99 Kinematic Solutions found.")
            else:
                print(f"{len(valid_solutions)} Inverse Kinematic Solutions found.")
                    
            ans = input("\nDo you wish to display all solutions? Y/N ").upper()
            if ans == 'Y':
                for x in valid_solutions:
                    print(x)
        else:
            print("\nInverse Kinematic solution not possible.")
            print(f'\nCloest solution: {invalid_solutions[0]}')
                    
        return valid_solutions, invalid_solutions
    
    def step3_inverse_kinematic_solver(self, transform):
        '''
        Given a transformation matrix, will calculate the inverse kinematic solutions. 
        Will calculate the error in the solution regardless of if a valid soltuion is found or not
        
        :param transform: Transformation matrix for the inverse kinematic solution to be found
        :type ndarray: Numpy ndarray
        '''
        index = 1
        ik_sol_1 = self.inverse_kinematics(transform, display=True)
        if self.compare_len(ik_sol_1[0], ik_sol_1[1]):
            index = 0
        
        lowest_error = 100
        best_sol = IKSolution
        for x in ik_sol_1[index]:
            ik_sol_1_error = self.ik_error(transform, x)
            if ik_sol_1_error < lowest_error:
                lowest_error = ik_sol_1_error
                best_sol = x
                
        print(f"\nBest IK solution {best_sol.q} with the lowest error of: {lowest_error}")

        if input(f"\nDisplay plot for best IK solution for transform: \n{transform}? \nY/N ").upper() == 'Y':
            self.plot(best_sol.q, block=True)
        
        return best_sol, ik_sol_1_error
    
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
        index = 1
        ik_sol_1 = self.inverse_kinematics(transform, display=True)
        if self.compare_len(ik_sol_1[0], ik_sol_1[1]):
            index = 0
        
        lowest_error = 100
        best_sol = IKSolution
        for x in ik_sol_1[index]:
            ik_sol_1_error = self.ik_error(transform, x)
            if ik_sol_1_error < lowest_error:
                lowest_error = ik_sol_1_error
                best_sol = x
        joint_angles = best_sol.q
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
        
        
    def compare_len(self, list1, list2):
        '''
        Given two lists, will return True if the first contains more items than the second
        
        :param list1: List which second list will be compared to
        :type list
        :param list2: list which will be compared to first
        :type list
        
        :return bool: True if list 1 is greater than list 2
        :type Bool
        '''
        return len(list1) > len(list2)
    
    

    class RPPRRRManipulatorSympy():
        def __init__(self):
            # Delcare Symbols for DH table abd transformation
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
            
            # Declare DH Table
            self.DH_TABLE = sy.Matrix(
                [
                    [0, 0, self.L0, 0],
                    [0, 0, 0, self.THETA1],
                    [0, 0, self.D2, 0],
                    [np.radians(90), 0, self.D3, 0],
                    [0, 0, self.L1, self.THETA4],
                    [0, self.L2, self.L5, self.THETA5],
                    [np.radians(90), 0, self.L3, self.THETA6],
                    [0, 0, self.L4, 0]
                ]
            )
            
            # Declare Transformation for each link
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
            
            # Calculate transformation from B to T
            self.TB_T = self.TB_1*self.T1_2*self.T2_3*self.T3_4*self.T4_5*self.T5_6*self.T6_T*self.TT_T
            self.T1_6 = self.T1_2*self.T2_3*self.T3_4*self.T4_5*self.T5_6*self.T6_T
            self.TB_T_FK = self.TB_T.subs({ # Substitude in FK joint values
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
            
            # Declare Centre of mass values
            self.PCB_B = np.array([0, 0, 0.05])
            self.PC1_1 = np.array([0, 0, 0.3])
            self.PC2_2 = np.array([0.0526, -0.0001526, 0.4998])
            self.PC3_3 = np.array([0.0526, -0.0001526, 0.4998])
            self.PC4_4 = np.array([0.2203, 0.1271, 0.4761])
            self.PC5_5 = np.array([0.2208, 0.2812, 0.2578])
            self.PC6_6 = np.array([0.2207, 0.2671, 0.0583])
            
            #  Link Masses
            self.LINK_MASSES = np.array([6.16, 9.81, 4.767, 4.676, 3.7632, 1.960, 0.147])
            self.M0 = self.LINK_MASSES[0] #Base
            self.M1 = self.LINK_MASSES[1] #Link 1
            self.M2 = self.LINK_MASSES[2] #Link 2
            self.M3 = self.LINK_MASSES[3] #Link 3
            self.M4 = self.LINK_MASSES[4] #Link 4
            self.M5 = self.LINK_MASSES[5] #Link 5
            self.M6 = self.LINK_MASSES[6] #Link 6
            self.M7 = 0 #Link 7
            
            # Inertia Tensors
            self.I_CB_B = np.array([
                [0.0244, 0, 0],
                [0, 0.0244, 0],
                [0, 0, 0.0077]
            ])
            
            self.I_C1_1 = np.array([
                [1.088, 0, 0],
                [0, 1.0882, 0],
                [0, 0, 0.004]
            ])
            
            self.I_C2_2 = np.array([
                [1.1932, 0.0009, -0.1254],
                [0.0009, 1.2268, 0.0003],
                [-0.1254, 0.0003, 0.0357]
            ])
            
            self.I_C3_3 = np.array([
                [1.1932, 0.0009, -0.1254],
                [0.0009, 1.2268, 0.0003],
                [-0.1254, 0.0003, 0.0357]
            ])
            
            self.I_C4_4 = np.array([
                [0.9429, -0.1055, -0.3949],
                [-0.1055, 1.0380, -0.2229],
                [-0.3949, -0.2229, 0.2714]
            ])
            
            self.I_C5_5 = np.array([
                [0.3116, -0.1217, -0.1116],
                [-0.1217, 0.2520, -0.1440],
                [-0.1116, -0.1440, 0.2509]
            ])
            
            self.I_C6_6 = np.array([
                [0.0110, -0.0087, -0.0019],
                [-0.0087, 0.0077, -0.0023],
                [-0.0019 ,-0.0023, 0.00177]
            ])
            
            # Intial Pose at time of emergancy stop
            self.INIT_POSE = [np.radians(45), 0.5, 0.09, np.radians(90), np.radians(45), np.radians(90)]
            
            # Joint Velocities at time of emergancy stop
            self.E_STOP_JOINT_VELOCITIES = [-35, 0.1, -0.01, -60, 50, 40]
            
            # 
            self.DEG = np.pi / 180
            
            # Emergency Stop speeds
            self.theta1_i_dot = self.E_STOP_JOINT_VELOCITIES[0] * self.DEG
            self.d2_i_dot = self.E_STOP_JOINT_VELOCITIES[1]
            self.d3_i_dot = self.E_STOP_JOINT_VELOCITIES[2]
            self.theta4_i_dot = self.E_STOP_JOINT_VELOCITIES[3] * self.DEG
            self.theta5_i_dot = self.E_STOP_JOINT_VELOCITIES[4] * self.DEG
            self.theta6_i_dot = self.E_STOP_JOINT_VELOCITIES[5] * self.DEG
            
            # Max Velocities
            self.T1_MAX_VEL = 90
            self.T2_MAX_VEL = 0.90
            self.T3_MAX_VEL = 0.65
            self.T4_MAX_VEL = self.T5_MAX_VEL = self.T6_MAX_VEL = 60
            
            # Max Torques
            self.T1_MAX_TORQUE = 100
            self.T2_MAX_TORQUE = self.T3_MAX_TORQUE = 80
            self.T4_MAX_TORQUE = 40
            self.T5_MAX_TORQUE = 20
            self.T6_MAX_TORQUE = 10
            
            # Mass
            self.MASS = 0.2 #Kg
            
            #Gravity
            self.G = 9.8
        
            #Force at Last Frame
            self.F6_6 = np.array([
                [0],
                [-self.MASS * self.G],
                [0]
            ])
            
            # Moment at Last Frame  
            self.N6_6 = np.array([
                [0],
                [0],
                [0]
            ])
            
            # Acceleration
            self.theta1_2dot = -self.T1_MAX_TORQUE / self.I_C1_1[2, 2]
            self.d2_2dot = -self.T2_MAX_TORQUE / self.I_C2_2[2, 2]
            self.d3_2dot = -self.T3_MAX_TORQUE / self.I_C3_3[2, 2]
            self.theta4_2dot = -self.T4_MAX_TORQUE / self.I_C4_4[2, 2]
            self.theta5_2dot = -self.T5_MAX_TORQUE / self.I_C5_5[2, 2]
            self.theta6_2dot = -self.T6_MAX_TORQUE / self.I_C6_6[2, 2]
            
            # Init angular accerlation at base
            self.omega_0_0 = np.matrix([[0], [0], [0]])
            self.omega_dot_0_0 = np.matrix([[0], [0], [0]])
            
            #Init Linear acceleration at base, given gravity
            v_dot_0_0 = np.matrix([[0], [self.G], [0]])
            
            
        def acc_revolute(self, transform, omega, theta_i_dot, omega_dot, theta_i_2dot, v_dot, PC, mass, I):
            rotation = transform[:3, :3]
            position = transform[:3, 3]
            
            #Velocity
            omega_new = rotation.transpose() @ omega + np.matrix([0, 0, theta_i_dot]).reshape(3, 1)
            
            #Angular Acceleration
            omega_dot_new = rotation.transpose() @ (omega_dot + np.transpose(np.cross(omega.transpose(),np.matrix([0,0,theta_i_dot]))) ) + np.matrix([[0],[0],[theta_i_2dot]])
            v_dot_new = rotation.transpose() @ (np.transpose(np.cross(omega_dot.transpose(),position.transpose())) + np.transpose(np.cross(omega.transpose(),np.cross(omega.transpose(),position.transpose())))+v_dot)
            
            # Adjusting for gravity
            v_centre_dot_new = np.cross(omega_dot_new.transpose(), PC.transpose()) + np.cross(omega_new.transpose(),np.cross(omega_new.transpose(),PC.transpose())) + v_dot_new.transpose()
            v_centre_dot_new = v_centre_dot_new.transpose()
            
            # force action to the centre of mass
            F_new = mass * v_centre_dot_new
            
            # torque action to the centre of mass
            N_new = I @ omega_dot_new + np.transpose(np.cross(omega_new.transpose(),np.transpose(I @ omega_new)))
            
            return omega_new, omega_dot_new, v_dot_new, v_centre_dot_new, F_new, N_new
        
        def acc_prismatic(self, transform, omega, d_i_dot, omega_dot, d_i_2dot, v_dot, PC, mass, I):
            rotation = transform[:3, :3]
            position = transform[:3, 3]
            
            # Velocity
            omega_new = rotation.transpose() @ omega
            
            # Angular Acceleration
            omega_dot_new = rotation.transpose() @ omega
            v_dot_new = rotation.transpose() @ ((np.cross(omega_dot, position.transpose()) + np.cross(omega, (np.cross(omega, position.transpose()))) + v_dot)) + np.cross((2 * omega), (np.matrix([0, 0, d_i_dot]) + np.matrix([0], [0], [d_i_2dot]))) 
        
            # Adjusting for gravity
            v_centre_dot_new = np.cross(omega_dot_new.transpose(), PC.transpose()) + np.cross(omega_new.transpose(),np.cross(omega_new.transpose(),PC.transpose())) + v_dot_new.transpose()
            v_centre_dot_new = v_centre_dot_new.transpose()
            
            # Force action at centre of mass
            F_new = mass * v_centre_dot_new

            # torque action at centre of mass
            N_new = I @ omega_dot_new + np.transpose(np.cross(omega_new.transpose(), np.transpose(I @ omega_new)))
            
            return omega_new, omega_dot_new, v_dot_new, v_centre_dot_new, F_new, N_new
        
        def forces(self, transform, N, F, n, f, PC):
            rotation = transform[:3, :3]
            position = transform[:3, 3]
            
            link_torque = N + rotation @ n + np.transpose(np.cross(PC.transpose(),F.transpose()) + np.cross(position.transpose(),np.transpose(rotation @ f)))
            
            link_force = rotation @ f + F
            
            tau = n.transpose() * np.matrix([0], [0], [1])
            
            return link_force, link_torque, tau
