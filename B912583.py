#!/usr/bin/env python
'''
@Author Aidan Crawford
23WSB141 Introduction To Robotics Coursework
Due Week 11, Semester 1 - W/C 11/12/23

Dependencies:
    Python Ver. 3.8.10 64 Bit 
    Robotics Toolbox for Python
    Matplotlib
    Spatialmath
    Numpy
    Sympy
    Pandas
'''

# Import libraries
from roboticstoolbox import DHRobot, RevoluteMDH, PrismaticMDH, IKSolution
import roboticstoolbox as rbt
import matplotlib.pyplot as plt
from spatialmath import SE3
import numpy as np
import sympy as sy
import pandas as pd

# Define global constants - Defined in various questions of the coursework
STEP1_FK_JOINT_ANGLES = [0, 0.5, 0, 0, 0, 0]

STEP2_IK_TRANSFORM_1 = np.array(
    [
        [1, 0, 0, -0.3],
        [0, -1, 0, -0.25],
        [0, 0, -1, 0.2],
        [0, 0, 0, 1]
    ]
)

STEP2_IK_TRANSFORM_2 = np.array(
    [
        [0.8419, 0, 0.5396, 0.7684],
        [0, -1, 0, -0.25],
        [0.5396, 0, -0.8419, 1.925],
        [0, 0, 0, 1]
    ]
)

STEP2_IK_TRANSFORM_3 = np.array(
    [
        [0, -1, 0, -0.25],
        [0.9, 0, -0.2588, 0.1863],
        [0.2588, 0, 0.9659, 1.064],
        [0, 0, 0, 1]
    ]
)

STEP3_JOINT_ANGLES = [-np.radians(60), 0.4, 0.1, np.radians(90), np.radians(180), np.radians(90)]

STEP3_JOINT_VELOCITIES = [15, 0, 0.1, -30, 15, 10]

STEP3_STATIC_FORCE_TRANSFORM = np.array(
    [
        [0.6791, -0.6403, 0.359, -0.4475],
        [-0.6403, 0.2775, -0.7162, -0.335],
        [0.359, 0.7162, 0.5985, 0.599],
        [0, 0, 0, 1]
    ]
)

STEP5_PARTD_CURRENT_POSE = np.array(
    [
        [-0.7071, 0, 0.7071, 0.0061],
        [0.7071, 0, 0.7071, 0.2596],
        [0, 1, 0, 0.1],
        [0, 0, 0, 1]
    ]
)

STEP5_PARTD_NEW_POSE = np.array(
    [
    [-0.7071, 0, 0.7071, 0.1061],
    [0.7071, 0, 0.7071, 0.3596],
    [0, 1, 0, 0.2],
    [0, 0, 0, 1]
    ]
)

class RPPRRRManipulator(DHRobot):
    """
    Class to model and interact with RPPRRR Manipulator as defined in 23WSB141 - Introduction To Robotics Coursework
    
    Inherits from the DHRobot class of the Robotics Toolbox for Python library.
    
    Attributes:
            L0: L0 of Manipulator
            L1: L1 of Mnaipulator
            L2: L2 of Manipulator
            L3: L3 of Manipulator
            L4: L4 of Manipulator
            L5: L5 of Mnaipulator
            
            THETA1: Init value of of manipulator theta 1
            THETA4: Init value of manipulaotr theta 4
            THETA5: Init value of manipulator theta 5
            THETA6: Init value of manipulator theta 6
            
            D2: Init value of mnaipulator d2
            D3: Init value of manipualtor d3
            
            DH_TABLE: Constant - DH table as specified in coursework brief
            
            links: List of RoboticsToolbox for Python Class instances for initalisation of DHRobot instace
    """

    def __init__(self):
        '''
        RPPRRR Manipulator constructor
        Defines predetermined values specified in 23WSB141 Coursework Brief which describe the manipulator 
        
        '''
        
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

        # Declare DH table - For use in RBT solution
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

        # Define list of RevoluteMDH or PrismaticMDH class instances to create definition of manipulator
        links = [
            RevoluteMDH(# Fake joint to mimic base frame 
                alpha=self.DH_TABLE[0][0], 
                a=self.DH_TABLE[0][1], 
                d=self.DH_TABLE[0][2], 
                offset=self.DH_TABLE[0][3], 
                qlim=np.array([0, 0])
                ), 

            RevoluteMDH( # First Revolute
                alpha=self.DH_TABLE[1][0], 
                a=self.DH_TABLE[1][1], 
                d=self.DH_TABLE[1][2], 
                offset=self.DH_TABLE[1][3], 
                qlim=np.array([np.radians(-180), np.radians(180)])
                ), 

            PrismaticMDH( # Second Prismatic
                alpha=self.DH_TABLE[2][0], 
                a=self.DH_TABLE[2][1], 
                q=self.DH_TABLE[2][2], 
                offset=self.DH_TABLE[2][3], 
                qlim=np.array([0.0, 0.5])
                ),
            PrismaticMDH( # Third Prismastic
                alpha=self.DH_TABLE[3][0], 
                a=self.DH_TABLE[3][1], 
                q=self.DH_TABLE[3][2], 
                offset=self.DH_TABLE[3][3], 
                qlim=np.array([-0.1, 0.1])
                ),

            RevoluteMDH( # Fourth Revolute
                alpha=self.DH_TABLE[4][0],
                a=self.DH_TABLE[4][1], 
                d=self.DH_TABLE[4][2], 
                offset=self.DH_TABLE[4][3], 
                qlim=np.array([np.radians(-90), np.radians(90)])
                ),
            RevoluteMDH( # Fifth Revolute
                alpha=self.DH_TABLE[5][0], 
                a=self.DH_TABLE[5][1], 
                d=self.DH_TABLE[5][2], 
                offset=self.DH_TABLE[5][3], 
                qlim=np.array([np.radians(-180), np.radians(180)])
                ),
            RevoluteMDH( # Sixth Revolute
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


        super().__init__( # To call superclass init, inheriting from DHRobot of Robotics Toolbox library
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
            - If a solution cannot be found False will be returned
        '''
        # Check the joint angles are compatible with fordward kinematic method of Robotics Toolbox
        self.__joint_type_check(joint_angles, list)
        joint_angles = self.__correct_list_size(joint_angles)

        # Calculate forward kinematics to return solution
        try: 
            fk_sol = self.fkine(joint_angles)
        except Exception as e:
            print("An error occured while calculating forawrd kinematics: ", e)
            return False

        return fk_sol
    
    def inverse_kinematics(self, transform: np.ndarray) -> IKSolution:
        '''
        Given a desired positon P and Orientation R, in the form of a SE3 transformation matrix, will return whether the manipulator can reach and all the solutions to do so. Has a 100 maximum search limit.
        If the desired position and orientation are out of reach, method will return the closest solution

        :param transform: Transformation matrix as a numpy ndarray.
        :type transform: transform matrix
        
        :return valid_solutions, invalid solutions: tuple of two lists of soltions which returned Ture and False for valid respectively 
        :type tuple<list<ik_solution>>: IKSolution
        ''' 
        
        print("\nCalculating Inverse Kinematics for the folllowing transformation matrix:\n", transform)
        
        # Checking the compability of the transfromation matrix (type check)
        self.__transform_type_check(transform)
        self.solutions = [] # Multiple soltuions possible for each inverse kinematic problem. Iterates to collect them
        for x in range(0, 16):
            ik = self.ikine_LM(transform)
            self.solutions.append(ik)
        
        # Lists requried for determining possible valid/invalid solutions 
        valid_solutions = []
        invalid_solutions = []
        
        # ikine_LM returns IKSoltuions which will return a soltuion regardless of its validity. This checks the success and only stores those which are successful
        for x in self.solutions:
            if x.success is True:
                valid_solutions.append(x)
            else:
                invalid_solutions.append(x)
        
        # Logic for disaplying the results based on the solutions found and user interactions        
        if len(valid_solutions) > len(invalid_solutions):          
            if len(valid_solutions) > 15:
                print("\n16 Kinematic Solutions found.")
            else:
                print(f"{len(valid_solutions)} Inverse Kinematic Solutions found.")
                    
            ans = input("\nDo you wish to display all solutions? Y/N ").upper()
            if ans == 'Y':
                for x in valid_solutions:
                    print(x)
        else:
            print("\nInverse Kinematic solution not possible.")
            print(f'\nCloest solution: {invalid_solutions[0]}')
        
        # Return solutions for further processing            
        return valid_solutions, invalid_solutions
    
    def step3_inverse_kinematic_solver(self, transform: SE3): # Method for specificly step 3 of the coursework brief
        '''
        Given a transformation matrix, will calculate the inverse kinematic solutions. 
        Will calculate the error in the solution regardless of if a valid soltuion is found or not
        
        :param transform: Transformation matrix for the inverse kinematic solution to be found
        :type ndarray: Numpy ndarray
        '''
        
        #Index used to track whether method should use valid or invalid kinematic solutions
        index = 1
        ik_sol_1 = self.inverse_kinematics(transform)
        if self.__compare_len(ik_sol_1[0], ik_sol_1[1]): # Comparing valid v invalid solutions to calculate whether the fkine_LM model was able to return a soltuion
            index = 0
        
        lowest_error = 100 # Arbitary value which an error won't be lower than for comparison
        best_sol = IKSolution 
        for x in ik_sol_1[index]: # Sort through solutions and find the lowest error == 'best  soltuion'
            ik_sol_1_error = self.ik_error(transform, x)
            if ik_sol_1_error < lowest_error:
                lowest_error = ik_sol_1_error
                best_sol = x
        
        # Interact with user for display of results
        print(f"\nBest IK solution {best_sol.q} with the lowest error of: {lowest_error}")

        if input(f"\nDisplay plot for best IK solution for transform: \n{transform}? \nY/N ").upper() == 'Y':
            self.plot(best_sol.q, block=True)
        
        return best_sol, ik_sol_1_error
    
    def ik_error(self, transform: SE3, ik_solution: IKSolution):
        '''
        Given a transform and IK Solution, will return the error
        
        :param transform: The transform the IK solution is based on
        :type transform: SE3 and numpy.ndarray
        :param ik_solution: proposed IK solution
        :type ik_solution: IKSolutions
        
        :return ik_error: numerical error
        :type ik_error: float
        '''
        # Type checking
        self.__transform_type_check(transform)
        self.__joint_type_check(ik_solution, IKSolution)
        ik_error = np.linalg.norm(transform - self.fkine(ik_solution.q)) # Norm difference error
        return ik_error
            
    def joint_velocities(self, joint_angles: 'list[float]', joint_velocities: 'list[float]'):
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
        self.__joint_type_check(joint_angles, list) # Will raise an exception if incorrect type
        joint_angles = self.__correct_list_size(joint_angles) #Check list has sufficient joint variables, add 0s to start/end for 'fake' joints if not already present
        
        #Type checking for joint_velocities
        self.__joint_type_check(joint_velocities, list) # Will raise an exception if incorrect type
        joint_velocities = self.__correct_list_size(joint_velocities) #Check list has sufficient joint variables, add 0s to start/end for 'fake' joints if not already present
        
        # Utilise robotics toolbox for generating jacobian soltuion and the concequent velocities
        jacobian_matrix = self.jacob0(joint_angles)
        velocites = jacobian_matrix @ joint_velocities
        linear_velocities = velocites[:3]
        angular_velocities = velocites[3:]
        
        return jacobian_matrix, linear_velocities, angular_velocities
    
    def static_torques(self, mass: float, g: float, transform: SE3):
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
        self.__transform_type_check(transform)
        
        # Calculate joint angles from transform using inverse_kinematics
        # Joint angles used to declare wrench variable 
        index = 1
        ik_sol_1 = self.inverse_kinematics(transform)
        if self.__compare_len(ik_sol_1[0], ik_sol_1[1]):
            index = 0
        
        # Arbitary error value which the error wont be higher then. Compares solutions to find lowest error
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
    
    def generate_trapezoida_velocity(self, theta_s: float, theta_f: float, t_r: float): # Obsolete
        '''
        Given a starting and final theta value, with a time to complete, will calculate the trapezoidal position, velocity, and acceleration
        
        Will display plot and print the generated function
        
        :param theta_s: Starting theta value
        :type theta_s: Float
        :param theta_f: End theta value
        :type theta_s: Floats
        '''
        
        tg = rbt.trapezoidal(q0=theta_s, qf=theta_f, t=t_r)
        print(len(tg))
        tg.plot(block=True)
        
        print(rbt.trapezoidal_func(q0=theta_s, qf=theta_f, T=t_r)(1))
        
    def calculate_cartesian_point_to_point(self, intial_transform: SE3, end_transform: SE3):
        '''
        Given an intial transfrom and goal transfrom, will calcualte simple straight line cartesian transform, will plot if so desired
        
        :param inital_transform: Starting pose
        :type intial_transform: SE3
        :param end_transform: Final pose
        :type end_transform: SE3
        '''
        
        # Ensuring the transformation matricies are compatible with robotics toolbox cartesian trajectory method
        intial_pose = self.ikine_LM(intial_transform)
        new_pose = self.ikine_LM(end_transform)
        intial_pose = self.fkine(intial_pose.q)
        new_pose = self.fkine(new_pose.q)
        tg = rbt.ctraj(intial_pose, new_pose, 3)
        
        print('Cartesian tranfrom: \n', tg)
        if input('Display trajectory plot? Y/N ').upper() == 'Y':
            tg.plot(block=False)
            plt.show()
    
    def __transform_type_check(self, var: SE3):
        '''
        Private method
        
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
    
    def __joint_type_check(self, var: any, output: any, alt_output=None):
        '''
        Private method
        
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
        
    def __correct_list_size(self, input_list: 'list[float]') -> list:
        '''
        Private method
        
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
        
    def __compare_len(self, list1: list, list2: list):
        '''
        Private method
        
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
        '''
        Sub-Class of RPPRRRManipulator Class which utilises Sympy library to expand capabilities of RPPRRRManipulator class
        
        Attributes:
            L0: Sympy L0 symbol
            L1: Sympy L1 symbol
            L2: Sympy L2 symbol
            L3: Sympy L3 symbol
            L4: Sympy L4 symbol
            L5: Sympy L5 symbol
            
            THETA1: Sympy THETA1 symbol
            TEHTA4: Sympy THETA4 symbol
            TEHTA5: Sympy THETA5 symbol
            TEHTA6: Sympy THETA6 symbol
            
            D2: Sympy D2 symbol
            D3: Sympy D3 symbol
            
            DH_TABLE: Sympy matrix of DH Table values as specified in coursework breif
            
            TB_1, T1_2, T2_3, T3_4, T4_5, T5_6, T6_T, T_T: Sympy Matrix of each transfrom of each frame
            TB_T: Sympy Matrix transfrom from base to tool frame
            T1_6: Sympy matrix transform from joint 1 to 6
            TB_T_FK: Substituted values, solved, transform for forward kinematics
            
            PCB_B, TC1_1, PC2_2, PC3_3, PC4_4, PC5_5, PC6_6 = Numpy array vectors of centre of mass values
            LINK_MASSES = Numpy array of all ink masses
            M0, M1, M2, M3, M4, M5, M6, M7 = Masses of each link
            
            I_CB_B, I_C1_1, I_C2_2, I_C3_3, I_C4_4, I_C5_5, I_C6_6: Numpy array for each link inertia tensor
            
            EMREGANCY_STOP_POSE: List of joint values at time of emergancy stop
            E_STOP_JOINT_VELOCITIES: Joint velocities at time of emergancy stop
            
            thetaB_i_dot, theta1_i_dot, d2_i_dot, d3_i_dot, theta4_i_dot, theta5_i_dot, theta6_i_dot: Velocities of each joint at emergancy stop
            
            T1_MAX_VEL, T2_MAX_VEL, T3_MAX_VEL, T4_MAX_VEL, T5_MAX_VEL, T6_MAX_VEL: +- maximum velocity of each joint
            T1_MAX_TORQUE, T2_MAX_TORQUE, T3_MAX_TORQUE, T4_MAX_TORQUE, T5_MAX_TORQUE, T6_MAX_TORQUE: +- maximum torque of each joint
            
            MASS: mass applied to tool frame
            G: Constant of gravity
            
            F6_6: Numpy array force vector for final frame
            N6_6: Numpy array moment vector for final frame
            
            thetaB_i_2dot, theta1_i_2dot, d2_i_2dot, d3_i_2dot, theta4_i_2dot, theta5_i_2dot, theta6_i_2dot: Acceleration of each joint at emergancy stop
            
            omega_0_0: Numpy matrix vecloity acceleration vector
            omega_dot_0_0: Numpy matrix angular acceleration vector
            v_dot_0_0: Numpy matrix linear velocity vector
        
        '''
        
        def __init__(self):
            # Delcare Symbols for DH table abd transformation, reassignment of previous variables now that they have been used in the initialisation of the RBT model
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
            
            # Declare DH Table - DH Table is reassigned from previous declaration for use in Sympy. Not neccessary but safe for compability
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
            self.TB_T_FK = self.TB_T.subs({ # Substitude in joint values to get FK
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
            
            # Delcaring all fo the constants for use in dynamics calculations
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
            self.EMERGANCY_STOP_POSE = [np.radians(45), 0.5, 0.09, np.radians(90), np.radians(45), np.radians(90)]
            
            # Joint Velocities at time of emergancy stop
            self.E_STOP_JOINT_VELOCITIES = [-35, 0.1, -0.01, -60, 50, 40]
            
            # Constant for converting to radians
            self.DEG = np.pi / 180
            
            # Emergency Stop speeds
            self.thetaB_i_dot = 0 * self.DEG
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
            self.thetaB_2dot = 0 / self.I_CB_B[2, 2]
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
            self.v_dot_0_0 = np.matrix([[0], [self.G], [0]])
            
            # Run method to compute angular/linear velocity/acceleration 
            # Run method to compute linke force/torque and Joint Torque
            self.calc_vel_acc()
            self.calc_joint_vals(F=self.F_values, N=self.N_values)
            self.__solve_matrices()
            self.__create_dot_dataframe()
            self.__create_totals_dataframe()
            pd.set_option('display.max_colwidth', None)
                        
            
        def acc_revolute(self, transform: SE3, omega: sy.Matrix, theta_i_dot: float, omega_dot: sy.Matrix, theta_i_2dot: float, v_dot: sy.Matrix, PC: sy.Matrix, mass: float, i: sy.Matrix):
            '''
            Given the transform, omega, theta_dot, omega_dot, theta_2dot, v_dot, PC (Centre of Mass), mass, and inertia tensor, will calcualte the needed values for dynamics calculations of a revolute joint
            
            :param transform: transformation matrix of curernt link
            :type transfrom: ndarray / SE3
            :param omega: Angular velocity of previous joint
            :type omage: Matrix
            :param theta_i_dot: Velocity of joint
            :type theta_i_dot: float
            :param omega_dot: Angular Acceleration of previous joint
            :type oemga_dot Matrix
            :param theta_i_2dot: Acceleration of joint
            :type theta_i_2dot: float
            :param v_dot: Linear acceleration of previous joint
            :type v_dot: Matrix
            :param PC: Centre of Mass vector of joint
            :type PC: Matrix
            :param mass: mass of current joint in KG
            :type mass: float
            :param i: Inertia tensor of joint in Kg.m^2
            :type i: Matrix
            
            :return omega_new, omega_dot_new, v_dot_new, v_centre_dot_new, F_new, N_new: Return the calculated Velcoties and Accelerations for the specified joint as well as the Moment and Forces
            :return type: tuple<Matrix>
            '''
            
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
            N_new = i @ omega_dot_new + np.transpose(np.cross(omega_new.transpose(),np.transpose(i @ omega_new)))
            
            return omega_new, omega_dot_new, v_dot_new, v_centre_dot_new, F_new, N_new
        
        def acc_prismatic(self, transform: SE3, omega: sy.Matrix, d_i_dot: float, omega_dot: sy.Matrix, d_i_2dot: float, v_dot: sy.Matrix, PC: sy.Matrix, mass: float, i: sy.Matrix):
            '''
            Given the transform, omega, d_i_dot, omega_dot, d_i_2dot, v_dot, PC (Centre of Mass), mass, and inertia tensor, will calcualte the needed values for dynamics calculations of a revolute joint
            
            :param transform: transformation matrix of curernt link
            :type transfrom: ndarray / SE3
            :param omega: Angular velocity of previous joint
            :type omage: Matrix
            :param d_i_dot: Velocity of joint
            :type d_i_dot: float
            :param omega_dot: Angular Acceleration of previous joint
            :type oemga_dot Matrix
            :param d_i_2dot: Acceleration of joint
            :type d_i_2dot: float
            :param v_dot: Linear acceleration of previous joint
            :type v_dot: Matrix
            :param PC: Centre of Mass vector of joint
            :type PC: Matrix
            :param mass: mass of current joint in KG
            :type mass: float 
            :param i: Inertia tensor of joint in Kg.m^2
            :type i: Matrix
            
            :return omega_new, omega_dot_new, v_dot_new, v_centre_dot_new, F_new, N_new: Return the calculated Velcoties and Accelerations for the specified joint as well as the Moment and Forces
            :return type: tuple<Matrix>
            '''
            
            rotation = transform[:3, :3]
            position = transform[:3, 3]
            
            # Velocity
            omega_new = rotation.transpose() @ omega
            
            # Angular Acceleration
            omega_dot_new = rotation.transpose() @ omega
            v_dot_new = rotation.T @ (np.transpose(np.cross(omega_dot.T, position.T))) + np.transpose(np.cross(omega.T, (np.cross(omega.T, position.T))) + v_dot.T) + np.transpose((np.cross((2 * omega.T), (np.matrix([0, 0, d_i_dot]) + np.matrix([0, 0, d_i_2dot])))))
        
            # Adjusting for gravity
            v_centre_dot_new = np.cross(omega_dot_new.transpose(), PC.transpose()) + np.cross(omega_new.transpose(),np.cross(omega_new.transpose(),PC.transpose())) + v_dot_new.transpose()
            v_centre_dot_new = v_centre_dot_new.transpose()
            
            # Force action at centre of mass
            F_new = mass * v_centre_dot_new

            # torque action at centre of mass
            N_new = i @ omega_dot_new + np.transpose(np.cross(omega_new.transpose(), np.transpose(i @ omega_new)))
            
            return omega_new, omega_dot_new, v_dot_new, v_centre_dot_new, F_new, N_new
        
        def dynamics_forces(self, transform: SE3, N: sy.Matrix, F: sy.Matrix, n: sy.Matrix, f: sy.Matrix, PC: sy.Matrix):
            '''
            Given the transform, Link Moment, Link Force, Previous link moment, Previous Link force, and Centre of Mass vector of the current joint, will calculate link torque, force and Joint Torque Tau
            
            :param transform: transformation matrix of current link
            :type transform: ndarray / SE3
            :param N: Moment of current link
            :type N: Matrix
            :param F: Force of current link
            :type F: Matrix
            :param: n: Moment of previous link
            :type n: Matrix
            :param f: Froce of previous link
            :type f: Matrix
            :param PC: Centre of mass vector of current link
            :type PC: Matrix
            
            :return link_force, link_torque, tau: Returns calculated values of link force and torque as well as Tau - the Joint Torque
            :return type: tuple<Matrix>
            '''
            
            rotation = transform[:3, :3]
            position = transform[:3, 3]
            
            link_torque = N + rotation @ n + np.transpose(np.cross(PC.transpose(),F.transpose()) + np.cross(position.transpose(),np.transpose(rotation @ f)))
            
            link_force = rotation @ f + F
            
            tau = n.transpose() * sy.Matrix([[0], [0], [1]])
            
            return link_force, link_torque, tau
        
        def calc_vel_acc(self):
            '''
            Method to run the calculations for the Angular/Linear Velocities and Accelerations
            Placed in Method to consolidate and not clutter class __init__ method
            
            :return omega_values, omega_dot_values, v_dot_values, v_centre_dot_values, F_values, N_values
            :return type: tuple<list<matrix<float>>>
            '''
            
            #Iterate outwards from base to end effector
            #Base
            omega_0_0, omega_dot_0_0, v_dot_0_0, v_centre_dot_0_0, F_0_0, N_0_0 = self.acc_revolute(transform=self.TB_1, omega=self.omega_0_0, theta_i_dot=self.thetaB_i_dot, omega_dot=self.omega_dot_0_0, theta_i_2dot=self.thetaB_2dot, v_dot=self.v_dot_0_0, PC=self.PCB_B, mass=self.M0, i=self.I_CB_B) 
            
            #Joint 1 - Revolute
            omega_1_1, omega_dot_1_1, v_dot_1_1, v_centre_dot_1_1, F_1_1, N_1_1 = self.acc_revolute(transform=self.T1_2, omega=self.omega_0_0, theta_i_dot=self.theta1_i_dot, omega_dot=self.omega_dot_0_0, theta_i_2dot=self.theta1_2dot, v_dot=self.v_dot_0_0, PC=self.PC1_1, mass=self.M1, i=self.I_C1_1)
            
            #Joint 2 - Prismatic
            omega_2_2, omega_dot_2_2, v_dot_2_2, v_centre_dot_2_2, F_2_2, N_2_2 = self.acc_prismatic(transform=self.T2_3, omega=omega_1_1, d_i_dot=self.d2_i_dot, omega_dot=omega_dot_1_1, d_i_2dot=self.d2_2dot, v_dot=v_dot_1_1, PC=self.PC2_2, mass=self.M2, i=self.I_C2_2)
            
            #Joint 3 - Prismatic
            omega_3_3, omega_dot_3_3, v_dot_3_3, v_centre_dot_3_3, F_3_3, N_3_3 = self.acc_prismatic(transform=self.T3_4, omega=omega_2_2, d_i_dot=self.d3_i_dot, omega_dot=omega_dot_2_2, d_i_2dot=self.d3_2dot, v_dot=v_dot_2_2, PC=self.PC3_3, mass=self.M3, i=self.I_C3_3)
            
            #Joint 4 - Revolute
            omega_4_4, omega_dot_4_4, v_dot_4_4, v_centre_dot_4_4, F_4_4, N_4_4 = self.acc_revolute(transform=self.T4_5, omega=omega_3_3, theta_i_dot=self.theta4_i_dot, omega_dot=omega_dot_3_3, theta_i_2dot=self.theta4_2dot, v_dot=v_dot_3_3, PC=self.PC4_4, mass=self.M4, i=self.I_C4_4)
            
            #Joint 5 - Revolute
            omega_5_5, omega_dot_5_5, v_dot_5_5, v_centre_dot_5_5, F_5_5, N_5_5 = self.acc_revolute(transform=self.T5_6, omega=omega_4_4, theta_i_dot=self.theta5_i_dot, omega_dot=omega_dot_4_4, theta_i_2dot=self.theta5_2dot, v_dot=v_dot_4_4, PC=self.PC5_5, mass=self.M5, i=self.I_C5_5)
            
            #Joint 6 - Revolute
            omega_6_6, omega_dot_6_6, v_dot_6_6, v_centre_dot_6_6, F_6_6, N_6_6 = self.acc_revolute(transform=self.T6_T, omega=omega_5_5, theta_i_dot=self.theta6_i_dot, omega_dot=omega_dot_5_5, theta_i_2dot=self.theta6_2dot, v_dot=v_dot_5_5, PC=self.PC6_6, mass=self.M6, i=self.I_C6_6)
            
            # Create lists of each values so that they are more easily worked with
            self.omega_values = [omega_0_0, omega_1_1, omega_2_2, omega_3_3, omega_4_4, omega_5_5, omega_6_6]
            self.omega_dot_values = [omega_dot_0_0, omega_dot_1_1, omega_dot_2_2, omega_dot_3_3, omega_dot_4_4, omega_dot_5_5, omega_dot_6_6]
            self.v_dot_values = [v_dot_0_0, v_dot_1_1, v_dot_2_2, v_dot_3_3, v_dot_4_4, v_dot_5_5, v_dot_6_6]
            self.v_centre_dot_values = [v_centre_dot_0_0, v_centre_dot_1_1, v_centre_dot_2_2, v_centre_dot_3_3, v_centre_dot_4_4, v_centre_dot_5_5, v_centre_dot_6_6]
            self.F_values = [F_0_0, F_1_1, F_2_2, F_3_3, F_4_4, F_5_5, F_6_6]
            self.N_values = [N_0_0, N_1_1, N_2_2, N_3_3, N_4_4, N_5_5, N_6_6]
            
            return self.omega_values, self.omega_dot_values, self.v_dot_values, self.v_centre_dot_values, self.F_values, self.N_values
            
        def calc_joint_vals(self, F: "list[sy.Matrix]", N:  "list[sy.Matrix]"):
            '''
            Method to run the inward iteration calculations and return the total Force, Moment and Joint Torque
            Placed in own method to consolidate and not clutter class __init__ method
            
            :param F: Force values calculated in 'calc_vel_acc'
            :type F: List<Matrix>
            :Param N: Moment values calculaed in 'calc_vel_acc'
            :type N: List<Matrix>
            '''
            
            # Iterate from end effector to base
            self.NT_T = np.array([[0], [0], [0]]) # No intial moment
            self.FT_T = np.array([[0], [-self.MASS*self.G], [0]]) # Froces of gravity
            
            # Transform from current joint
            # N from current joint
            # F from current joint
            # n from previous joint
            # f from previous joint
            # PC from previous joint
            
            # Joint 6
            F6_6_total, N_6_6_total, Torque_Joint6 = self.dynamics_forces(transform=self.T6_T, N=N[6], F=F[6], n=self.NT_T, f=self.FT_T, PC=self.PC6_6)
            
            # Joint 5
            F5_5_total, N_5_5_total, Torque_Joint5 = self.dynamics_forces(transform=self.T5_6, N=N[5], F=F[5], n=N[6], f=F[6], PC=self.PC5_5)
            
            # Joint 4
            F4_4_total, N_4_4_total, Torque_Joint4 = self.dynamics_forces(transform=self.T4_5, N=N[4], F=F[4], n=N[5], f=F[5], PC=self.PC4_4)
            
            # Joint 3
            F3_3_total, N_3_3_total, Torque_Joint3 = self.dynamics_forces(transform=self.T3_4, N=N[3], F=F[3], n=N[4], f=F[4], PC=self.PC3_3)
            
            # Joint 2
            F2_2_total, N_2_2_total, Torque_Joint2 = self.dynamics_forces(transform=self.T2_3, N=N[2], F=F[2], n=N[3], f=F[3], PC=self.PC2_2)
            
            # Joint 1
            F1_1_total, N_1_1_total, Torque_Joint1 = self.dynamics_forces(transform=self.T1_2, N=N[1], F=F[1], n=N[2], f=F[2], PC=self.PC1_1)
            
            # Joint 0 - Base
            FB_B_total, N_B_B_total, Torque_JointB = self.dynamics_forces(transform=self.TB_1, N=N[0], F=F[0], n=N[1], f=F[1], PC=self.PCB_B)
            
            
            self.F_totals = [F6_6_total, F5_5_total, F4_4_total, F3_3_total, F2_2_total, F1_1_total, FB_B_total]
            self.N_totals = [N_6_6_total, N_5_5_total, N_4_4_total, N_3_3_total, N_2_2_total, N_1_1_total, N_B_B_total]
            self.joint_torque_totals = [Torque_Joint6, Torque_Joint5, Torque_Joint4, Torque_Joint3, Torque_Joint2, Torque_Joint1, Torque_JointB]
            
            return self.F_totals, self.N_totals, self.joint_torque_totals
        
        def display_all_equations(self):
            '''
            Method to iterate through all equations and display the pure, unsubstituted equations
            
            '''
            
            print('Outward iteration from base to end effector')
            print('Displaying Omega, Omega_Dot, V_dot, V_centre_dot, Force and Moment for each link...\n\n\n\n')
            for x in range(0, 5):
                print(f'\n\n\n\nLink {x} Omega: ', self.omega_values[x].evalf())
                print(f'\n\n\n\nLink {x} Omega_Dot: ', self.omega_dot_values[x].evalf())
                print(f'\n\n\n\nLink {x} v_dot: ', self.v_dot_values[x].evalf())
                print(f'\n\n\n\nLink {x} v_centre_dot: ', self.v_centre_dot_values[x].evalf())
                print(f'\n\n\n\nLink {x} F_value: ', self.F_values[x].evalf())
                print(f'\n\n\n\nLink {x} N_Value: ', self.N_values[x].evalf())
            
            print('\n\n\n\nInward iteration from end effector to base')
            print('\n\n\n\nDisaplying Total Force and Moment on each link, as well as torque required on each joint...\n\n\n\n')
            for x in range(5, 0, -1):
                print(f'\n\n\n\nLink {x} F_Total: ', self.F_totals[x].evalf())
                print(f'\n\n\n\nLink {x} N_Total: ', self.N_totals[x].evalf())
                print(f'\n\n\n\nJoint {x} Tau: ', self.joint_torque_totals[x].evalf())
        
        def cubic_polynomial_TG(self, theta_0: float, theta_f: float, t_f:float) -> "tuple[float]":
            '''
            Given the starting (theta_0) and finishing (theta_f) joint angle and the time to compelte the movement (t_f) will compute the cubic polynomial coeffiecents required
            
            :param theta_0: Starting Joint Angle 
            :type theta_0: float
            :param theta_f: Final Joint Angle
            :type theta_f: float
            :param t_f: Time to move in Seconds
            :type t_f: float
            
            :return a_0, a_1, a_2, a_3: Tuple of cubic coefficients
            :type [a_0, a_1, a_2, a_3]: tuple[float]
            '''
            
             # Equations from Introduction to Robotics - John Craig (4th Edition)
            a_0 = theta_0
            a_1 = 0
            a_2 = (3 / np.power(t_f, 2)) * (theta_f - theta_0)
            a_3 = (-2 /np.power(t_f, 3)) * (theta_f - theta_0)
            
            return [a_0, a_1, a_2, a_3]
        
        def quintic_polynomial_TG(self, theta_0: float, theta_f: float, t_f: float):
            '''
            Given the starting (theta_0) and finishing (theta_f) joint angle and the time to compelte the movement (t_f) will compute the quintic polynomial coeffiecents required
            
            :param theta_0: Starting Joint Angle 
            :type theta_0: float
            :param theta_f: Final Joint Angle
            :type theta_f: float
            :param t_f: Time to move in Seconds
            :type t_f: float
            
            :return a_0, a_1, a_2, a_3. a_4, a_5: Tuple of quintic coefficients
            :type [a_0, a_1, a_2, a_3, a_4, a_5]: tuple[float]
            '''
            
            #Differentiate for deriviatves
            t = sy.Symbol('t')
            theta_0_dot = sy.diff(theta_0, t)
            theta_f_dot = sy.diff(theta_f, t)
            theta_0_2dot = sy.diff(theta_0_dot, t)
            theta_f_2dot = sy.diff(theta_f_dot, t)
            
            # Equations from Introduction to Robotics - John Craig (4th Edition)
            a_0 = theta_0
            a_1 = theta_0_dot
            a_2 = theta_0_2dot / 2
            
            a_3 = ((20 * theta_f) - (20 * theta_0) - ((8 * theta_f_dot) + (12 * theta_0_dot)) * t_f - ((3 * theta_0_2dot) - (theta_f_2dot)) * np.power(t_f, 2)) / (2 * np.power(t_f, 3))
            
            a_4 = ((30 * theta_0) - (30 * theta_f) + ((14 * theta_f_dot) + (16 * theta_0_dot)) * t_f +  ((3 * theta_0_2dot) - (2 * theta_f_2dot)) * np.power(t_f, 2)) / (2 * np.power(t_f, 4))
            
            a_5 = ((12 * theta_f) - (12 * theta_0) - ((6 * theta_f_dot) + (6 * theta_0_dot)) * t_f - (theta_0_2dot - theta_f_2dot) * np.power(t_f, 2)) / (2 * np.power(t_f, 5))
            
            return [a_0, a_1, a_2, a_3, a_4, a_5]
        
        def generate_polynomial_plot(self, coeff: "list[float]", t_f: float, joint_type: str, order=3):
            '''
            Given the calculated cubic polynomial coeffients and time, will generate plots for Joint Angle, Joint Velocity and Joint Acceleration
            
            :param coeff: List of coefficents from cubic polynomial
            :type coeff: List[float]
            :param t_f: Time for movement in seconds
            :type t_f: float
            :param joint_type: type of joint for title labelling
            :type joint_type: str
            
            :return traj_plot: Joint Angle plot
            :type traj_plot: sy.Plot
            :return vel_plot: Joint Velocity plot
            :type vel_plot: sy.Plot
            :return acc_plot: Joint Acceleration plot
            :type acc_plot: sy.Plot
            '''
            
            # List of Lists for selecting coour based on whether the plot is cubic or quintic
            line_color = [['blue', 'green', 'red', 'yellow'], ['dodgerblue', 'magenta', 'darkviolet', 'orange']]
            index = 0# Index determined by quintic or cubic for selection of colours later on
            t = sy.Symbol('t')
            if order == 5:
                traj = coeff[0] + (coeff[1] * t) + (coeff[2] * np.power(t, 2)) + (coeff[3] * np.power(t, 3)) + (coeff[4] * np.power(t, 4)) + (coeff[5] * np.power(t, 5))
                index = 1
            else:
                traj = coeff[0] + (coeff[1] * t) + (coeff[2] * np.power(t, 2)) + (coeff[3] * np.power(t, 3))
            # Eqs either cubic or quintic
            vel = sy.diff(traj, t)
            acc = sy.diff(vel, t)
            jerk = sy.diff(acc, t)
            
            # Generate plots dependent on Revolute, D1 (Prismatic) or D2 (Prismatic) joints 
            if joint_type.upper() == 'REVOLUTE':
                traj_plot = sy.plot(traj, (t, 0, t_f), ylabel='Theta (Degrees)', show=False, title='Position', line_color=line_color[index][0])
                vel_plot = sy.plot(vel, (t, 0, t_f), ylabel='Theta Dot', show=False, title='Velocity', line_color=line_color[index][1])
                acc_plot = sy.plot(acc, (t, 0, t_f), ylabel='Theta Dot Dot', show=False, title='Acceleration', line_color=line_color[index][2])
                jerk_plot = sy.plot(jerk, (t, 0, t_f), ylabel='Theta Dot Dot Dot', show=False, title='Angular Jerk', line_color=line_color[index][3])
            elif joint_type.upper() == 'D1':
                traj_plot = sy.plot(traj, (t, 0, t_f), ylabel='D1 (Degrees)', show=False, title='Position', line_color=line_color[index][0])
                vel_plot = sy.plot(vel, (t, 0, t_f), ylabel='D1 Dot', show=False, title='Velocity', line_color=line_color[index][1])
                acc_plot = sy.plot(acc, (t, 0, t_f), ylabel='D1 Dot Dot', show=False, title='Acceleration', line_color=line_color[index][2])
                jerk_plot = sy.plot(jerk, (t, 0, t_f), ylabel='D1 Dot Dot Dot', show=False, title='Angular Jerk', line_color=line_color[index][3])
            elif joint_type.upper() == "D2":
                traj_plot = sy.plot(traj, (t, 0, t_f), ylabel='D2 (Degrees)', show=False, title='Position', line_color=line_color[index][0])
                vel_plot = sy.plot(vel, (t, 0, t_f), ylabel='D2 Dot', show=False, title='Velocity', line_color=line_color[index][1])
                acc_plot = sy.plot(acc, (t, 0, t_f), ylabel='D2 Dot Dot', show=False, title='Acceleration', line_color=line_color[index][2])
                jerk_plot = sy.plot(jerk, (t, 0, t_f), ylabel='D2Dot Dot Dot', show=False, title='Angular Jerk', line_color=line_color[index][3])
            
            return traj_plot, vel_plot, acc_plot, jerk_plot
        
        def display_polynomials(self, revolute_plots: 'tuple[sy.plot]', d1_plots: 'tuple[sy.plot]', d2_plots: 'tuple[sy.plot]'):
            '''
            Method to plot the differnt revolute, d1, and d2, polynomial trajectories
            
            :param revolute_plot: List of Sympy.plot objects of revolute Position, Velocity and Acceeleration plots
            :type revolute_plot: List[sy.Plot]
            :param d1_plot: List of Sympy.plot objects of prismatic Position, Velocity and Acceeleration plots
            :type d1_plot: List[sy.Plot]
            :param d2_plot: List of Sympy.plot objects of prismatic Position, Velocity and Acceeleration plots
            :type d2_plot: List[sy.Plot]
            '''
            # Utilise sympy PlotGrid to quickly plot all graphs
            sy.plotting.PlotGrid(4, 3, revolute_plots[0], d1_plots[0], d2_plots[0], revolute_plots[1], d1_plots[1], d2_plots[1], revolute_plots[2], d1_plots[2], d2_plots[2], revolute_plots[3], d1_plots[3], d2_plots[3])
            
        def display_all_polynomial_plots(self, revolute_cubic: 'tuple[sy.plot]', revolute_quintic: 'tuple[sy.plot]', d1_cubic: 'tuple[sy.plot]', d1_quintic: 'tuple[sy.plot]', d2_cubic: 'tuple[sy.plot]', d2_quintic: 'tuple[sy.plot]'):
            '''
            Method to plot all revolute, d1, and d1 polynomial trajectories of both cubic and quintic solutions
            
            :param revolute_cubic: List of Sympy.plot objects of revolute Position, Velocity and Acceeleration plots
            :type revolute_cubic: List[sy.Plot]
            :param revolute_quintic: List of Sympy.plot objcects of revolute Position, Velocity and Acceeleration plots
            :type revolute_quintic: List[sy.plot]
            :param d1_cubic: List of Sympy.plot objects of prismatic Position, Velocity and Acceeleration plots
            :type d1_cubic: List[sy.Plot]
            :param d1_quintic: List of Sympy.plot objects of pristmatic Position, Velocity, and Acceleration plot
            :type d1_quintic: List[sy.plot]
            :param d2_cubic: List of Sympy.plot objects of prismatic Position, Velocity and Acceeleration plots
            :type d2_cubic: List[sy.Plot]
            :param d2_quintic: List of Sympy.plot objects of prismtatic Position, Velcoity and ACceleration plots
            '''
            # Utilise sympy PlotGrid to quickly plot all graphs
            sy.plotting.PlotGrid(4, 6, revolute_cubic[0], revolute_quintic[0], d1_cubic[0], d1_quintic[0], d2_cubic[0], d2_quintic[0], revolute_cubic[1], revolute_quintic[1], d1_cubic[1], d1_quintic[1], d2_cubic[1], d2_quintic[1], revolute_cubic[2], revolute_quintic[2], d1_cubic[2], d1_quintic[2], d2_cubic[2], d2_quintic[2], revolute_cubic[3], revolute_quintic[3], d1_cubic[3], d1_quintic[3], d2_cubic[3], d2_quintic[3])
        
        def calculate_polynomials_via_point(self, theta_1: float, theta_2: float, theta_3: float, acc=50, joint_type='revolute'):
            '''
            Given a starting, via, and final point will calculate and plot the position, velcoity and acceleration graph as a function of time
            
            :param theta_1: starting theta value
            :type theta_1: float
            :param theta_2: Via point theta value
            :type theta_2: float
            :param theta_3: final theta value
            :type theta_3: float
            '''
            
            # Time difference predetermined in coursework brief - This should be a method parameter
            td_12 = td_23 = 2
            
            # Calc values using equations in Introduction To Robotics - John Craig (4th Edition)
            theta1_2dot = self.__sign(theta_2, theta_1) * acc
            t_1 = td_12 - np.sqrt(np.power(td_12 , 2) - (2 * ( theta_2 - theta_1)) / theta1_2dot)
            theta12_dot = (theta_2 - theta_1) / (td_12 - (1/2 * t_1))
            
            theta3_2dot = self.__sign(theta_2, theta_3) * acc
            t_3 = td_23 - np.sqrt(np.power(td_23 , 2) + (2 * ( theta_3 - theta_2)) / theta3_2dot)
            theta23_dot = (theta_3 - theta_2) / (td_23 - (1/2 * t_3))
            
            theta2_2dot = self.__sign(theta23_dot, theta12_dot) * acc #  Know that theta 2 doesn't have a curve due to teh values of t1 and t3 resulting in t2 being 0
            t_2 = (theta23_dot - theta12_dot) / theta3_2dot
            
            #Linear time 
            t_13 = (td_12 + td_23) - t_1 - (1/2 * t_3)
            
            # Declare arrays so that they can be appended using numpy.append
            # Arrays to be passed to plottting function
            time = np.array([])
            theta = np.array([])
            theta_dot = np.array([])  
            theta_dot_dot = np.array([])  
            for t in np.arange(0, t_1 + t_13 + t_3, 0.001): # Iterate over the time 
                time = np.append(time, t) # time appended consistently
                if t <= t_1: # For each region (1st blend, linear, 2nd blend) calculate theta, theta_dot, theta_dot_dot and append to relevant array. 
                    
                    t_inb = t - (1/2 * t_1 + t_13)
                    theta = np.append(theta, ( theta_1 + (1/2 * theta12_dot * np.power(t, 2))))
                    theta_dot = np.append(theta_dot, (theta12_dot + (theta3_2dot * t_inb)))
                    theta_dot_dot = np.append(theta_dot_dot, theta1_2dot)
                    
                elif t > t_1 and t < t_1 + t_13:
                    
                    theta = np.append(theta, (theta_1 + (theta12_dot * t)))
                    theta_dot = np.append(theta_dot, theta12_dot)
                    theta_dot_dot = np.append(theta_dot_dot, 0)
                    
                elif t > t_1 + t_13 and t < t_1 + t_13 + t_3:
                    
                    t_inb = t - (1/2 * t_1 + t_13)
                    theta = np.append(theta, ( theta_1 + theta12_dot * (1/2 * t_1 + t_13) + theta12_dot * t_inb + 1/2 * theta3_2dot * np.power(t_inb, 2)))
                    theta_dot = np.append(theta_dot, (theta12_dot + (theta3_2dot * t_inb)))
                    theta_dot_dot = np.append(theta_dot_dot, theta3_2dot)
                    
                else: # Exists for edge cases of elif limits to ensure there is not a array mismatch error in plotting

                    theta = np.append(theta, t)
        
            self.__generate_plot(time, theta, theta_dot, theta_dot_dot, joint_type=joint_type) # Call private function to generate and display plots
        
        def __sign(self, val1: float, val2: float) -> int:
            '''
            Private function to see if the difference between two values is positive or negative
            
            :param val1: Value to be deducted from
            :type val1: float
            :param val2: Value to take away from val1
            :type val2: float
            '''
            
            return -1 if val1 - val2 < 0 else 1 # Inline conditonal
                
        def __generate_plot(self, time: list, theta: list, theta_dot: list, theta_dot_dot: list, joint_type: str):
            '''
            Private function to generate plots for Position, Velocity and Acceleration
            
            :param time: List of time values
            :type time: list<float>
            :param theta: List of theta values
            :type theta: list<float>
            :param theta_dot: List of theta_dot values
            :type theta_dot: list<float>
            :param theta_dot_dot: List of theta_dot_dot values
            :type theta_dot_dot: list<float>
            :param joint_type: either 'PRISMATIC' or 'REVOLUTE', for correct labelling of graph
            :type joint_type: str
            '''
            
            # Create matplotlib fig and axes
            fig, axs = plt.subplots(3)
            axs[0].plot(time, theta)
            axs[0].set_title('Position')
            axs[1].plot(time, theta_dot)
            axs[1].set_title('Velocity')
            axs[2].plot(time, theta_dot_dot)
            axs[2].set_title('Acceleration')
            
            # Determine graph lables based on 'joint_type' - defaults to theta
            ylbl1 = 'Theta'
            ylbl2 = 'Theta Dot'
            ylbl3 = 'Theta Dot Dot'
            fig.suptitle('Positon, Velocity and Acceleration of Revolute Joints')
            if joint_type.upper() == 'PRISMATIC':
                ylbl1 = 'D'
                ylbl2 = 'D Dot'
                ylbl3 = 'D Dot Dot'
                fig.suptitle('Positon, Velocity and Acceleration of Revolute Joints')
            
            axs[0].set(ylabel=ylbl1)
            axs[1].set(ylabel=ylbl2)
            axs[2].set(ylabel=ylbl3)
            
            # xlables are independent of joint type
            for ax in axs.flat:
                ax.set(xlabel='t')
                
            fig.tight_layout()
            
            fig.show()
            plt.show()
                        
        def __create_dot_dataframe(self): #private method
            '''
            Private method to create dataframe for visulisation of acc/vel calcs
            
            :return dynamics_table: Pandas dataframe of dynamics calcs (UnSubbed)
            :type DataFrame
            '''
            
            self.dynamics_data = {
                "Omega": self.solved_omega_values,
                "Omega_Dot": self.solved_omega_dot_values,
                "V_Dot": self.solved_v_dot_values,
                "V_Centre_Dot": self.solved_v_centre_dot_values,
                "Force": self.solved_F_values,
                "Torque": self.solved_N_values
            }
            link_labels = ['Base', 'Link 1', 'Link 2', 'Link 3', 'Link 4', 'Link 5', 'Link 6']
            self.dynamics_table = pd.DataFrame(self.dynamics_data, index=link_labels)
            return self.dynamics_table
        
        def __create_totals_dataframe(self): #private method
            '''
            Private method to create dataframe for visulisation of F/N/Tau totals
            
            :return totals_data: Pandas dataframe of F/N totals and Joint Tau
            :type DataFrame
            '''
            
            self.totals_data = {
                "Total Link Force": self.solved_F_totals,
                "Total Link Moment": self.solved_F_totals,
                "Joint Torque": self.solved_tau_values
            }
            link_labels = ["Joint 6", "Joint 5", "Joint 4", "Joint 3", "Joint 2", "Joint 1", "Base"]
            self.totals_table = pd.DataFrame(self.totals_data, index=link_labels)
            return self.totals_table
        
        def __solve_matrices(self):
            '''
            Private method to substitute values into new lists, removed from class __init__ method
            '''
            
            self.solved_omega_values = self.__subsitute_values(self.omega_values)
            self.solved_omega_dot_values = self.__subsitute_values(self.omega_dot_values)
            self.solved_v_dot_values = self.__subsitute_values(self.v_dot_values)
            self.solved_v_centre_dot_values = self.__subsitute_values(self.v_centre_dot_values)
            self.solved_F_values = self.__subsitute_values(self.F_values)
            self.solved_N_values = self.__subsitute_values(self.N_values)
            self.solved_F_totals = self.__subsitute_values(self.F_totals)
            self.solved_N_totals = self.__subsitute_values(self.N_totals)
            self.solved_tau_values = self.__subsitute_values(self.joint_torque_totals)
            
        def __subsitute_values(self, value_list: 'list[float]') -> 'list[float]':
            '''
            Private method to handle the logic of value substituion using the Sympy library
            
            :param value_list: List of values for the sympy subsititution to happen to
            :type value_list: list[float]
            
            :return target_list: New list for the results of subsitution of the value_list
            :type target_list: list[float]
            '''
            
            target_list = []
            for x in value_list:
                target_list.append(x.subs({ # Substitude in joint values
                self.L0: 0.10, 
                self.L1: 0.20,
                self.L2: 0.30,
                self.L3: 0.30,
                self.L4: 0.10,
                self.L5: 0.05,
                self.THETA1: self.EMERGANCY_STOP_POSE[0],
                self.D2: self.EMERGANCY_STOP_POSE[1],
                self.D3: self.EMERGANCY_STOP_POSE[2],
                self.THETA4: self.EMERGANCY_STOP_POSE[3],
                self.THETA5: self.EMERGANCY_STOP_POSE[4],
                self.THETA6: self.EMERGANCY_STOP_POSE[5]
            }))
            return target_list


if __name__ == "__main__":

    #Intialise instances of classes for both RBT and Sympy Solutions
    print('Initialising instaces of RPPRRR class...')
    manipulator = RPPRRRManipulator()
    sympy_manipulator = RPPRRRManipulator.RPPRRRManipulatorSympy()
    
    #Print DH table
    print('Manipulator DH Table:\n')
    print(manipulator.DH_TABLE)
    
    print('Transfrom as function of joint angles:\n')
    print(np.array(sympy_manipulator.TB_T))
    
    #Print instance of manipulator to check DH and Qlim Values
    print(manipulator)
    
    #STEP 1: Forward Kinematics (FK)
    #Calcualte forward kinematic solution using the joint angles specified in 'Step 1: Forward Kinematics (FK)'
    # Robotics Toolbox solution
    print('\nStep 1: Forward Kinematics (FK).\nCalculating forward kinematics from the joint angles provided using the Robotics Toolbox model.\n')
    manipulator_fk = manipulator.forward_kinematics(STEP1_FK_JOINT_ANGLES)
    
    # Sympy Solution - Calculated within the initialisation of the class
    print('Calculating forward kinematics using the sympy solution.\n')
    sympy_manipulator_fk = np.round(np.array(sympy_manipulator.TB_T_FK).astype(np.float64), 2)
    
    # Print both RBT and Sympy solutions
    print(f"RBT FK Solution:\n {manipulator_fk}\nSympy FK Solution:\n {sympy_manipulator_fk}\n") #Display solutions to the user
    
    #Calulate inverse kinematic solution using the forward kinematic transform to test error
    #Robotics Toolbox Solution
    print('\nCalculating inverse kinematics of RBT forward kinematic solution to test error')
    manipulator_ik = manipulator.ikine_LM(manipulator_fk)
    rbt_fk_ik_error = manipulator.ik_error(manipulator_fk, manipulator_ik)
    
    # Sympy solution, compared to RBT Solution
    print('\nCalculating inverse kinematics of sympy solution to test error')
    sympy_joint_angles = manipulator.ikine_LM(np.array(sympy_manipulator.TB_T_FK).astype(np.float64))
    sympy_fk_ik_error = manipulator.ik_error(manipulator_fk, sympy_joint_angles)
    
    #Display difference in errors to the user
    print(f'\nRBT IK Error: {rbt_fk_ik_error}\nSympy FK Error: {sympy_fk_ik_error}')




    # STEP 2: Inverse Kinematics (IK)
    # Now that we have confirmed a very small error in our IK we can use it to solve the transforms specified in 'Step 2: Inverse Kinematics (IK)'
    # Solve each transform specified in the breif. Q limits applied to manipualtor at initialisation
    print('\n\n\n\nStep 2: Inverse Kinematics (IK)\nCalculating inverse kinematics of a variety of transformation specified Step 2')
    print('Using a method to handle all aspects of each transform.\nFor each transform, will calculate all possible inverse solutions, display them and the solution with the lowest error, and plot joint angles.')
    best_sol_1, error_1 = manipulator.step3_inverse_kinematic_solver(STEP2_IK_TRANSFORM_1)
    best_sol_2, error_2 = manipulator.step3_inverse_kinematic_solver(STEP2_IK_TRANSFORM_2)
    best_sol_3, error_3 = manipulator.step3_inverse_kinematic_solver(STEP2_IK_TRANSFORM_3)
    
    
    
    
    
    #STEP 3: Velocity and Static Force
    #Calculate Jacobian, Velocities and Static Forces, print the Linear and Angular velocities respetively
    print('\n\n\n\nStep 3: Velocities and Static Forces (of the manipulator)')
    print('Calculating the jacobian, linear velocities and angular velcoties for the following joint angles and velocities:\n')
    print(f'Joint Angles:\n{STEP3_JOINT_ANGLES}\nJoint Velocities:\n{STEP3_JOINT_VELOCITIES}\n')
    jacobian, linear_velocities, angular_velocities = manipulator.joint_velocities(joint_angles = STEP3_JOINT_ANGLES, joint_velocities=STEP3_JOINT_VELOCITIES)
    print('\nJacobian Operator: \n', jacobian)
    print(f'\n\nFor the the given joint velocities: {STEP3_JOINT_VELOCITIES}, the resultant velocities on the tool frame are as follows:\n')
    print("Linear Velocities [X, Y, Z]:\n", linear_velocities)
    print("Angular Velocities [X, Y, Z]:\n",angular_velocities)
    
    #Find the torque acting on each joint wuith a point mass of 0.2kg at the tool frame, at the specific transform
    print('\nCalculating the joint torque acting on each joint if a point mass of 0.2 is acting on a manipulator with the transform:\n', STEP3_STATIC_FORCE_TRANSFORM)
    print('\nCalculating the joint angles required for the specified transform using inverse kinematics:\n')
    joint_torques = manipulator.static_torques(mass=0.2, g=9.8, transform=STEP3_STATIC_FORCE_TRANSFORM)
    print('\nJoint torques: ', joint_torques)
    
    
    
    
    
    #Step 4: Manipulator Dynamics
    print('\n\n\n\nStep 4: Manipulator Dynamics')
    print('\nCalculate the required torques in each joint to bring the manipualtor to an emergancy stop.')
    print('\nThe manipulator is specified to be carrying a load of 0.2KG and is in the following pose:\n', sympy_manipulator.EMERGANCY_STOP_POSE)
    print('\nThe manipuattor is specified to be moving at the following joint velocities at the time of the emergancy stop:\n', sympy_manipulator.E_STOP_JOINT_VELOCITIES)
    print('\nCalculating equations for manipulator dynmaics...')
    if input('\n\n\n\nDisplay all unsubsituted equations? Y/N ').upper() == 'Y':
        sympy_manipulator.display_all_equations()
    if input('\n\n\n\nDisplay table of values for manipulator dynamics? Y/N ').upper() == 'Y':
        print('Table of Dynmaics values for each Link:\n')
        print(sympy_manipulator.dynamics_table)
        
        print('\n\nTable of Force, Moment totals as well as Joint Torques:\n')
        print(sympy_manipulator.totals_table)
        print('\n Base Joint Torque should be 0 as the base is not a joint')
    
    
    
    
    
    # Step 5: Trajectory Generation
    # Calculate cubic coefficents for both Revolute and Prismatic joint
    print('\n\n\n\nStep 4: Trajectory Generation')
    # Part a
    print('\n\n\n\nPart A: Cubic Polynomial')
    print('Calculating cubic polynomial coeffiecients for both Revolute and D1, D2 Prismatic joint...')
    revolute_cubic = sympy_manipulator.cubic_polynomial_TG(theta_0=100, theta_f=20, t_f=1)
    d1_cubic = sympy_manipulator.cubic_polynomial_TG(theta_0=0.1, theta_f=0.5, t_f=1)
    d2_cubic = sympy_manipulator.cubic_polynomial_TG(theta_0=0.05, theta_f=0.1, t_f=1)
    print(f'Coeffecients for revolute:\n {revolute_cubic}\nd1 prismatic:\n{d1_cubic}\nd2 prismatic:\n{d2_cubic}')
    
    print('Generating plots for Positon, Velocitiy, and Acceleration of each joint\n')
    revolute_plots_cubic = sympy_manipulator.generate_polynomial_plot(revolute_cubic, 1, joint_type='revolute')
    d1_plots_cubic = sympy_manipulator.generate_polynomial_plot(d1_cubic, 1, joint_type='d1')
    d2_plots_cubic = sympy_manipulator.generate_polynomial_plot(d2_cubic, 1, joint_type='d2')
    print('Displaying plot for Revolute, D1, and D2 Cubic soltuions to Position, Velocity and Acceleration')
    
    print('\n\nThe position plots follow the expected path for a cubic function of time, with a smooth blend from the starting theta value to the final')
    print('Similarly the velocity depicts an accurae dereiviatve of teh position, while the acceleration further demonstrates the impracitaility due to the immense acceleration values displayed.')
    sympy_manipulator.display_polynomials(revolute_plots_cubic, d1_plots_cubic, d2_plots_cubic)
    

    # Calculate quintic coefficents for both Revolute and Pristmatic joints
    # part b
    print('\n\n\n\nPart B: Quintic Polynomial')
    print('Calculating quintic polynomial coeffiecients for both Revolute and D1, D2 Prismatic joint...')
    revolute_quintic = sympy_manipulator.quintic_polynomial_TG(theta_0=100, theta_f=20, t_f=1)
    d1_quintic = sympy_manipulator.quintic_polynomial_TG(theta_0=0.1, theta_f=0.5, t_f=1)
    d2_quintic = sympy_manipulator.quintic_polynomial_TG(theta_0=0.05, theta_f=0.1, t_f=1)
    print(f'Coeffecients for revolute:\n {revolute_quintic}\nd1 prismatic:\n{d1_quintic}\nd2 prismatic:\n{d2_quintic}')
    
    print('Generating plots for Positon, Velocitiy, and Acceleration of each joint\n')
    revolute_plots_quintic = sympy_manipulator.generate_polynomial_plot(revolute_quintic, 1, joint_type='revolute', order=5)
    d1_plots_quintic = sympy_manipulator.generate_polynomial_plot(d1_quintic, 1, joint_type='d1', order=5)
    d2_plots_quintic = sympy_manipulator.generate_polynomial_plot(d2_quintic, 1, joint_type='d2', order=5)
    print('Displaying plot for Revolute, D1, and D2 quintic soltuions to Position, Velocity and Acceleration')
    
    print('\n\nThe position plots follow the expected path for a quintic function of time, with a smooth blend from the starting theta value to the final')
    print('However, it is in the velocity and accerelaion where the changes are more noticable. The velocity now incorporates a smooth blend region at the beginning and end.')
    print('This smooth blend is expected and more suitable to a manipulator in practice')
    print('This is further demonstrated in the acceleration which now follows a sinusoidal curve creating a smoother acceleration curve')
    sympy_manipulator.display_polynomials(revolute_plots_quintic, d1_plots_quintic, d2_plots_quintic)
    
    
    print('Disaplying Cubic and Quantic plots side by side.')
    print('Legend')
    print('Cubic Velocity: Blue\nCubic Velocity: Green\nCubic Acceleration: Red\nCubic Angular Jerk: Yellow')
    print('Quintic Velocity: Dodger Blue\nQuintic Velocity: Magenta\nQuintic Acceleration: Dark Violet\nQuintic Angular Jerk: Orange')
    print('When comparing Cubic and Quintic side by side, the prominence of the improved smoothing of the higher order quintic become significantly more pronouced.')
    sympy_manipulator.display_all_polynomial_plots(revolute_plots_cubic, revolute_plots_quintic, d1_plots_cubic, d1_plots_quintic, d2_plots_cubic, d2_plots_quintic)
    
    
    # Calculating two cubic polynomials with via point
    # part c
    print('\n\n\n\nPart C: Two Cubic Polynomials with via point')
    print('\n\n\n\nDisplaying plot for two cubic polynomials with via point for a revolute joint, where the starting theta is 100, via point is 50 and final theta is 20:\n')
    sympy_manipulator.calculate_polynomials_via_point(theta_1=100, theta_2=60, theta_3=20, acc=50)
    
    print('\n\nDisplaying plot for two cubic polynomials with via point for a prismatic joint, where the starting d is 0.05, via point is 0.075 and final d is 0.10:\n')
    sympy_manipulator.calculate_polynomials_via_point(theta_1=0.05, theta_2=0.075, theta_3=0.10, acc=50, joint_type='PRISMATIC')
    
    
    # Linear Interpolation in cartesian space
    # part d
    print('\n\n\n\nPart D: Linear Interpolation in Cartesian Space')
    manipulator.calculate_cartesian_point_to_point(STEP5_PARTD_CURRENT_POSE, STEP5_PARTD_NEW_POSE)