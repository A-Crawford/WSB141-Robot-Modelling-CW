import numpy as np
from rpprrr_manipulator import RPPRRRManipulator

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
        [-0.8419, 0, 0.5396, 0.7684],
        [0, -1, 0, -0.25],
        [0.5396, 0, -0.8419, 1.925],
        [0, 0, 0, 1]
    ]
)

STEP2_IK_TRANSFORM_3 = np.array(
    [
        [-0.0023, -1, 0.1, -0.257],
        [-0.002, 1, 0.2, -0.299],
        [-1, 0, -1, 3.34],
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

if __name__ == "__main__":

    #Intialise instances of classes for both RBT and Sympy Solutions
    manipulator = RPPRRRManipulator()
    sympy_manipulator = RPPRRRManipulator.RPPRRRManipulatorSympy()
    
    #Print instance of manipulator to check DH and Qlim Values
    print(manipulator)
    
    # #STEP 1: Forward Kinematics (FK)
    # #Calcualte forward kinematic solution using the joint angles specified in 'Step 1: Forward Kinematics (FK)'
    # # Robotics Toolbox solution
    # manipulator_fk = manipulator.forward_kinematics(STEP1_FK_JOINT_ANGLES)
    
    
    # # Sympy Solution - Calculated within the initialisation of the class
    # sympy_manipulator_fk = np.round(np.array(sympy_manipulator.TB_T_FK).astype(np.float64), 2)
    
    # print(f"RBT FK Solution:\n {manipulator_fk}\nSympy FK Solution:\n {sympy_manipulator_fk}") #Display solutions to the user
    
    # #Calulate inverse kinematic solution using the forward kinematic transform to test error
    # #Robotics Toolbox Solution
    # manipulator_ik = manipulator.inverse_kinematics(manipulator_fk)
    # rbt_fk_ik_error = np.linalg.norm(manipulator_fk - manipulator.fkine(manipulator_ik.q))
    
    # # Sympy solution, compared to RBT Solution
    # sympy_joint_angles = manipulator.ikine_LM(np.array(sympy_manipulator.TB_T_FK).astype(np.float64))
    # sympy_fk_ik_error = np.linalg.norm(np.array(sympy_manipulator.TB_T_FK).astype(np.float64) - manipulator.fkine(sympy_joint_angles.q))
    
    # print(f'\nRBT IK Error: {rbt_fk_ik_error}\nSympy FK Error: {sympy_fk_ik_error}')

    # STEP 2: Inverse Kinematics (IK)
    # Now that we have confirmed a very small error in our IK we can use it to solve the transforms specified in 'Step 2: Inverse Kinematics (IK)'
    # Solve each transform specified in the breif. Q limits applied to manipualtor at initialisation
    ik_sol_1 = manipulator.inverse_kinematics(STEP2_IK_TRANSFORM_1, display=True)
    # ik_sol_1_error = manipulator.ik_error(STEP2_IK_TRANSFORM_1, ik_sol_1)
    # print('IK Error for Transform 1: ', ik_sol_1_error)
    
    ik_sol_2 = manipulator.inverse_kinematics(STEP2_IK_TRANSFORM_2, display=True)
    # ik_sol_2_error = manipulator.ik_error(STEP2_IK_TRANSFORM_2, ik_sol_2)
    # print('IK Error for Transform 2: ', ik_sol_2_error)
    
    ik_sol_3 = manipulator.inverse_kinematics(STEP2_IK_TRANSFORM_3, display=True)
    # ik_sol_3_error = manipulator.ik_error(STEP2_IK_TRANSFORM_3, ik_sol_3)
    # print('IK Error for Transform 3: ', ik_sol_3_error)
    
    # print(ik_sol_3)
    
    # #STEP 3: Velocity and Static Force
    # #Calculate Jacobian, Velocities and Static Forces, print the Linear and Angular velocities respetively
    # jacobian, linear_velocities, angular_velocities = manipulator.joint_velocities(joint_angles = STEP3_JOINT_ANGLES, joint_velocities=STEP3_JOINT_VELOCITIES)
    # print('Jacobian Operator: \n', jacobian)
    # print(f'\nFor the the given joint velocities: {STEP3_JOINT_VELOCITIES}, the resultant velocities on the tool frame are as follows:')
    # print("Linear Velocities [X, Y, Z]:", linear_velocities)
    # print("Angular Velocities [X, Y, Z]:",angular_velocities)
    
    # #Find the torque acting on each joint wuith a point mass of 0.2kg at the tool frame, at the specific transform
    # joint_torques = manipulator.static_torques(mass=0.2, g=9.8, transform=STEP3_STATIC_FORCE_TRANSFORM)
    # print('Joint torques: ', joint_torques)
