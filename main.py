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
    
    #TODO add statemnts tracking where you are in the code, the part of the coursework being asseseed and explainign the values presented for the manipualtor dynamics
    #TODO change Inverse kinematic fucntions to only display the number of genuine possible soltuions, found in textbox based on a0=a3=a6 etc. or something

    #Intialise instances of classes for both RBT and Sympy Solutions
    manipulator = RPPRRRManipulator()
    sympy_manipulator = RPPRRRManipulator.RPPRRRManipulatorSympy()
    
    # #Print instance of manipulator to check DH and Qlim Values
    # print(manipulator)
    
    # #STEP 1: Forward Kinematics (FK)
    # #Calcualte forward kinematic solution using the joint angles specified in 'Step 1: Forward Kinematics (FK)'
    # # Robotics Toolbox solution
    # manipulator_fk = manipulator.forward_kinematics(STEP1_FK_JOINT_ANGLES)
    
    # # Sympy Solution - Calculated within the initialisation of the class
    # sympy_manipulator_fk = np.round(np.array(sympy_manipulator.TB_T_FK).astype(np.float64), 2)
    
    # print(f"RBT FK Solution:\n {manipulator_fk}\nSympy FK Solution:\n {sympy_manipulator_fk}\n") #Display solutions to the user
    
    # #Calulate inverse kinematic solution using the forward kinematic transform to test error
    # #Robotics Toolbox Solution
    # manipulator_ik, invalid_ik = manipulator.inverse_kinematics(manipulator_fk)
    # rbt_fk_ik_error = manipulator.ik_error(manipulator_fk, manipulator_ik[0])
    
    # # Sympy solution, compared to RBT Solution
    # sympy_joint_angles = manipulator.ikine_LM(np.array(sympy_manipulator.TB_T_FK).astype(np.float64))
    # sympy_fk_ik_error = manipulator.ik_error(manipulator_fk, sympy_joint_angles)
    
    # print(f'\nRBT IK Error: {rbt_fk_ik_error}\nSympy FK Error: {sympy_fk_ik_error}')

    # # STEP 2: Inverse Kinematics (IK)
    # # Now that we have confirmed a very small error in our IK we can use it to solve the transforms specified in 'Step 2: Inverse Kinematics (IK)'
    # # Solve each transform specified in the breif. Q limits applied to manipualtor at initialisation
    # best_sol_1, error_1 = manipulator.step3_inverse_kinematic_solver(STEP2_IK_TRANSFORM_1)
    # best_sol_2, error_2 = manipulator.step3_inverse_kinematic_solver(STEP2_IK_TRANSFORM_2)
    # best_sol_3, error_3 = manipulator.step3_inverse_kinematic_solver(STEP2_IK_TRANSFORM_3)
    
    # #STEP 3: Velocity and Static Force
    # #Calculate Jacobian, Velocities and Static Forces, print the Linear and Angular velocities respetively
    # jacobian, linear_velocities, angular_velocities = manipulator.joint_velocities(joint_angles = STEP3_JOINT_ANGLES, joint_velocities=STEP3_JOINT_VELOCITIES)
    # print('\nJacobian Operator: \n', jacobian)
    # print(f'\nFor the the given joint velocities: {STEP3_JOINT_VELOCITIES}, the resultant velocities on the tool frame are as follows:')
    # print("Linear Velocities [X, Y, Z]:", linear_velocities)
    # print("Angular Velocities [X, Y, Z]:",angular_velocities)
    
    # #Find the torque acting on each joint wuith a point mass of 0.2kg at the tool frame, at the specific transform
    # joint_torques = manipulator.static_torques(mass=0.2, g=9.8, transform=STEP3_STATIC_FORCE_TRANSFORM)
    # print('Joint torques: ', joint_torques)
    
    # #Step 4: Manipulator Dynamics
    # if input('\n\n\n\n\nDisplay all unfiltered equations? Y/N ').upper() == 'Y':
    #     sympy_manipulator.display_all_equations()
    # if input('\n\n\n\nDisplay table of values for manipulator dynamics? Y/N').upper() == 'Y':
    #     print('Table of Dynmaics values for each Link:\n')
    #     print(sympy_manipulator.dynamics_table)
        
    #     print('\n\nTable of Force, Moment totals as well as Joint Torques:\n')
    #     print(sympy_manipulator.totals_table)
    #     print('\n Base Joint Torque should be 0 as the base is not a joint')
    
    # revolute_cubic = sympy_manipulator.cubic_polynomial_TG(theta_0=100, theta_f=20, t_f=1)
    # d1_cubic = sympy_manipulator.cubic_polynomial_TG(theta_0=0.1, theta_f=0.5, t_f=1)
    # d2_cubic = sympy_manipulator.cubic_polynomial_TG(theta_0=0.05, theta_f=0.1, t_f=1)
    # revolute_plots_cubic = sympy_manipulator.generate_polynomial_plot(revolute_cubic, 1, joint_type='revolute')
    # d1_plots_cubic = sympy_manipulator.generate_polynomial_plot(d1_cubic, 1, joint_type='d1')
    # d2_plots_cubic = sympy_manipulator.generate_polynomial_plot(d2_cubic, 1, joint_type='d2')
    # # sympy_manipulator.display_polynomials(revolute_plots_cubic, d1_plots_cubic, d2_plots_cubic)

    # revolute_quintic = sympy_manipulator.quintic_polynomial_TG(theta_0=100, theta_f=20, t_f=1)
    # d1_quintic = sympy_manipulator.quintic_polynomial_TG(theta_0=0.1, theta_f=0.5, t_f=1)
    # d2_quintic = sympy_manipulator.quintic_polynomial_TG(theta_0=0.05, theta_f=0.1, t_f=1)
    # revolute_plots_quintic = sympy_manipulator.generate_polynomial_plot(revolute_quintic, 1, joint_type='revolute', order=5)
    # d1_plots_quintic = sympy_manipulator.generate_polynomial_plot(d1_quintic, 1, joint_type='d1', order=5)
    # d2_plots_quintic = sympy_manipulator.generate_polynomial_plot(d2_quintic, 1, joint_type='d2', order=5)
    # # sympy_manipulator.display_polynomials(revolute_plots_quintic, d1_plots_quintic, d2_plots_quintic)
    
    # sympy_manipulator.display_all_polynomial_plots(revolute_plots_cubic, revolute_plots_quintic, d1_plots_cubic, d1_plots_quintic, d2_plots_cubic, d2_plots_quintic)
    
    # manipulator.generate_trapezoida_velocity(theta_s=100, theta_f=20, t_r=4)