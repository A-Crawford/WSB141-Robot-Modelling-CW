import numpy as np
import matplotlib.pyplot as plt #Main support Library for plotting Figures
from mpl_toolkits.mplot3d import Axes3D # Necessary for 3D plot support

L0 = 0.1
L1 = 0.2
L2 = 0.3
L3 = 0.3
L4 = 0.1
L5 = 0.05
D2 = 0.5
D3 = 0
T1 = 0
T4 = 0
T5 = 0
T6 = 0

DEG = np.radians(90)

T0_B = np.array([
    [np.cos(0), -np.sin(0), 0, 0],
    [(np.sin(0)*np.cos(0)), (np.cos(0)*np.cos(0)), -np.sin(0), -(np.sin(0)*L0)],
    [(np.sin(0)*np.sin(0)), (np.cos(0)*np.sin(0)), np.cos(0), (np.cos(0)*L0)],
    [0, 0, 0, 1]
])

TB_1 = np.array([
    [np.cos(T1), -np.sin(T1), 0, 0],
    [(np.sin(T1)*np.cos(0)), (np.cos(T1)*np.cos(0)), -np.sin(0), -(np.sin(0)*0)],
    [(np.sin(T1)*np.sin(0)), (np.cos(T1)*np.sin(0)), np.cos(0), (np.cos(0)*0)],
    [0, 0, 0, 1]
])

T1_2 = np.array([
    [np.cos(0), -np.sin(0), 0, 0],
    [(np.sin(0)*np.cos(0)), (np.cos(0)*np.cos(0)), -np.sin(0), -(np.sin(0)*D2)],
    [(np.sin(0)*np.sin(0)), (np.cos(0)*np.sin(0)), np.cos(0), (np.cos(0)*D2)],
    [0, 0, 0, 1]
])

T2_3 = np.array([
    [np.cos(0), -np.sin(0), 0, 0],
    [(np.sin(0)*np.cos(DEG)), (np.cos(0)*np.cos(DEG)), -np.sin(DEG), -(np.sin(DEG)*D3)],
    [(np.sin(0)*np.sin(DEG)), (np.cos(0)*np.sin(DEG)), np.cos(DEG), (np.cos(DEG)*D3)],
    [0, 0, 0, 1]
])

T3_4 = np.array([
    [np.cos(T4), -np.sin(T4), 0, 0],
    [(np.sin(T4)*np.cos(0)), (np.cos(T4)*np.cos(0)), -np.sin(0), -(np.sin(0)*L1)],
    [(np.sin(T4)*np.sin(0)), (np.cos(T4)*np.sin(0)), np.cos(0), (np.cos(0)*L1)],
    [0, 0, 0, 1]
])

T4_5 = np.array([
    [np.cos(T5), -np.sin(T5), 0, L2],
    [(np.sin(T5)*np.cos(0)), (np.cos(T5)*np.cos(0)), -np.sin(0), -(np.sin(0)*L5)],
    [(np.sin(T5)*np.sin(0)), (np.cos(T5)*np.sin(0)), np.cos(0), (np.cos(0)*L5)],
    [0, 0, 0, 1]
])

T5_6 = np.array([
    [np.cos(T6), -np.sin(T6), 0, 0],
    [(np.sin(T6)*np.cos(DEG)), (np.cos(T6)*np.cos(DEG)), -np.sin(DEG), -(np.sin(DEG)*L3)],
    [(np.sin(T6)*np.sin(DEG)), (np.cos(T6)*np.sin(DEG)), np.cos(DEG), (np.cos(DEG)*L3)],
    [0, 0, 0, 1]
])

T6_T = np.array([
    [np.cos(0), -np.sin(0), 0, 0],
    [(np.sin(0)*np.cos(0)), (np.cos(0)*np.cos(0)), -np.sin(0), -(np.sin(0)*L4)],
    [(np.sin(0)*np.sin(0)), (np.cos(0)*np.sin(0)), np.cos(0), (np.cos(0)*L4)],
    [0, 0, 0, 1]
])

T0_T = T0_B.dot(TB_1).dot(T1_2).dot(T2_3).dot(T3_4).dot(T4_5).dot(T5_6).dot(T6_T)
T1_6 = TB_1.dot(T1_2).dot(T2_3).dot(T3_4).dot(T4_5).dot(T5_6)
print(np.round(T0_T, 2))

# Create Figure Plot for Transformation matrix frame T0_1, T1_2, T2_3
fig = plt.figure() # Creat Figure object
ax = fig.add_subplot(111, projection='3d') #Projects the Plot as 3-Dimentional Figure
ax.set_xlim([-1, 1]) #Set the x-axis limits
ax.set_ylim([-1, 1]) #Set y-axis limits
ax.set_zlim([-1, 1])
ax.set_xlabel('X') #Label the X-axis
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# # Plotting Transformation Matrix T0_1
# origin = np.zeros((4, 4)) #Create 4x4 Matrix of Zeros which signify the point of origin for the
# origin_R=origin[:3,:3] #Sets the Rotation of the Reference Frame
# translation_01 = T0_B[:3, 3] #Sets the Translation of the Reference Frame
# rotation_01 = T0_B[:3, :3].astype(float)
# #Origin of Fixed axis reference Frame x,y,z = [0,0,0] and this frame has no rotational component, so we also assign
# ax.quiver(origin[:3,0], origin[:3,1], origin[:3,2], origin_R[0],origin_R[1],origin_R[2],color='k')
# ax.quiver(translation_01[0], translation_01[1], translation_01[2], rotation_01[0, 0], rotation_01[1, 0], rotation_01[2,
# 0],color='r')
# ax.quiver(translation_01[0], translation_01[1], translation_01[2], rotation_01[0, 1], rotation_01[1, 1], rotation_01[2,
# 1],color='g')
# ax.quiver(translation_01[0], translation_01[1], translation_01[2], rotation_01[0, 2], rotation_01[1, 2], rotation_01[2, 2],
# color='b')
# # Add frame name {T0_1}
# frame_name = ' {T0_B}'
# ax.text(translation_01[0], translation_01[1], translation_01[2], frame_name, color='k', fontsize=7)

# T0_2=TB_1.dot(T1_2)
# translation_12 = T0_2[:3, 3]
# rotation_12 = T0_2[:3, :3].astype(float)
# ax.quiver(translation_12[0], translation_12[1], translation_12[2], rotation_12[0, 0], rotation_12[1, 0],
# rotation_12[2, 0], color='r')
# ax.quiver(translation_12[0], translation_12[1], translation_12[2], rotation_12[0, 1], rotation_12[1, 1],
# rotation_12[2, 1], color='g')
# ax.quiver(translation_12[0], translation_12[1], translation_12[2], rotation_12[0, 2], rotation_12[1, 2],
# rotation_12[2, 2], color='b')
# # Add frame name {T1_2}
# frame_name = ' {TB_1}'
# ax.text(translation_12[0], translation_12[1], translation_12[2], frame_name, color='k', fontsize=7)

# T0_3=T0_2.dot(T2_3)
# translation_23 = T0_3[:3, 3]
# rotation_23 = T0_3[:3, :3].astype(float)
# ax.quiver(translation_23[0], translation_23[1], translation_23[2], rotation_23[0, 0], rotation_23[1, 0],
# rotation_23[2, 0], color='r')
# ax.quiver(translation_23[0], translation_23[1], translation_23[2], rotation_23[0, 1], rotation_23[1, 1],
# rotation_23[2, 1], color='g')
# ax.quiver(translation_23[0], translation_23[1], translation_23[2], rotation_23[0, 2], rotation_23[1, 2],
# rotation_23[2, 2], color='b')
# # Add frame name {T2_3}
# frame_name = '{T2_3}'
# ax.text(translation_23[0], translation_23[1], translation_23[2], frame_name, color='k', fontsize=7)

# T0_4=T0_3.dot(T3_4)
# translation_23 = T0_4[:3, 3]
# rotation_23 = T0_4[:3, :3].astype(float)
# ax.quiver(translation_23[0], translation_23[1], translation_23[2], rotation_23[0, 0], rotation_23[1, 0],
# rotation_23[2, 0], color='r')
# ax.quiver(translation_23[0], translation_23[1], translation_23[2], rotation_23[0, 1], rotation_23[1, 1],
# rotation_23[2, 1], color='g')
# ax.quiver(translation_23[0], translation_23[1], translation_23[2], rotation_23[0, 2], rotation_23[1, 2],
# rotation_23[2, 2], color='b')
# # Add frame name {T2_3}
# frame_name = '{T3_4}'
# ax.text(translation_23[0], translation_23[1], translation_23[2], frame_name, color='k', fontsize=7)

# T0_5=T0_4.dot(T4_5)
# translation_23 = T0_5[:3, 3]
# rotation_23 = T0_5[:3, :3].astype(float)
# ax.quiver(translation_23[0], translation_23[1], translation_23[2], rotation_23[0, 0], rotation_23[1, 0],
# rotation_23[2, 0], color='r')
# ax.quiver(translation_23[0], translation_23[1], translation_23[2], rotation_23[0, 1], rotation_23[1, 1],
# rotation_23[2, 1], color='g')
# ax.quiver(translation_23[0], translation_23[1], translation_23[2], rotation_23[0, 2], rotation_23[1, 2],
# rotation_23[2, 2], color='b')
# # Add frame name {T2_3}
# frame_name = '{T4_5}'
# ax.text(translation_23[0], translation_23[1], translation_23[2], frame_name, color='k', fontsize=7)

# T0_6=T0_5.dot(T5_6)
# translation_23 = T0_6[:3, 3]
# rotation_23 = T0_6[:3, :3].astype(float)
# ax.quiver(translation_23[0], translation_23[1], translation_23[2], rotation_23[0, 0], rotation_23[1, 0],
# rotation_23[2, 0], color='r')
# ax.quiver(translation_23[0], translation_23[1], translation_23[2], rotation_23[0, 1], rotation_23[1, 1],
# rotation_23[2, 1], color='g')
# ax.quiver(translation_23[0], translation_23[1], translation_23[2], rotation_23[0, 2], rotation_23[1, 2],
# rotation_23[2, 2], color='b')
# # Add frame name {T2_3}
# frame_name = '{T5_6}'
# ax.text(translation_23[0], translation_23[1], translation_23[2], frame_name, color='k', fontsize=7)

# T0_T=T0_6.dot(T6_T)
# translation_23 = T0_T[:3, 3]
# rotation_23 = T0_T[:3, :3].astype(float)
# ax.quiver(translation_23[0], translation_23[1], translation_23[2], rotation_23[0, 0], rotation_23[1, 0],
# rotation_23[2, 0], color='r')
# ax.quiver(translation_23[0], translation_23[1], translation_23[2], rotation_23[0, 1], rotation_23[1, 1],
# rotation_23[2, 1], color='g')
# ax.quiver(translation_23[0], translation_23[1], translation_23[2], rotation_23[0, 2], rotation_23[1, 2],
# rotation_23[2, 2], color='b')
# # Add frame name {T2_3}
# frame_name = '{T6_T}'
# ax.text(translation_23[0], translation_23[1], translation_23[2], frame_name, color='k', fontsize=7)

# plt.show()