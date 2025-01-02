import pybullet as p
import numpy as np
import time
from moviepy.editor import ImageSequenceClip
from matplotlib import pyplot as plt
from robotiq_state_machine import GripperFingerStateMachine as GSM
import os

def make_video(images, output_video_file):
    clip = ImageSequenceClip(list(images), fps=60)
    clip.write_videofile(output_video_file, codec="libx264")

def create_and_save_plot(data, title, xlabel, ylabel, filename, legend_labels=None):
    plt.figure()
    if data.ndim > 1:
        for i in range(data.shape[1]):
            plt.plot(data[:, i], label=legend_labels[i] if legend_labels else f'Dimension {i}')
        plt.legend()
    else:
        plt.plot(data)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(filename)
    plt.close()

saved_dir = "data/withtarget2"

# Connect to PyBullet
p.connect(p.DIRECT)

# Set gravity
p.setGravity(0, 0, 0)

# Load URDFs
gripper = p.loadURDF("urdf/robotiq.urdf", [0, 0, 1], p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=True)
target = p.loadURDF("urdf/block.urdf", [0, 0.155, 1], p.getQuaternionFromEuler([0, np.pi/2, 0]))

# Get initial positions and orientations
target_pose = p.getBasePositionAndOrientation(target)
gripper_pose = p.getBasePositionAndOrientation(gripper)

# Create finger state machines
fingers = [GSM(gripper, target, i) for i in range(1, 4)]

# Data recording
num_steps = 400
num_fingers = 3
lat_contact_data = np.zeros((num_steps, num_fingers, 3))  # Storing lateral contact for each finger in each direction
total_contact_data = np.zeros((num_steps, num_fingers))   # Storing total contact for each finger
target_vel_data = np.zeros((num_steps, 3))               # Storing target velocity
# check and create directory
if not os.path.exists(saved_dir):
    os.makedirs(saved_dir)

# Camera settings
width = 1920  # Increased width for higher resolution
height = 1080  # Increased height for higher resolution
pic_array = np.zeros((num_steps, height, width, 3), dtype=np.uint8)
view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=gripper_pose[0], distance=.4, yaw=0, pitch=-100, roll=0, upAxisIndex=2)
projMatrix = p.computeProjectionMatrixFOV(fov=60, aspect=float(width) / height, nearVal=0.1, farVal=10)
projMatrix = p.computeProjectionMatrixFOV(
    fov=60, 
    aspect=float(width) / height, 
    nearVal=0.05,  # Reduced near plane for better depth precision
    farVal=5        # Increased far plane for a broader view
)


# Physics engine parameters
p.setPhysicsEngineParameter(
    numSolverIterations=1000,          # Increased for more accurate simulations
    contactERP=0.8,                    # Error reduction parameter
    globalCFM=0.01,                    # Constraint force mixing
    enableConeFriction=1,              # Enables cone friction for better contact modeling
    contactSlop=0.0001,                # Minimal allowed penetration
    maxNumCmdPer1ms=1000,              # Maximum number of commands per millisecond
    contactBreakingThreshold=0.001,    # Threshold for breaking contacts
    enableFileCaching=1,               # Enables caching for performance
    restitutionVelocityThreshold=0.01  # Threshold for restitution
)

# Simulation loop
for i in range(num_steps):
    # p.applyExternalForce(target, -1, [0.15, 0.15, 0], [0, 0, 0], p.LINK_FRAME)
    p.stepSimulation()

    # Capture camera image
    # _, _, rgb, _, _ = p.getCameraImage(width, height, viewMatrix=view_matrix, projectionMatrix=projMatrix)
    _, _, rgbImg2, _, _ = p.getCameraImage(width, height, 
                                               viewMatrix = view_matrix, 
                                               projectionMatrix = projMatrix, 
                                               renderer=p.ER_BULLET_HARDWARE_OPENGL,  # Use hardware OpenGL for rendering
                                               lightDirection=[1, 0, 2],  # Light coming from an angle
                                               lightColor=[1.0, 1.0, 1.0],  # White sunlight
                                               lightDistance=1000,  # Large value for sunlight-like directional light
                                               lightAmbientCoeff=0.3,  # Reduced ambient light for sharper lighting
                                               lightDiffuseCoeff=0.7,  # Reduced diffuse lighting for sharper shadows
                                               lightSpecularCoeff=0.8,  # Increased specular for sharper highlights
                                               )
    rgbImg2 = rgbImg2[:, :, :3]
    # rgbImg2 = cv2.cvtColor(rgbImg2, cv2.COLOR_RGBA2BGR)
    # rgbImg2 = cv2.GaussianBlur(rgbImg2, (5, 5), 0)

    # Alternatively, use more advanced techniques like bilateral filtering
    # rgbImg2 = cv2.bilateralFilter(rgbImg2, d=9, sigmaColor=75, sigmaSpace=75)
    # pic_array[i] = np.reshape(rgbImg2, (height, width, 3))
    pic_array[i] = rgbImg2

    # Close fingers
    for finger in fingers:
        finger.close()
    # print(f"finger 3 sensor: {fingers[2].ReadSensor()}")

    # Process finger contacts and record data
    for finger_idx, finger in enumerate(fingers):
        num_contacts, total_force, lat_force = finger.fingertip_contact()
        lat_contact_data[i, finger_idx] = lat_force
        total_contact_data[i, finger_idx] = total_force

    target_vel_data[i] = p.getBaseVelocity(target)[0]

# Create and save the video
make_video(pic_array, f"{saved_dir}/simple_close.mp4")

# Create and save plots for each finger's lateral contact
for finger_idx in range(num_fingers):
    create_and_save_plot(lat_contact_data[:, finger_idx, :], 
                         f"Lateral Contact Force Finger {finger_idx+1}", 
                         "Time", 
                         "Lateral Force", 
                         f"{saved_dir}/lateral_contact_finger_{finger_idx+1}.png", 
                         legend_labels=['X', 'Y', 'Z'])

# create_and_save_plot(target_vel_data, "Target Velocity", "Time", "Velocity", f"{saved_dir}/target_velocity_plot.png", legend_labels=['X', 'Y', 'Z'])
# create_and_save_plot(total_contact_data, "Total Contact Force", "Time", "Force", f"{saved_dir}/total_contact_plot.png", legend_labels=['Finger 1', 'Finger 2', 'Finger 3'])