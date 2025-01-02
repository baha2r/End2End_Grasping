import os, inspect
import math
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
import pybullet as p
import pickle
import trimesh
import robotiq
import pybullet_data
from scipy.spatial.transform import Rotation
from scipy.spatial import distance
from stable_baselines3 import SAC
from statetest import GripperFingerStateMachine as GSM
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, currentdir)

largeValObservation = 10000

RENDER_HEIGHT = 720
RENDER_WIDTH = 960

class graspGymEnv(gym.Env):
    """
    This class describes the environment for the Robotiq grasp.
    """

    def __init__(self,
                 urdf_root=pybullet_data.getDataPath(),
                 action_repeat = 1,
                 renders = False,
                 records = False,
                 max_episode_steps = 1500,
                 store_data = False,
                 data_path = "data.pkl"):
        """
        Initialize the environment.
        """
        self._timeStep = 1. / 240.
        self._urdf_root = urdf_root
        self._robotiqRoot = "urdf/"
        self._action_repeat = action_repeat
        self.positioning_observation = []
        self._observation = []
        self._achieved_goal = []
        self._desired_goal = []
        self._stepcounter = 0
        self._grasp_stepcounter = 0
        self._renders = renders
        self._records = records
        self._max_steps = max_episode_steps
        self.terminated = False
        self._cam_dist = 0.4
        self._cam_yaw = 180
        self._cam_pitch = -40
        self._reach = 0
        self._grasp = 0
        self._keypoints = 100
        self.distance_threshold = 0.04
        self._accumulated_contact_force = 0
        self._cam1_images = []
        self._cam2_images = []
        self._cam3_images = []
        self.data_path = data_path
        self.store_data = store_data

        # connect to PyBullet
        if self._renders:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        # set a random seed
        seed = np.random.randint(0, 10000)
        self.reset(seed=seed)

        # Define observation space
        observation_high = np.array([largeValObservation] * len(self.getObservation()), dtype=np.float32)
        self.observation_space = spaces.Box(-observation_high, observation_high, dtype=np.float32)

        # Define action space
        # self.action_space = spaces.MultiDiscrete([25, 25, 25])
        self.action_space = spaces.Box(low=np.array([0, 0, 0]), high=np.array([10, 10, 100]), dtype=np.float32)

        self.viewer = None

    def reset(self, seed=None):
        """
        Reset the environment to its initial state and return the initial observation.
        """
        print("Resetting environment")
        super().reset(seed=seed)
        self.success_counter = 0
        self.grasp_success_counter = 0
        self._stepcounter = 0
        self._reach = 0
        self._grasp = 0
        self._action = np.array([0, 0, 0])
        self.terminated = False
        self._cam1_images = []
        self._cam2_images = []
        self._cam3_images = []
        p.resetSimulation()
        # Physics engine parameters
        # contactERP_ = np.random.uniform(0.9, 0.99)
        # globalCFM_ = np.random.uniform(0.001, 0.01)
        # contactSlop_ = np.random.uniform(0.0001, 0.001)
        p.setPhysicsEngineParameter(
            numSolverIterations=1000,
            contactERP=0.9,
            globalCFM=0.01,
            enableConeFriction=1,
            contactSlop=0.0001,
            maxNumCmdPer1ms=1000,
            contactBreakingThreshold=0.0001,
            enableFileCaching=1,
            restitutionVelocityThreshold=0.01
        )

        # p.setTimeStep(self._timeStep)
        p.loadURDF(os.path.join(self._urdf_root, "plane.urdf"), [0, 0, 0])

        self._robotiq = robotiq.robotiq()

        # Generate random values
        randx, randy, randz, randf1, randf2, randf3 = np.random.uniform(-1, 1, 6)

        targetpos = [0.0 + 0.50 * randx, 0.8 + 0.2 * randy, 1.0 + 0.40 * randz]
        targetorn = p.getQuaternionFromEuler([0, 0, 0])

        self.blockUid = p.loadURDF(
            "urdf/block.urdf", 
            basePosition=targetpos, 
            baseOrientation=targetorn, 
            # useMaximalCoordinates=True,
            # flags=p.URDF_INITIALIZE_SAT_FEATURES,
            # useFaceContact = 1,
        )
        lateral_friction = np.random.uniform(0.5, 0.9)
        spinning_friction = np.random.uniform(0.005, 0.01)
        rolling_friction = np.random.uniform(0.01, 0.02)
        lateral_friction = 0.9
        spinning_friction = 0.01
        rolling_friction = 0.02

        self.targetmass = np.random.uniform(30, 50)
        # self.targetmass = 50
        p.changeDynamics(self.blockUid, -1,
                 mass=self.targetmass, # adjust mass
                 lateralFriction=lateral_friction, # adjust lateral friction
                 spinningFriction=spinning_friction, # adjust spinning friction
                 rollingFriction=rolling_friction, # adjust rolling friction               
                #  contactStiffness=1000, contactDamping=10
                ) # adjust contact stiffness and damping
        
        extforce = np.array([randf1, randf2, randf3]) * (30 * self.targetmass)
        p.applyExternalForce(self.blockUid, -1 , extforce , [0,0,0] , p.LINK_FRAME)
        
        p.setGravity(0, 0, 0)
        p.stepSimulation()
        
        # Define the column names
        columns = ['stepcounter', 'grasp_stepcounter', 'position_action', 'orientation_action', 'gripper_position',
                    'gripper_orientation', 'gripper_linear_velocity', 'gripper_angular_velocity', 'block_position',
                    'block_orientation', 'block_linear_velocity', 'block_angular_velocity', 'closest_points',
                    'positioning_reward', 'grasp_reward', 'action_fingers_closing_speed', 'action_fingers_closing_force',
                    'action_fingers_grasping_force', 'joints1_angles', 'joints1_velocity', 'joints1_appliedJointMotorTorque',
                    'min_ftip_distance', 'ftipContactPoints', 'ftipNormalForce', 'accumulated_Normal_Force',
                    'ftip_lateral_friction_X', 'ftip_lateral_friction_Y', 'ftip_lateral_friction_Z',
                    'finger1_angle', 'finger2_angle', 'finger3_angle', 
                    'is_reach', 'is_grasp', 'grasp_success',
                    ]

        # Initialize a list to store data dictionaries
        self.df = []
        
        model_file = "models/Positioning_Agent/best_model.zip"
        self.model = self._load_model(model_file)
    
        self.fingers = [GSM(self._robotiq.robotiq_uid, self.blockUid, i) for i in range(1, 4)]
        
        while self._reach != 1:
            self.positioning_observation = self.getPositioningObservation()
            self.base_commands, _ = self.model.predict(self.positioning_observation, deterministic=True)
            self._robotiq.apply_positioning_action(self.base_commands)
            p.stepSimulation()
            self._stepcounter += 1
            if self.store_data:
                self.update_data()
            self._is_success()
            if self._reach == 0 and self._stepcounter>1000:
                print("Failed to reach the object")
                self.reset()
            if self._records:
                self.render()
        
        info = {"is_success": 0}
        
        return self.getObservation(), info
    
    def reset_positioning(self):
        """
        Reset the positioning env
        """        

        self.positioning_observation = self.getPositioningObservation()

        return self.positioning_observation
    
    def grasp_success(self):
        """
        Check if the current state is successful. 
        Success is defined as having np.all(ftipContactPoints > 0) for more than 100 consecutive steps.
        """
        if np.all(self._contactinfo()[3] > 0):
            self.grasp_success_counter += 1
        else:
            self.grasp_success_counter = 0

        # return 1 if the gripper is in contact with the object for more than 100 steps and also object velocity is less than 0.001
        return np.linalg.norm(p.getBaseVelocity(self.blockUid)[0]) < 0.001

    def _is_success(self):
        """
        Check if the current state is successful. 
        Success is defined as having a reward greater than 2 for more than 100 consecutive steps.
        """
        if self.positioning_reward() > 2:
            self.success_counter += 1
        else:
            self.success_counter = 0

        if np.float32(self.success_counter > 300):
            self._reach = 1
            
    def getPositioningObservation(self):
        """
        Return an extended observation that includes information about the block in the gripper.
        The extended observation includes relative position, velocity, and contact information.
        """
        self.positioning_observation = self._robotiq.get_observation()

        # Fetch base position and orientation of gripper and block
        gripperPos, gripperOrn = p.getBasePositionAndOrientation(self._robotiq.robotiq_uid)
        griplinvel, gripangvel = p.getBaseVelocity(self._robotiq.robotiq_uid)
        blockPos, blockOri = p.getBasePositionAndOrientation(self.blockUid)
        blocklinVel, blockangVel = p.getBaseVelocity(self.blockUid)

        # Convert block and gripper orientation from Quaternion to Euler for ease of manipulation
        blockEul = p.getEulerFromQuaternion(blockOri)
        gripEul = p.getEulerFromQuaternion(gripperOrn)

        # Define block pose and append to observation
        blockPose = np.array([
            *blockPos, 
            *blockEul
        ], dtype=np.float32)
        self.positioning_observation = np.append(self.positioning_observation, blockPose)

        # Define relative pose and append to observation
        relPose = np.array([
            *(np.subtract(blockPos, gripperPos)), 
            *(np.subtract(blockEul, gripEul))
        ], dtype=np.float32)
        self.positioning_observation = np.append(self.positioning_observation, relPose)

        # Define block velocity and append to observation
        blockVel = np.array([
            *blocklinVel, 
            *blockangVel
        ], dtype=np.float32)
        self.positioning_observation = np.append(self.positioning_observation, blockVel)

        # Define relative velocity and append to observation
        relVel = np.array([
            *(np.subtract(blocklinVel, griplinvel)), 
            *(np.subtract(blockangVel, gripangvel))
        ], dtype=np.float32)
        self.positioning_observation = np.append(self.positioning_observation, relVel)

        # Add minimum distance between the robot and the block to observation
        closestpoints = p.getClosestPoints(self._robotiq.robotiq_uid, self.blockUid, 100, -1, -1)
        minpos = np.subtract(closestpoints[0][5], closestpoints[0][6])
        self.positioning_observation = np.append(self.positioning_observation, minpos)

        # Add contact information to observation
        totalforce = self._contactinfo()[5]
        # self.positioning_observation = np.append(self.positioning_observation, totalforce)

        return self.positioning_observation
    
    def getObservation(self):
        """
        Return the observation for the current state of the environment.
        """
        self._observation = self._robotiq.finger_observation()

        # get the minimum distance between the robotiq fingertips and the block
        min_ftip_distance = [p.getClosestPoints(self.blockUid, self._robotiq.robotiq_uid, 10, -1, self._robotiq.third_joint_idx[i])[0][8] for i in range(3)]
        self._observation = np.append(self._observation, min_ftip_distance)

        _contactinfo = self._contactinfo()
        # get the number of contact points for each fingertip
        ftipContactPoints = _contactinfo[4]
        self._observation = np.append(self._observation, ftipContactPoints)

        ftipNormalForce = _contactinfo[0]
        self._observation = np.append(self._observation, ftipNormalForce)

        accumulated_Normal_Force = self._accumulated_contact_force
        self._observation = np.append(self._observation, accumulated_Normal_Force)
        
        ftip_lateral_friction_X = _contactinfo[1][0]
        self._observation = np.append(self._observation, ftip_lateral_friction_X)
        
        ftip_lateral_friction_Y = _contactinfo[1][1]
        self._observation = np.append(self._observation, ftip_lateral_friction_Y)
        
        ftip_lateral_friction_Z = _contactinfo[1][2]
        self._observation = np.append(self._observation, ftip_lateral_friction_Z)

        self._observation = np.array(self._observation, dtype=np.float32)

        # print(len(self._observation))

        return self._observation
    
    def step(self, action):
        """
        Apply an action to the robot, simulate the physics for the defined action repeat,
        and return the current state, reward, termination, and success state of the environment.
        """
        self._action = action
        if self._renders:
            time.sleep(self._timeStep)

        if self._grasp == 1:
            self.base_commands = self.base_commands - self.base_commands * .01
            # print("Grasping the object")

        for _ in range(self._action_repeat):
            self._robotiq.apply_positioning_action(self.base_commands)
            self.close_fingers(self.fingers, self._action)
            p.stepSimulation()
            self._termination()
            if self.terminated:
        # if self._stepcounter == self._max_steps:
                serialized_data = pickle.dumps(self.df)
                with open(self.data_path, "wb") as f:
                    f.write(serialized_data)
            if self.store_data:
                self.update_data()
            if self._records:
                self.render()
            self._is_success()
            self._stepcounter += 1
            self._grasp_stepcounter += 1

        self._termination()
        if self.terminated:
        # if self._stepcounter == self._max_steps:
                serialized_data = pickle.dumps(self.df)
                with open(self.data_path, "wb") as f:
                    f.write(serialized_data)
        
        reward = self.grasp_reward()
        infos = {"is_success": self.grasp_success()}
        truncated = self._stepcounter > self._max_steps
        return self.getObservation(), reward, self.terminated, truncated, infos

    def close_fingers(self, fingers, action):
        """
        Close the fingers.
        """
        # action[1] *= 10
        # action[2] *= 10
        for finger in fingers:
            finger.close(close_vel=action[0], close_force=action[1], grasp_force=action[2])

    def render(self, mode="rgb_array", close=False):
        """
        Render the environment to an image array.

        Arguments:
        - mode: The mode to render with. The default is "rgb_array".
        - close: If True, closes the rendering window. The default is False.

        Returns: 
        - A RGB array of the rendered environment if mode is "rgb_array", otherwise an empty array.
        """
        # If mode is not "rgb_array", return an empty array
        if mode != "rgb_array":
            return np.array([])

        # Get the position and orientation of the base and target
        grip_pos, _ = p.getBasePositionAndOrientation(self._robotiq.robotiq_uid)
        target_pos, _ = p.getBasePositionAndOrientation(self.blockUid)

        # Calculate the camera position based on the base position
        camera_pos = grip_pos + np.array([0, -0.05, 0.2])

        # Define the size of the rendered image
        width, height = RENDER_WIDTH, RENDER_HEIGHT

        # # Calculate the view and projection matrices for the camera
        view_matrix_1 = p.computeViewMatrix(cameraEyePosition=camera_pos, cameraTargetPosition=target_pos, cameraUpVector=[0, 1, 0])
        view_matrix_2 = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=grip_pos, distance=0.9, yaw=90, pitch=-20, roll=0, upAxisIndex=2)
        view_matrix_3 = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=target_pos, distance=0.9, yaw=180, pitch=-20, roll=0, upAxisIndex=2)
        proj_matrix = p.computeProjectionMatrixFOV(fov=60, aspect=float(width) / height, nearVal=0.1, farVal=10.0)

        # First camera
        # _, _, rgbImg1, _, _ = p.getCameraImage(width, height, viewMatrix = view_matrix_1, projectionMatrix = proj_matrix)
        # rgbImg1 = rgbImg1[:, :, :3]
        # self._cam1_images.append(rgbImg1)

        # Second camera
        _, _, rgbImg2, _, _ = p.getCameraImage(width, height, 
                                               viewMatrix = view_matrix_2, 
                                               projectionMatrix = proj_matrix, 
                                               renderer=p.ER_BULLET_HARDWARE_OPENGL,
                                               lightDirection=[5, -10, 0],  # Light coming from an angle
                                               lightColor=[1.0, 1.0, 1.0],  # White sunlight
                                               lightDistance=10000,  # Large value for sunlight-like directional light
                                               lightAmbientCoeff=0.3,  # Reduced ambient light for sharper lighting
                                               lightDiffuseCoeff=0.7,  # Reduced diffuse lighting for sharper shadows
                                               lightSpecularCoeff=0.8,  # Increased specular for sharper highlights
                                               )        
        rgbImg2 = rgbImg2[:, :, :3]
        self._cam2_images.append(rgbImg2)
                
        # # third camera
        # _, _, rgbImg3, _, _ = p.getCameraImage(width, height, viewMatrix = view_matrix_3, projectionMatrix = proj_matrix)
        # rgbImg3 = rgbImg3[:, :, :3]
        # self._cam3_images.append(rgbImg3)
        

        return np.array([])

    def _termination(self):
        """
        Check whether the environment should be terminated.

        Returns:
        - True if the environment has reached its maximum number of steps.
        - False otherwise.
        """
        if self._stepcounter > self._max_steps:
            self.terminated = True

    def ReadSensor(self, finger):
        # Transitions Event = [C1 C2 C3 L1 L2 L3 L4],
        # Ci = 1 indicates that link i is blocked by the object
        # Li = 1 indicates that link i has reached the limit
        # L4 = 1 indicates that link 3 has reached the lower limit
        # finger = [1, 2, 3]
        Limits = [67, 90, 43, -55]
        finger -= 1
        tol = 1
            
        C1 = p.getContactPoints(self.blockUid, self._robotiq.robotiq_uid, -1, self._robotiq.first_joint_idx[finger])
        C2 = p.getContactPoints(self.blockUid, self._robotiq.robotiq_uid, -1, self._robotiq.second_joint_idx[finger])
        C3 = p.getContactPoints(self.blockUid, self._robotiq.robotiq_uid, -1, self._robotiq.third_joint_idx[finger])

        theta1 = p.getJointState(self._robotiq.robotiq_uid, self._robotiq.first_joint_idx[finger])[0]
        theta2 = p.getJointState(self._robotiq.robotiq_uid, self._robotiq.second_joint_idx[finger])[0]
        theta3 = p.getJointState(self._robotiq.robotiq_uid, self._robotiq.third_joint_idx[finger])[0]

        L1 = int(np.abs(theta1 - Limits[0]) <= tol)
        L2 = int(np.abs(theta2 - Limits[1]) <= tol)

        L3 = int(Limits[2] - tol <= theta3 <= Limits[2] + tol)
        L4 = int(Limits[3] - tol <= theta3 <= Limits[3] + tol)

        return [C1, C2, C3, L1, L2, L3, L4]

    def _contactinfo(self):
        """
        Compute various contact forces between the block and the robotiq and 
        the number of contact points for each fingertip.
        
        Returns:
        - ftipNormalForce: total normal force at the fingertips
        - ftipLateralFriction1: total lateral friction in one direction at the fingertips
        - ftipLateralFriction2: total lateral friction in the other direction at the fingertips
        - ftipContactPoints: array containing the number of contact points for each fingertip
        - totalNormalForce: total normal force between block and robotiq
        - totalLateralFriction1: total lateral friction in one direction between block and robotiq
        - totalLateralFriction2: total lateral friction in the other direction between block and robotiq
        """
        totalNormalForce = 0

        # Get contact points between block and robotiq
        self.contactpoints = p.getContactPoints(self.blockUid, self._robotiq.robotiq_uid)        

        # get number of contact points
        number_of_contact_points = len(self.contactpoints)
        totalLateralFrictionForce = [0, 0, 0]
        
        if self.contactpoints:
            for i , c in enumerate(self.contactpoints):
                # calculate the total normal force
                totalNormalForce += c[9]
                # calculate the total lateral friction force
                totalLateralFrictionForce[0] += c[11][0] * c[10] + c[13][0] * c[12]
                totalLateralFrictionForce[1] += c[11][1] * c[10] + c[13][1] * c[12]
                totalLateralFrictionForce[2] += c[11][2] * c[10] + c[13][2] * c[12]

        # Get contact points between block and each fingertip of robotiq
        ftip1 = p.getContactPoints(self.blockUid, self._robotiq.robotiq_uid, -1, self._robotiq.third_joint_idx[0])
        ftip2 = p.getContactPoints(self.blockUid, self._robotiq.robotiq_uid, -1, self._robotiq.third_joint_idx[1])
        ftip3 = p.getContactPoints(self.blockUid, self._robotiq.robotiq_uid, -1, self._robotiq.third_joint_idx[2])

        # Initialize arrays to store forces
        ftipNormalForce = [0, 0, 0]
        ftipLateralFrictionForce = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

        # Calculate the average forces for each fingertip
        for i, ftip in enumerate([ftip1, ftip2, ftip3]):
            if ftip:  # Only proceed if there are contact points
                n_contacts = len(ftip)  # Number of contact points
                for c in ftip:
                    ftipNormalForce[i] += c[9]
                    ftipLateralFrictionForce[i][0] += c[11][0] * c[10] + c[13][0] * c[12]
                    ftipLateralFrictionForce[i][1] += c[11][1] * c[10] + c[13][1] * c[12]
                    ftipLateralFrictionForce[i][2] += c[11][2] * c[10] + c[13][2] * c[12]
                # Average the forces by the number of contacts
                ftipNormalForce[i] /= n_contacts
                ftipLateralFrictionForce[i] = [f / n_contacts for f in ftipLateralFrictionForce[i]]

        # Count the number of contact points for each fingertip
        ftipContactPoints = np.array([len(ftip1), len(ftip2), len(ftip3)])

        # create an indicator when all fingertips are in contact
        if np.all(ftipContactPoints > 0):
            self._grasp = 1
        
        self._accumulated_contact_force += totalNormalForce

        return ftipNormalForce, ftipLateralFrictionForce, totalLateralFrictionForce, number_of_contact_points, ftipContactPoints, totalNormalForce

    def positioning_reward(self):
        """
        Compute the reward for the current state of the environment.
        
        Reward is based on distance, orientation, force applied, speed of movement, and whether fingertips are in contact with the object.

        Returns:
        - reward: The calculated reward for the current state.
        """

        # Get position and orientation for block and gripper
        blockPos, blockOrn = p.getBasePositionAndOrientation(self.blockUid)
        blockOrnEuler = p.getEulerFromQuaternion(blockOrn)

        gripperPos , gripperOrn = p.getBasePositionAndOrientation(self._robotiq.robotiq_uid)
        gripOrnEuler = p.getEulerFromQuaternion(gripperOrn)

        # Get velocities for block and gripper
        blocklinvel, blockangvel = p.getBaseVelocity(self.blockUid)
        griplinvel, gripangvel = p.getBaseVelocity(self._robotiq.robotiq_uid)

        # Convert velocities to magnitude (speed)
        blocklinvel = np.linalg.norm(blocklinvel)
        blockangvel = np.linalg.norm(blockangvel)
        griplinvel = np.linalg.norm(griplinvel)
        gripangvel = np.linalg.norm(gripangvel)

        # Get closest distance between block and gripper, and subtract distance threshold
        closestPoints = np.absolute(p.getClosestPoints(self.blockUid, self._robotiq.robotiq_uid, 100, -1, -1)[0][8] - self.distance_threshold)

        # Compute orientation related quantities
        r = Rotation.from_quat(gripperOrn)
        normalvec = np.matmul(r.as_matrix(), np.array([0,1,0]))
        diffvector = np.subtract(np.array(blockPos), np.array(gripperPos))
        dotvec = np.dot(diffvector/np.linalg.norm(diffvector), normalvec/np.linalg.norm(normalvec))

        # Compute contact information
        ftipNormalForce, ftipLateralFrictionForce, totalLateralFrictionForce, number_of_contact_points, ftipContactPoints, totalNormalForce = self._contactinfo()

        r_top = self._r_topology()
        r_top = 1 if r_top > 0 else 0

        contactpenalize = -1 if totalNormalForce > 0 else 0

        # Compute rewards for different aspects
        distanceReward = 1 - math.tanh(closestPoints)
        oriReward = 1 - math.tanh(distance.euclidean(np.array(blockOrnEuler),np.array(gripOrnEuler)))
        normalForceReward = 1 - math.tanh(totalNormalForce)
        gripangvelReward = 1 - math.tanh(gripangvel)
        # fingerActionReward = 1 - math.tanh(abs(self._action[-1]))

        # positionActionReward = np.linalg.norm(self._action[0:3]) / np.sqrt(3)
        # orientationActionReward = np.linalg.norm(self._action[3:6]) / np.sqrt(3)

        # Combine rewards to get final reward
        reward = distanceReward + oriReward + r_top + contactpenalize 

        return reward

    def grasp_reward(self):
        """
        Compute the reward for the current state of the environment.
        
        Reward is based on distance, orientation, force applied, speed of movement, and whether fingertips are in contact with the object.

        Returns:
        - reward: The calculated reward for the current state.
        """
        if self._reach:

            # Compute contact information
            ftipNormalForce, ftipLateralFrictionForce, totalLateralFrictionForce, number_of_contact_points, ftipContactPoints, totalNormalForce = self._contactinfo()

            # r_top = self._r_topology()
            # r_top = 1 if r_top > 0 else 0
            ftip_dist_list = self._observation[9:12]            
            ftip_dist_list = [0 if ftip_dist_list[i] < 0 else ftip_dist_list[i] for i in range(3)]

            ftip_distance = np.linalg.norm(ftip_dist_list)
            distanceReward = 1 - math.tanh(ftip_distance*100)

            # contactpenalize = -1 if totalNormalForce > 0 else 0

            # Compute rewards for different aspects
            # distanceReward = 1 - math.tanh(closestPoints)
            # oriReward = 1 - math.tanh(distance.euclidean(np.array(blockOrnEuler),np.array(gripOrnEuler)))
            normalForceReward = 1 - math.tanh(np.linalg.norm(ftipNormalForce) / 10)
            # gripangvelReward = 1 - math.tanh(gripangvel)
            # fingerActionReward = 1 - math.tanh(abs(self._action[-1]))
            
            # positionActionReward = np.linalg.norm(self._action[0:3]) / np.sqrt(3)
            # orientationActionReward = np.linalg.norm(self._action[3:6]) / np.sqrt(3)
            
            # Combine rewards to get final reward
            target_v = np.linalg.norm(p.getBaseVelocity(self.blockUid)[0])
            targetReward = 1 - math.tanh(100*target_v)
            reward = targetReward + normalForceReward + distanceReward
            
            return reward
        else:
            return 0

    def _r_topology(self):
        """
        Calculates the ratio of points within a generated point cloud that lie inside the convex hull of the gripper.

        Returns:
        - n_count: The ratio of contained points over total keypoints.
        """

        # Small sphere properties
        red_point_dot_radius = 2
        red_point_dot_color = [1, 0, 0]  # Red color: [r, g, b]
        red_point_dot_opacity = 1.0  # Fully opaque

        # Compute axis-aligned bounding box (AABB) of the block
        aabb_min, aabb_max = p.getAABB(self.blockUid)

        # Generate a grid of points (5 points along each axis) within the AABB
        x = np.linspace(aabb_min[0], aabb_max[0], 5)
        y = np.linspace(aabb_min[1], aabb_max[1], 5)
        z = np.linspace(aabb_min[2], aabb_max[2], 5)
        
        # Store all generated points in a list
        points = []
        for i in x:
            for j in y:
                for k in z:
                    points.append([i, j, k])
        points = np.array(points)

        # Color for each point
        red_point_dot_color = np.array([red_point_dot_color] * len(points))
        
        # Add points to debug visualizer
        # p.addUserDebugPoints(pointPositions=points,
        #                     pointColorsRGB=red_point_dot_color,
        #                     pointSize=red_point_dot_radius)

        # Get the pose of each gripper link
        gripper_link_pose = [p.getLinkState(self._robotiq.robotiq_uid, i)[0] for i in range(self._robotiq.num_joints)]
        gripper_link_pose = np.array(gripper_link_pose)

        # Calculate the convex hull of the gripper link poses
        hull = trimesh.convex.convex_hull(gripper_link_pose)

        # Check which points are inside the hull
        n = hull.contains(points)

        # Calculate the ratio of points inside the hull
        n_count = np.count_nonzero(n) / self._keypoints

        return n_count

    def _load_model(self, file_path):
        """
        Load the model from the given file path.
        """
        return SAC.load(file_path)
    
    def contact_point_to_dict(self, contact_point):
        return {
            'bodyUniqueIdA': contact_point[1],
            'bodyUniqueIdB': contact_point[2],
            'linkIndexA': contact_point[3],
            'linkIndexB': contact_point[4],
            'positionOnAInWS': contact_point[5],
            'positionOnBInWS': contact_point[6],
            'contactNormalOnBInWS': contact_point[7],
            'contactDistance': contact_point[8],
            'normalForce': contact_point[9],
            'lateralFrictionForce1': contact_point[10],
            'lateralFrictionForce2': contact_point[11],
            'lateralFrictionDir1': contact_point[12],
            'lateralFrictionDir2': contact_point[13]
        }

    def update_data(self):
        
        """columns = ['stepcounter', 'grasp_stepcounter', 'position_action', 'orientation_action', 'gripper_position',
                    'gripper_orientation', 'gripper_linear_velocity', 'gripper_angular_velocity', 'block_position',
                    'block_orientation', 'block_linear_velocity', 'block_angular_velocity', 'closest_points',
                    'positioning_reward', 'grasp_reward', 'action_fingers_closing_speed', 'action_fingers_closing_force',
                    'action_fingers_grasping_force', 'joints1_angles', 'joints1_velocity', 'joints1_appliedJointMotorTorque',
                    'min_ftip_distance', 'ftipContactPoints', 'ftipNormalForce', 'accumulated_Normal_Force',
                    'ftip_lateral_friction_X', 'ftip_lateral_friction_Y', 'ftip_lateral_friction_Z',
                    'finger1_angle', 'finger2_angle', 'finger3_angle', 
                    'is_reach', 'is_grasp',
                    ]
        """
        self.positioning_observation = self.getPositioningObservation()
        self.grasp_observation = self.getObservation()
        # Ensure that numpy arrays are stored as object type within the DataFrame
        stepcounter = self._stepcounter if isinstance(self._stepcounter, np.ndarray) else np.array([self._stepcounter])
        grasp_stepcounter = self._grasp_stepcounter if isinstance(self._grasp_stepcounter, np.ndarray) else np.array([self._grasp_stepcounter])
        position_action = self.base_commands[0:3] if isinstance(self.base_commands[0:3], np.ndarray) else np.array([self.base_commands[0:3]])
        orientation_action = self.base_commands[3:6] if isinstance(self.base_commands[3:6], np.ndarray) else np.array([self.base_commands[3:6]])
        gripper_position = self.positioning_observation[0:3] if isinstance(self.positioning_observation[0:3], np.ndarray) else np.array([self.positioning_observation[0:3]])
        gripper_orientation = self.positioning_observation[3:6] if isinstance(self.positioning_observation[3:6], np.ndarray) else np.array([self.positioning_observation[3:6]])
        gripper_linear_velocity = self.positioning_observation[6:9] if isinstance(self.positioning_observation[6:9], np.ndarray) else np.array([self.positioning_observation[6:9]])
        gripper_angular_velocity = self.positioning_observation[9:12] if isinstance(self.positioning_observation[9:12], np.ndarray) else np.array([self.positioning_observation[9:12]])
        block_position = self.positioning_observation[12:15] if isinstance(self.positioning_observation[12:15], np.ndarray) else np.array([self.positioning_observation[12:15]])
        block_orientation = self.positioning_observation[15:18] if isinstance(self.positioning_observation[15:18], np.ndarray) else np.array([self.positioning_observation[15:18]])
        block_linear_velocity = self.positioning_observation[24:27] if isinstance(self.positioning_observation[24:27], np.ndarray) else np.array([self.positioning_observation[24:27]])
        block_angular_velocity = self.positioning_observation[27:30] if isinstance(self.positioning_observation[27:30], np.ndarray) else np.array([self.positioning_observation[27:30]])
        closest_points = self.positioning_observation[36:39] if isinstance(self.positioning_observation[36:39], np.ndarray) else np.array([self.positioning_observation[36:39]])
        positioning_reward = self.positioning_reward() if isinstance(self.positioning_reward(), np.ndarray) else np.array([self.positioning_reward()])
        grasp_reward = self.grasp_reward() if isinstance(self.grasp_reward(), np.ndarray) else np.array([self.grasp_reward()])
        action_fingers_closing_speed = self._action[0] if isinstance(self._action[0], np.ndarray) else np.array([self._action[0]])
        action_fingers_closing_force = self._action[1] if isinstance(self._action[1], np.ndarray) else np.array([self._action[1]])
        action_fingers_grasping_force = self._action[2] if isinstance(self._action[2], np.ndarray) else np.array([self._action[2]])
        joints1_angles = self.grasp_observation[0:3] if isinstance(self.grasp_observation[0:3], np.ndarray) else np.array([self.grasp_observation[0:3]])
        joints1_velocity = self.grasp_observation[3:6] if isinstance(self.grasp_observation[3:6], np.ndarray) else np.array([self.grasp_observation[3:6]])
        joints1_appliedJointMotorTorque = self.grasp_observation[6:9] if isinstance(self.grasp_observation[6:9], np.ndarray) else np.array([self.grasp_observation[6:9]])
        min_ftip_distance = self.grasp_observation[9:12] if isinstance(self.grasp_observation[9:12], np.ndarray) else np.array([self.grasp_observation[9:12]])
        ftipContactPoints = self.grasp_observation[12:15] if isinstance(self.grasp_observation[12:15], np.ndarray) else np.array([self.grasp_observation[12:15]])
        ftipNormalForce = self.grasp_observation[15:18] if isinstance(self.grasp_observation[15:18], np.ndarray) else np.array([self.grasp_observation[15:18]])
        accumulated_Normal_Force = self.grasp_observation[18] if isinstance(self.grasp_observation[18], np.ndarray) else np.array([self.grasp_observation[18]])
        ftip_lateral_friction_X = self.grasp_observation[19:22] if isinstance(self.grasp_observation[19:22], np.ndarray) else np.array([self.grasp_observation[19:22]])
        ftip_lateral_friction_Y = self.grasp_observation[22:25] if isinstance(self.grasp_observation[22:25], np.ndarray) else np.array([self.grasp_observation[22:25]])
        ftip_lateral_friction_Z = self.grasp_observation[25:28] if isinstance(self.grasp_observation[25:28], np.ndarray) else np.array([self.grasp_observation[25:28]])                                                                                          
        joint_indices = [1, 2, 3, 5, 6, 7, 9, 10, 11]
        joint_states = p.getJointStates(self._robotiq.robotiq_uid, joint_indices)
        finger1_angle = np.array([np.float64(joint_states[0][0]), np.float64(joint_states[1][0]), np.float64(joint_states[2][0])])
        finger2_angle = np.array([np.float64(joint_states[3][0]), np.float64(joint_states[4][0]), np.float64(joint_states[5][0])])
        finger3_angle = np.array([np.float64(joint_states[6][0]), np.float64(joint_states[7][0]), np.float64(joint_states[8][0])])
        is_reach = self._reach if isinstance(self._reach, np.ndarray) else np.array([self._reach])
        is_grasp = self._grasp if isinstance(self._grasp, np.ndarray) else np.array([self._grasp])
        grasp_success = self.grasp_success() if isinstance(self.grasp_success(), np.ndarray) else np.array([self.grasp_success()])
        contacts_list = [self.contact_point_to_dict(contact) for contact in self.contactpoints if contact[9] > 0]

        # Create a new DataFrame for the current timestep
        new_data = {
            'stepcounter': stepcounter,
            'grasp_stepcounter': grasp_stepcounter,
            'position_action': position_action,
            'orientation_action': orientation_action,
            'gripper_position': gripper_position,
            'gripper_orientation': gripper_orientation,
            'gripper_linear_velocity': gripper_linear_velocity,
            'gripper_angular_velocity': gripper_angular_velocity,
            'block_position': block_position,
            'block_orientation': block_orientation,
            'block_linear_velocity': block_linear_velocity,
            'block_angular_velocity': block_angular_velocity,
            'closest_points': closest_points,
            'positioning_reward': positioning_reward,
            'grasp_reward': grasp_reward,
            'action_fingers_closing_speed': action_fingers_closing_speed,
            'action_fingers_closing_force': action_fingers_closing_force,
            'action_fingers_grasping_force': action_fingers_grasping_force,
            'joints1_angles': joints1_angles,
            'joints1_velocity': joints1_velocity,
            'joints1_appliedJointMotorTorque': joints1_appliedJointMotorTorque,
            'min_ftip_distance': min_ftip_distance,
            'ftipContactPoints': ftipContactPoints,
            'ftipNormalForce': ftipNormalForce,
            'accumulated_Normal_Force': accumulated_Normal_Force,
            'ftip_lateral_friction_X': ftip_lateral_friction_X,
            'ftip_lateral_friction_Y': ftip_lateral_friction_Y,
            'ftip_lateral_friction_Z': ftip_lateral_friction_Z,
            'finger1_angle': finger1_angle,
            'finger2_angle': finger2_angle,
            'finger3_angle': finger3_angle,
            'is_reach': is_reach,
            'is_grasp': is_grasp,
            'grasp_success': grasp_success,
            'contact': contacts_list,
        }
        # use pickle to add the new data to the existing pickle variable
        self.df.append(new_data)

    def close(self):
        """
        Disconnects the physics client.
        """
        p.disconnect()
