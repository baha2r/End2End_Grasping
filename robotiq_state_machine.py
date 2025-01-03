from statemachine import State, StateMachine
import pybullet as p
import numpy as np

class GripperFingerStateMachine(StateMachine):
    """
    State machine for a single finger of a robotic gripper.
    """
    # states
    J1_Closing_J3_Opening = State('J1_Closing_J3_Opening', initial=True)
    J2_Closing = State('J2_Closing')
    J3_Closing = State('J3_Closing')
    FingerStopped = State('FingerStopped')#, final=True)

    # Transitions
    start_j2_closing = J1_Closing_J3_Opening.to(J2_Closing)
    start_j3_closing_from_j1 = J1_Closing_J3_Opening.to(J3_Closing)
    start_j3_closing_from_j2 = J2_Closing.to(J3_Closing)
    stop_finger_from_j1 = J1_Closing_J3_Opening.to(FingerStopped)
    stop_finger_from_j2 = J2_Closing.to(FingerStopped)
    stop_finger_from_j3 = J3_Closing.to(FingerStopped)
    resume_j1_closing_j3_opening = FingerStopped.to(J1_Closing_J3_Opening)
    resume_j3_closing_from_j2 = FingerStopped.to(J3_Closing)


    def __init__(self, gripper, target, finger):
        """
        Initialize the state machine with the robot ID and sensor data.
        """
        self.gripper = gripper
        self.target = target
        self.finger = finger - 1
        self.first_joint_idx = [1, 5, 9]
        self.second_joint_idx = [2, 6, 10]
        self.third_joint_idx = [3, 7, 11]
        self.sensors = self.ReadSensor()  # Assuming sensors is a list or dict representing [C1, C2, C3, L1, L2, L3, L4]
        self.close_vel = 1
        self.close_force = 1
        self.grasp_force = 5
        # if self.finger == 2:
        #     self.close_force *= 2
        #     self.grasp_force *= 2
        super().__init__()

    def fingertip_contact(self):
        
        C3 = p.getContactPoints(self.target, self.gripper, -1, self.second_joint_idx[self.finger])
        number_of_contact_points = len(C3)
        totalNormalForce = 0
        totalLateralFrictionForce = [0, 0, 0]
        
        for c in C3:
            totalNormalForce += c[9]
            totalLateralFrictionForce[0] += c[11][0] * c[10] + c[13][0] * c[12]
            totalLateralFrictionForce[1] += c[11][1] * c[10] + c[13][1] * c[12]
            totalLateralFrictionForce[2] += c[11][2] * c[10] + c[13][2] * c[12]
        totalLateralFrictionForce = np.array(totalLateralFrictionForce)
        
        return number_of_contact_points, totalNormalForce, totalLateralFrictionForce
    
    def ReadSensor(self):
        # Transitions Event = [C1 C2 C3 L1 L2 L3 L4],
        # Ci = 1 indicates that link i is blocked by the object
        # Li = 1 indicates that link i has reached the limit
        # L4 = 1 indicates that link 3 has reached the lower limit
        # finger = [1, 2, 3]
        Limits = [67, 90, 43, -55]
        # change Limits to radians
        for i in range(len(Limits)):
                Limits[i] = Limits[i] * np.pi / 180
        tol = 0.01
            
        C1 = p.getContactPoints(self.target, self.gripper, -1, self.first_joint_idx[self.finger])
        C2 = p.getContactPoints(self.target, self.gripper, -1, self.second_joint_idx[self.finger])
        C3 = p.getContactPoints(self.target, self.gripper, -1, self.third_joint_idx[self.finger])

        C1 = int(len(C1) > 0)
        C2 = int(len(C2) > 0)
        C3 = int(len(C3) > 0)

        theta1 = p.getJointState(self.gripper, self.first_joint_idx[self.finger])[0]
        theta2 = p.getJointState(self.gripper, self.second_joint_idx[self.finger])[0]
        theta3 = p.getJointState(self.gripper, self.third_joint_idx[self.finger])[0]

        L1 = int((Limits[0] - theta1) <= tol)
        L2 = int((Limits[1] - theta2) <= tol)

        L3 = int((Limits[2] - theta3) <= tol)
        L4 = int((theta3 - Limits[3]) <= tol)

        return [C1, C2, C3, L1, L2, L3, L4]
    
    def joint_angles(self):
        theta1 = p.getJointState(self.gripper, self.first_joint_idx[self.finger])[0]
        theta2 = p.getJointState(self.gripper, self.second_joint_idx[self.finger])[0]
        theta3 = p.getJointState(self.gripper, self.third_joint_idx[self.finger])[0]
        return [theta1, theta2, theta3]
    
    def close(self, close_vel=1, close_force=1, grasp_force=100):
        """
        Close the gripper.
        """
        self.close_vel = close_vel
        self.close_force = close_force
        self.grasp_force = grasp_force
        # if self.finger == 2:
        #     self.close_force *= 2
        #     self.grasp_force *= 2
        self.sensors = self.ReadSensor()
        # print(f"finger {self.finger + 1} sensor: {self.sensors} state: {self.current_state}")
        # print(f"finger {self.finger + 1} joint 1 and 3 difference: {p.getJointState(self.gripper, self.first_joint_idx[self.finger])[0] + p.getJointState(self.gripper, self.third_joint_idx[self.finger])[0]}")
        # print(f"finger {self.finger + 1} lsat sensor: {self.sensors[6]}")
        # Handle state transitions and actions based on current state and sensors
        if self.current_state == self.J1_Closing_J3_Opening:
            self.on_j1_closing_j3_opening()
        elif self.current_state == self.J2_Closing:
            self.on_j2_closing()
        elif self.current_state == self.J3_Closing:
            self.on_j3_closing()
        elif self.current_state == self.FingerStopped:
            self.stop_finger()
        elif self.current_state == self.resume_j1_closing_j3_opening:
            self.on_j1_closing_j3_opening()

    def control_joint(self, joint_index, targetVel, force, mode="close"):
        """
        Control a specific joint of the gripper.
        """
        
        ## Uncomment the following line to control the gripper using position control
        # if mode == "close":
        #     targetPos = p.getJointState(self.gripper, joint_index)[0] + 0.01
        # else:
        #     targetPos = p.getJointState(self.gripper, joint_index)[0] - 0.01
        # p.setJointMotorControl2(bodyUniqueId=self.gripper,
        #                         jointIndex=joint_index,
        #                         controlMode=p.POSITION_CONTROL,
        #                         targetPosition=targetPos,
        #                         targetVelocity=0,
        #                         force=force,
        # )
        
        ## Uncomment the following line to control the gripper using velocity control
        p.setJointMotorControl2(bodyUniqueId=self.gripper, 
                                jointIndex=joint_index, 
                                controlMode=p.VELOCITY_CONTROL,
                                targetVelocity = targetVel,
                                force=force,
        )
        ## Uncomment the following line to control the gripper using torque control
        # p.setJointMotorControl2(bodyUniqueId=self.gripper, 
        #                         jointIndex=joint_index, 
        #                         controlMode=p.TORQUE_CONTROL,
        #                         targetVelocity = targetVel,
        #                         force=force,
        # )
    
    def on_j1_closing_j3_opening(self):
        # Logic for closing j1 and opening j3
        if self.sensors[2]:
            self.stop_finger_from_j1()
        elif self.sensors[1]:
            self.start_j3_closing_from_j1()
        elif self.sensors[0]:
            self.start_j2_closing()
        elif self.sensors[3] and self.sensors[6]:
            self.start_j2_closing()
        else:
            self.control_joint(joint_index=self.first_joint_idx[self.finger], targetVel=self.close_vel, force=self.close_force)  # example force value for closing j1
            self.control_joint(joint_index=self.third_joint_idx[self.finger], targetVel=-self.close_vel, force=self.close_force)   # example force value for opening j3

    def on_j2_closing(self):
        # Logic for closing j2
        if self.sensors[2] or self.sensors[4]:
            self.stop_finger_from_j2()
        elif self.sensors[1]:
            self.start_j3_closing_from_j2()
        else:
            self.control_joint(joint_index=self.second_joint_idx[self.finger], 
                               targetVel=self.close_vel, force=self.close_force)  # example force value for closing j2

    def on_j3_closing(self):
        # Logic for closing j3
        # if self.sensors[2] or self.sensors[5]:
        if self.sensors[5]:
            self.stop_finger_from_j3()
        elif self.sensors[2] and not self.sensors[1]:
            self.stop_finger_from_j3()
        else:
            self.control_joint(joint_index=self.third_joint_idx[self.finger], 
                               targetVel=self.close_vel, force=self.close_force)  # example force value for closing j3

    def stop_finger(self):
        if self.sensors[1]:
            self.resume_j3_closing_from_j2()
        elif self.sensors[2]:
            # get the current state of each joint
            joints = self.joint_angles()

            # stop the finger
            p.setJointMotorControl2(bodyUniqueId=self.gripper, 
                                    jointIndex=self.first_joint_idx[self.finger], 
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition = joints[0],
                                    targetVelocity = 0,
                                    force=self.grasp_force,
            )
            p.setJointMotorControl2(bodyUniqueId=self.gripper, 
                                    jointIndex=self.second_joint_idx[self.finger], 
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition = joints[1],
                                    targetVelocity = 0,
                                    force=self.grasp_force,
            )
            p.setJointMotorControl2(bodyUniqueId=self.gripper, 
                                    jointIndex=self.third_joint_idx[self.finger], 
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition = joints[2],
                                    targetVelocity = 0,
                                    force=self.grasp_force,
            )
        else:
            self.resume_j1_closing_j3_opening()

