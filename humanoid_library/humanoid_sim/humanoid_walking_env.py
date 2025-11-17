import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import numpy as np
import os

class HumanoidWalkingEnv(gym.Env):
    """
    Gymnasium environment for training a humanoid to walk.
    Uses the simple_humanoid.urdf
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    
    def __init__(self, render_mode=None, urdf_path="simple_humanoid.urdf"):
        super().__init__()
        
        self.render_mode = render_mode
        self.urdf_path = urdf_path
        
        # Joint configuration
        self.joint_names = [
            "spine",
            "right_hip", "right_knee", "right_ankle",
            "left_hip", "left_knee", "left_ankle",
            "right_shoulder", "right_elbow",
            "left_shoulder", "left_elbow"
        ]
        self.num_joints = len(self.joint_names)
        
        # Observation: [joint_positions(11), joint_velocities(11), pelvis_pos(3), pelvis_ori(4), pelvis_vel(3)]
        # Total: 11 + 11 + 3 + 4 + 3 = 32
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(32,), dtype=np.float32
        )
        
        # Action: target joint positions (deltas or absolute)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_joints,), dtype=np.float32
        )
        
        # Physics client
        self.physics_client = None
        self.humanoid_id = None
        self.plane_id = None
        self.joint_indices = {}
        
        # Episode tracking
        self.steps = 0
        self.max_steps = 1000
        self.initial_pelvis_pos = None
        
        # Connect to PyBullet
        self._setup_pybullet()
    
    def _setup_pybullet(self):
        """Initialize PyBullet physics engine"""
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
        
        if self.render_mode == "human":
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1/240)  # 240 Hz for stability
        
        # Load ground plane
        self.plane_id = p.loadURDF("plane.urdf")
    
    def _load_humanoid(self):
        """Load humanoid URDF"""
        start_pos = [0, 0, 1.0]  # Start 1m above ground
        start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        
        self.humanoid_id = p.loadURDF(
            self.urdf_path,
            start_pos,
            start_orientation,
            useFixedBase=False,  # Free-floating base
            flags=p.URDF_USE_SELF_COLLISION
        )
        
        # Map joint names to indices
        self.joint_indices = {}
        for i in range(p.getNumJoints(self.humanoid_id)):
            joint_info = p.getJointInfo(self.humanoid_id, i)
            joint_name = joint_info[1].decode('utf-8')
            if joint_name in self.joint_names:
                self.joint_indices[joint_name] = i
        
        # Enable force/torque sensors on feet
        p.enableJointForceTorqueSensor(self.humanoid_id, self.joint_indices["right_ankle"])
        p.enableJointForceTorqueSensor(self.humanoid_id, self.joint_indices["left_ankle"])
        
        # Store initial position
        self.initial_pelvis_pos = np.array(start_pos)
    
    def _get_observation(self):
        """Get current state observation"""
        obs = []
        
        # Joint positions and velocities
        joint_positions = []
        joint_velocities = []
        
        for joint_name in self.joint_names:
            joint_idx = self.joint_indices[joint_name]
            joint_state = p.getJointState(self.humanoid_id, joint_idx)
            joint_positions.append(joint_state[0])  # position
            joint_velocities.append(joint_state[1])  # velocity
        
        obs.extend(joint_positions)
        obs.extend(joint_velocities)
        
        # Pelvis (base) state
        pelvis_pos, pelvis_ori = p.getBasePositionAndOrientation(self.humanoid_id)
        pelvis_vel, pelvis_ang_vel = p.getBaseVelocity(self.humanoid_id)
        
        obs.extend(pelvis_pos)  # x, y, z
        obs.extend(pelvis_ori)  # quaternion (x, y, z, w)
        obs.extend(pelvis_vel)  # linear velocity
        
        return np.array(obs, dtype=np.float32)
    
    def _apply_action(self, action):
        """Apply action to joints with PD control"""
        # Scale actions to reasonable joint ranges
        action_scaled = action * np.pi  # Scale to [-pi, pi]
        
        for i, joint_name in enumerate(self.joint_names):
            joint_idx = self.joint_indices[joint_name]
            target_pos = action_scaled[i]
            
            # PD control parameters
            kp = 100.0  # Position gain
            kd = 10.0   # Velocity gain
            max_force = 200.0
            
            p.setJointMotorControl2(
                bodyUniqueId=self.humanoid_id,
                jointIndex=joint_idx,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_pos,
                positionGain=kp,
                velocityGain=kd,
                force=max_force
            )
    
    def _compute_reward(self):
        """Compute reward for walking task"""
        # Get pelvis state
        pelvis_pos, pelvis_ori = p.getBasePositionAndOrientation(self.humanoid_id)
        pelvis_vel, pelvis_ang_vel = p.getBaseVelocity(self.humanoid_id)
        
        pelvis_pos = np.array(pelvis_pos)
        pelvis_vel = np.array(pelvis_vel)
        
        # Reward components
        
        # 1. Forward velocity (main objective)
        forward_reward = pelvis_vel[0]  # X-axis is forward
        
        # 2. Upright penalty (penalize tilting)
        # Quaternion to euler
        euler = p.getEulerFromQuaternion(pelvis_ori)
        upright_penalty = -0.5 * (euler[0]**2 + euler[1]**2)  # Penalize roll and pitch
        
        # 3. Height penalty (stay around 0.8-1.0m)
        target_height = 0.9
        height_penalty = -0.1 * (pelvis_pos[2] - target_height)**2
        
        # 4. Energy penalty (minimize action magnitude)
        # We'll penalize large joint velocities
        joint_velocities = []
        for joint_name in self.joint_names:
            joint_idx = self.joint_indices[joint_name]
            joint_state = p.getJointState(self.humanoid_id, joint_idx)
            joint_velocities.append(joint_state[1])
        energy_penalty = -0.001 * np.sum(np.square(joint_velocities))
        
        # 5. Alive bonus
        alive_bonus = 1.0
        
        # Total reward
        reward = (
            forward_reward +
            upright_penalty +
            height_penalty +
            energy_penalty +
            alive_bonus
        )
        
        return reward, {
            "forward_reward": forward_reward,
            "upright_penalty": upright_penalty,
            "height_penalty": height_penalty,
            "energy_penalty": energy_penalty
        }
    
    def _is_terminated(self):
        """Check if episode should terminate"""
        pelvis_pos, pelvis_ori = p.getBasePositionAndOrientation(self.humanoid_id)
        
        # Terminate if fallen (pelvis too low or too tilted)
        if pelvis_pos[2] < 0.4:  # Below 40cm
            return True
        
        # Check if too tilted
        euler = p.getEulerFromQuaternion(pelvis_ori)
        if abs(euler[0]) > 0.8 or abs(euler[1]) > 0.8:  # ~45 degrees
            return True
        
        return False
    
    def reset(self, seed=None, options=None):
        """Reset environment"""
        super().reset(seed=seed)
        
        # Reset simulation
        p.resetSimulation(physicsClientId=self.physics_client)
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1/240)
        
        # Reload objects
        self.plane_id = p.loadURDF("plane.urdf")
        self._load_humanoid()
        
        # Reset tracking
        self.steps = 0
        
        # Get initial observation
        obs = self._get_observation()
        info = {}
        
        return obs, info
    
    def step(self, action):
        """Execute one step"""
        # Apply action
        self._apply_action(action)
        
        # Step simulation (multiple substeps for stability)
        for _ in range(4):  # 4 substeps = 240Hz / 60Hz
            p.stepSimulation()
        
        # Get observation
        obs = self._get_observation()
        
        # Compute reward
        reward, reward_info = self._compute_reward()
        
        # Check termination
        terminated = self._is_terminated()
        truncated = self.steps >= self.max_steps
        
        self.steps += 1
        
        # Info
        info = reward_info.copy()
        pelvis_pos, _ = p.getBasePositionAndOrientation(self.humanoid_id)
        info["distance_traveled"] = pelvis_pos[0] - self.initial_pelvis_pos[0]
        
        return obs, reward, terminated, truncated, info
    
    def render(self):
        """Render the environment"""
        if self.render_mode == "rgb_array":
            # Get camera image
            view_matrix = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=[0, 0, 1],
                distance=3.0,
                yaw=45,
                pitch=-30,
                roll=0,
                upAxisIndex=2
            )
            proj_matrix = p.computeProjectionMatrixFOV(
                fov=60, aspect=1.0, nearVal=0.1, farVal=100.0
            )
            (_, _, px, _, _) = p.getCameraImage(
                width=640,
                height=480,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                renderer=p.ER_BULLET_HARDWARE_OPENGL
            )
            rgb_array = np.array(px, dtype=np.uint8)
            rgb_array = np.reshape(rgb_array, (480, 640, 4))[:, :, :3]
            return rgb_array
        elif self.render_mode == "human":
            # GUI mode - rendering is automatic
            pass
    
    def close(self):
        """Clean up"""
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
            self.physics_client = None


