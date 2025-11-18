import os
import time
import numpy as np
import mujoco
import mujoco.viewer
import gymnasium as gym
from gymnasium import spaces

class AdamLiteEnv(gym.Env):
    """Gymnasium wrapper for Adam Lite (MuJoCo Menagerie).
       Renders using mujoco.viewer.launch_passive.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, scene_xml_path: str, render_mode="human"):
        super().__init__()
        assert os.path.exists(scene_xml_path), f"scene file not found: {scene_xml_path}"

        self.scene_xml_path = scene_xml_path
        self.render_mode = render_mode

        # Load model & data
        self.model = mujoco.MjModel.from_xml_path(self.scene_xml_path)
        self.data = mujoco.MjData(self.model)

        # timestep
        self.dt = self.model.opt.timestep

        # Build action / obs spaces
        # Use actuators' ctrlrange for action space if available, otherwise [-1,1]
        if self.model.nu > 0:
            ctrl_low = np.array([r[0] for r in self.model.actuator_ctrlrange])
            ctrl_high = np.array([r[1] for r in self.model.actuator_ctrlrange])
            # If ctrlrange all zeros (some models), fallback to symmetric range
            if np.all(ctrl_low == ctrl_high):
                ctrl_low = -1.0 * np.ones(self.model.nu, dtype=np.float32)
                ctrl_high = 1.0 * np.ones(self.model.nu, dtype=np.float32)
            self.action_space = spaces.Box(low=ctrl_low.astype(np.float32),
                                           high=ctrl_high.astype(np.float32),
                                           dtype=np.float32)
        else:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(0,), dtype=np.float32)

        # Observation: concatenated qpos (positions) and qvel (velocities)
        obs_dim = self.model.nq + self.model.nv
        high = np.finfo(np.float32).max
        self.observation_space = spaces.Box(low=-high, high=high, shape=(obs_dim,), dtype=np.float32)

        # viewer handle (passive viewer) created on first render
        self._viewer = None
        
        # Frame timing for passive viewer
        self._last_render_time = 0.0
        self._render_fps = self.metadata["render_fps"]
        self._frame_duration = 1.0 / self._render_fps

        # Convenience: index mapping
        self._qpos_start = 0
        self._qvel_start = 0

        # initialize
        self.reset()

    def step(self, action):
        # Clip action to action_space and write to ctrl
        action = np.asarray(action, dtype=np.float64)
        if self.model.nu:
            clipped = np.clip(action, self.action_space.low, self.action_space.high)
            # model.ctrl is shape (nu,)
            self.data.ctrl[:] = clipped

        # advance physics one step (you can step multiple times per action if you want)
        mujoco.mj_step(self.model, self.data)

        # obs
        qpos = self.data.qpos.ravel().copy()
        qvel = self.data.qvel.ravel().copy()
        obs = np.concatenate([qpos, qvel]).astype(np.float32)

        # simple placeholder reward: forward velocity of root body (change to suit)
        # find root body (pelvis) forward vel in world frame: use site or body sensor if present
        # fallback: use qvel of root generalized velocity for an approximate measure
        # here we measure global x velocity of body 0 (world) -> often you want to query a specific body.
        root_body_xvel = 0.0
        try:
            # many menagerie scenes name the base 'pelvis' or similar; find by name if available
            if "pelvis" in self.model.body_names:
                idx = self.model.body_name2id("pelvis")
                root_body_xvel = self.data.sensordata[idx] if len(self.data.sensordata) > idx else 0.0
        except Exception:
            root_body_xvel = 0.0

        reward = float(root_body_xvel)

        terminated = False
        truncated = False
        info = {}

        # If render mode human, sync viewer with frame rate limiting
        if self.render_mode == "human":
            self._sync_viewer()

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        # reset state to model qpos0 and small noise
        # Set qpos = default qpos in model (model.key_qpos0 if present), otherwise zeros
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        qpos = self.data.qpos.ravel().copy()
        qvel = self.data.qvel.ravel().copy()
        obs = np.concatenate([qpos, qvel]).astype(np.float32)

        if self.render_mode == "human":
            self._sync_viewer()

        return obs, {}

    def render(self):
        # For Gym compatibility (calls viewer sync too)
        if self.render_mode == "human":
            self._sync_viewer()
        return None

    def _sync_viewer(self):
        # lazy-create the passive viewer
        if self._viewer is None:
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            
            # Set camera parameters directly on viewer's cam object
            # This is the public API and works reliably
            self._viewer.cam.azimuth = 165       # rotation around vertical axis
            self._viewer.cam.elevation = -20     # camera tilt (pitch)
            self._viewer.cam.distance = 2.5      # zoom distance
            self._viewer.cam.lookat[:] = [0, 0, 1]  # point to look at (x, y, z)
            
            self._last_render_time = time.time()
        
        # Frame rate limiting: only sync at target FPS
        current_time = time.time()
        elapsed = current_time - self._last_render_time
        
        if elapsed >= self._frame_duration:
            # sync viewer to current sim state
            self._viewer.sync()
            self._last_render_time = current_time
        else:
            # Sleep to maintain consistent frame rate
            sleep_time = self._frame_duration - elapsed
            time.sleep(sleep_time)
            self._viewer.sync()
            self._last_render_time = time.time()

    def close(self):
        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception:
                pass
            self._viewer = None

    def seed(self, seed=None):
        np.random.seed(seed)