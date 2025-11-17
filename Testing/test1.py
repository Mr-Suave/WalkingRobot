# run_demo.py
import os
from humanoid_library import AdamLiteEnv
import time
import numpy as np

# change this to the path where you cloned the menagerie repo.
# recommended: point to the scene.xml file in the adam_lite folder, e.g.:
SCENE_PATH = r"C:\Anvay\Github\WalkingRobot\humanoid_library\humanoid_sim\scene.xml"

env = AdamLiteEnv(scene_xml_path=SCENE_PATH, render_mode="human")
obs, _ = env.reset()

try:
    for _ in range(5000):
        # simple random action (or zeros)
        if env.action_space.shape[0] > 0:
            a = env.action_space.sample()
        else:
            a = np.array([])
        obs, reward, terminated, truncated, info = env.step(a)
        time.sleep(env.dt)  # slow down to realtime-ish
finally:
    env.close()

