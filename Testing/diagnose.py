import numpy as np
from humanoid_library import AdamLiteEnv

SCENE_PATH = r"C:\Anvay\Github\WalkingRobot\humanoid_library\humanoid_sim\scene.xml"

env = AdamLiteEnv(scene_xml_path=SCENE_PATH, render_mode="human")

print("=" * 60)
print("MUJOCO MODEL DIAGNOSTICS")
print("=" * 60)

# Check basic model info
print(f"\nNumber of actuators (nu): {env.model.nu}")
print(f"Number of joints (nq): {env.model.nq}")
print(f"Number of DOF (nv): {env.model.nv}")

# Print actuator information
if env.model.nu > 0:
    print(f"\n{'Idx':<4} {'Actuator Name':<35} {'Control Range':<20}")
    print("-" * 60)
    for i in range(env.model.nu):
        try:
            name = env.model.actuator(i).name or f"actuator_{i}"
        except:
            name = f"actuator_{i}"
        ctrl_range = env.model.actuator_ctrlrange[i]
        print(f"{i:<4} {name:<35} [{ctrl_range[0]:>8.2f}, {ctrl_range[1]:>8.2f}]")
    
    print(f"\nAction space shape: {env.action_space.shape}")
    print(f"Action space low:  {env.action_space.low[:5]}... (first 5)")
    print(f"Action space high: {env.action_space.high[:5]}... (first 5)")
else:
    print("\n⚠️  NO ACTUATORS FOUND! The robot has no motors.")

# Test if controls actually affect the simulation
print("\n" + "=" * 60)
print("TESTING CONTROL EFFECT")
print("=" * 60)

obs, _ = env.reset()
initial_qpos = obs[:env.model.nq].copy()
print(f"\nInitial joint positions (first 10): {initial_qpos[:10]}")

# Apply maximum control to first actuator
if env.model.nu > 0:
    test_action = np.zeros(env.model.nu)
    
    # Test with maximum value from action space
    test_action[0] = env.action_space.high[0]
    
    print(f"\nApplying control to actuator 0...")
    print(f"Control value: {test_action[0]:.2f}")
    
    # Step simulation 200 times
    for step in range(200):
        obs, _, _, _, _ = env.step(test_action)
        if step % 50 == 0:
            current_qpos = obs[:env.model.nq]
            print(f"Step {step:3d}: qpos[0:5] = {current_qpos[:5]}")
    
    final_qpos = obs[:env.model.nq].copy()
    
    # Check if anything changed
    qpos_diff = np.abs(final_qpos - initial_qpos)
    max_change = np.max(qpos_diff)
    changed_indices = np.where(qpos_diff > 0.001)[0]
    
    print(f"\n" + "=" * 60)
    print(f"RESULTS:")
    print(f"=" * 60)
    print(f"Max joint position change: {max_change:.6f}")
    
    if max_change < 0.001:
        print("\n❌ WARNING: Controls are NOT affecting the robot!")
        print("   Possible causes:")
        print("   1. Actuators have gain=0 or kp=0 in the XML")
        print("   2. Joints are locked or have limited range")
        print("   3. Control values are outside expected range")
    else:
        print(f"\n✅ Controls ARE working! Joints moved.")
        print(f"   Changed joint indices: {changed_indices}")
        print(f"   Position changes: {qpos_diff[changed_indices]}")
        
        # Now test your actual windmill motion
        print("\n" + "=" * 60)
        print("TESTING YOUR WINDMILL MOTION")
        print("=" * 60)
        
        obs, _ = env.reset()
        initial_qpos = obs[:env.model.nq].copy()
        
        for i in range(200):
            action = np.zeros(25)
            t = i * env.dt
            
            # Your windmill code
            action[15] = 60 * np.sin(t * 2)
            action[16] = 60 * np.sin(t * 2)
            
            obs, _, _, _, _ = env.step(action)
        
        final_qpos = obs[:env.model.nq].copy()
        windmill_diff = np.abs(final_qpos - initial_qpos)
        max_windmill_change = np.max(windmill_diff)
        
        print(f"\nAfter windmill motion:")
        print(f"Max joint change: {max_windmill_change:.6f}")
        print(f"Control values were: action[15]=±{60:.1f}, action[16]=±{60:.1f}")
        print(f"Action space range: [{env.action_space.low[15]:.1f}, {env.action_space.high[15]:.1f}]")
        
        if max_windmill_change < 0.001:
            print("\n❌ Your windmill values (±60) are likely OUTSIDE the control range!")
            print(f"   Try scaling down to match action space: ±{env.action_space.high[15]:.1f}")
else:
    print("\n⚠️  Cannot test - no actuators in model!")

print("\n" + "=" * 60)

env.close()