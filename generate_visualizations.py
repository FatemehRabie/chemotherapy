import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import gymnasium as gym
from stable_baselines3 import PPO
from baselines import make_baseline_policy


def _extract_spatial_state(obs):
    """Return the 4x32x32x32 tensor from the environment observation."""
    if isinstance(obs, tuple):
        obs = obs[0]
    if isinstance(obs, dict):
        if "variables" in obs:
            return obs["variables"]
        if "spatial" in obs:
            return obs["spatial"]
        # Fallback to the first value to avoid hard failures on unexpected keys
        return next(iter(obs.values()))
    return obs

# Assuming your environment is registered, or you can import it directly
# import your_env_module 

def run_episode(env, policy, seed=42):
    """
    Runs a single episode and collects the 3D spatial states over time.
    Accepts any policy with a Stable-Baselines3-compatible `predict` method.
    """
    obs, info = env.reset(seed=seed)  # Fixed seed for fair comparison
    
    states = []
    actions_taken = []
    rewards = []
    
    done = False
    while not done:
        # State shape is [4, 32, 32, 32] -> [Normal, Tumor, Immune, Drug]
        spatial_state = _extract_spatial_state(obs)
        states.append(spatial_state.copy())
        
        action, _ = policy.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        actions_taken.append(action)
        rewards.append(reward)
        done = terminated or truncated

    # Append final state
    spatial_state = _extract_spatial_state(obs)
    states.append(spatial_state.copy())
    
    return states, actions_taken, sum(rewards)

def plot_2d_slices(states_dict, time_steps, slice_idx=16, save_path="comparison_2d.png"):
    """
    Plots 2D mid-slices of the 3D tumor and immune populations over time 
    for different treatment regimens.
    """
    regimens = list(states_dict.keys())
    num_regimens = len(regimens)
    num_times = len(time_steps)
    
    fig, axes = plt.subplots(num_regimens * 2, num_times, figsize=(3 * num_times, 3 * num_regimens * 2))
    
    for i, regimen in enumerate(regimens):
        # Left-margin label describing which regimen the next two rows belong to
        row_center = 1 - ((i * 2 + 1) / (num_regimens * 2))
        fig.text(0.02, row_center, regimen, ha="right", va="center", fontsize=13, fontweight="bold", rotation=90)

        states = states_dict[regimen]
        total_steps = len(states)
        
        for j, t_ratio in enumerate(time_steps):
            step_idx = min(int(t_ratio * total_steps), total_steps - 1)
            state = states[step_idx]
            
            tumor_slice = state[1, slice_idx, :, :]  # Index 1 is Tumor
            immune_slice = state[2, slice_idx, :, :] # Index 2 is Immune
            
            # Plot Tumor
            ax_t = axes[i * 2, j]
            ax_t.imshow(tumor_slice, cmap="Reds", vmin=0, vmax=1)
            if j == 0:
                ax_t.set_ylabel("Tumor", fontsize=12)
            if i == 0:
                ax_t.set_title(f"Progress: {int(t_ratio*100)}%", fontsize=12)
            ax_t.axis("off")
            
            # Plot Immune
            ax_i = axes[i * 2 + 1, j]
            ax_i.imshow(immune_slice, cmap="Blues", vmin=0, vmax=1)
            if j == 0:
                ax_i.set_ylabel("Immune", fontsize=12)
            ax_i.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved 2D slices to {save_path}")

def plot_3d_render(state, title, save_path, threshold=0.2):
    """
    Creates a 3D scatter plot rendering of the tumor mass.
    """
    tumor_vol = state[1, :, :, :]
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract coordinates where tumor concentration is significant
    z, y, x = np.where(tumor_vol > threshold)
    c = tumor_vol[tumor_vol > threshold]
    
    sc = ax.scatter(x, y, z, c=c, cmap='Reds', marker='o', alpha=0.6)
    fig.colorbar(sc, ax=ax, label='Tumor Density', shrink=0.5)
    
    ax.set_title(title, fontsize=14)
    ax.set_xlim(0, 31)
    ax.set_ylim(0, 31)
    ax.set_zlim(0, 31)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved 3D render to {save_path}")


def plot_3d_timepoints(states_dict, time_steps, out_dir="3d_renders", threshold=0.2):
    """
    Saves 3D renders at multiple episode progress points for each regimen.
    """
    os.makedirs(out_dir, exist_ok=True)
    for regimen, states in states_dict.items():
        total_steps = len(states)
        slug = regimen.lower().replace(" ", "_").replace("/", "_")
        for t_ratio in time_steps:
            step_idx = min(int(t_ratio * total_steps), total_steps - 1)
            title = f"{regimen} - {int(t_ratio*100)}%"
            filename = os.path.join(out_dir, f"{slug}_{int(t_ratio*100)}.png")
            plot_3d_render(states[step_idx], title, filename, threshold=threshold)

if __name__ == "__main__":
    # 1. Initialize environment
    if "ReactionDiffusion-v0" not in gym.envs.registry:
        gym.envs.registration.register(
            id="ReactionDiffusion-v0",
            entry_point="env.reaction_diffusion:ReactionDiffusionEnv",
            kwargs={"render_mode": "human"}
        )
    env = gym.make("ReactionDiffusion-v0", render_mode="human") 
    
    # 2. Load the trained model (adjust path and algorithm as needed)
    model_path = "logs_PPO_0.0_baseline/best_model.zip"
    model = PPO.load(model_path)

    # Episode checkpoints to visualize
    time_points = [0.0, 0.33, 0.66, 1.0]

    print("Running DRL Agent...")
    drl_states, drl_actions, drl_reward = run_episode(env, policy=model)
    
    print("Running MTD Baseline (FixedSchedule)...")
    mtd_policy = make_baseline_policy(
        "FIXEDSCHEDULE",
        env.action_space,
        duration=int(env.action_space.nvec[0] - 1),
        dose=int(env.action_space.nvec[1] - 1),
        drug_type=0,
    )
    mtd_states, _, mtd_reward = run_episode(env, policy=mtd_policy)
    
    print("Running Metronomic Baseline (FixedSchedule, low dose)...")
    metronomic_policy = make_baseline_policy(
        "FIXEDSCHEDULE",
        env.action_space,
        duration=int(env.action_space.nvec[0] - 1),
        dose=2,
        drug_type=0,
    )
    metro_states, _, metro_reward = run_episode(env, policy=metronomic_policy)

    print("Running Proportional Control Baseline...")
    proportional_policy = make_baseline_policy(
        "PROPORTIONALCONTROL",
        env.action_space,
        duration_scale=4.0,
        dose_scale=8.0,
        drug_type=0,
    )
    proportional_states, _, proportional_reward = run_episode(env, policy=proportional_policy)

    print("Running Random Policy Baseline...")
    random_policy = make_baseline_policy(
        "RANDOMPOLICY",
        env.action_space,
        seed=42,
    )
    random_states, _, random_reward = run_episode(env, policy=random_policy)

    # 3. Generate 2D Comparison Mosaics
    states_dict = {
        "Optimized DRL": drl_states,
        "MTD Baseline": mtd_states,
        "Metronomic": metro_states,
        "Proportional Control": proportional_states,
        "Random Policy": random_states,
    }
    # Plot at 0%, 33%, 66%, and 100% of the episode
    plot_2d_slices(states_dict, time_steps=time_points, save_path="treatment_comparison_2d.png")
    
    # 4. Generate 3D Renderings across the episode
    plot_3d_timepoints(states_dict, time_steps=time_points, out_dir="3d_renders")
    
    print("Visualization generation complete.")
