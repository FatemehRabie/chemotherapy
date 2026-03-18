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

def plot_2d_slices(states_dict, time_steps, slice_idx=16, save_path="comparison_2d.png", actions_dict=None):
    """
    Plots 2D mid-slices of the 3D tumor, immune, and drug populations over time
    for different treatment regimens. Adds shared colorbars, compact spacing,
    and a time-series panel (with reference vertical dashed lines) for each
    regimen showing the total tumor, immune, and drug concentrations.
    """
    regimens = list(states_dict.keys())
    num_regimens = len(regimens)
    num_times = len(time_steps)

    # Collect handles for shared colorbars and later label alignment
    tumor_axes, immune_axes, drug_axes = [], [], []
    tumor_im = immune_im = drug_im = None
    label_refs = []  # (regimen_name, top_axis, bottom_axis)
    line_axes = []
    cbar_slot_axes = []

    # Figure geometry: extra column for the time-series plots
    fig_width = 2.0 * num_times + 10.0  # extra width to separate line plots from heatmaps
    fig_height = 2.0 * num_regimens * 3
    fig = plt.figure(figsize=(fig_width, fig_height))
    outer_gs = fig.add_gridspec(
        nrows=num_regimens,
        ncols=1,
        height_ratios=[3] * num_regimens,
        hspace=0.18,  # tighter vertical spacing
    )

    for i, regimen in enumerate(regimens):
        states = states_dict[regimen]
        total_steps = len(states)

        # Split each regimen into a heatmap block, a shared colorbar lane, and a line plot.
        regimen_gs = outer_gs[i].subgridspec(
            nrows=1,
            ncols=3,
            width_ratios=[num_times * 0.9, 0.85, 2.25],
            wspace=0.05,
        )
        heatmap_gs = regimen_gs[0, 0].subgridspec(
            nrows=3,
            ncols=num_times,
            wspace=0.0,  # keep subplot boxes flush; reduced block width removes visible gaps
            hspace=0.035,
        )
        cbar_slot_ax = fig.add_subplot(regimen_gs[0, 1])
        cbar_slot_ax.axis("off")
        cbar_slot_axes.append(cbar_slot_ax)

        # Heatmap panels
        top_axis = bottom_axis = None
        for j, t_ratio in enumerate(time_steps):
            step_idx = min(int(t_ratio * total_steps), total_steps - 1)
            state = states[step_idx]

            tumor_slice = state[1, slice_idx, :, :]  # Index 1 is Tumor
            immune_slice = state[2, slice_idx, :, :]  # Index 2 is Immune
            drug_slice = state[3, slice_idx, :, :]  # Index 3 is Drug

            # Plot Tumor
            ax_t = fig.add_subplot(heatmap_gs[0, j])
            im = ax_t.imshow(tumor_slice, cmap="Reds", vmin=0, vmax=1)
            if tumor_im is None:
                tumor_im = im
            tumor_axes.append(ax_t)
            if j == 0:
                ax_t.set_ylabel("Tumor", fontsize=12)
            if i == 0:
                ax_t.set_title(f"{int(t_ratio*100)}%", fontsize=12)
            ax_t.axis("off")
            if j == 0:
                top_axis = ax_t

            # Plot Immune
            ax_i = fig.add_subplot(heatmap_gs[1, j])
            im = ax_i.imshow(immune_slice, cmap="Blues", vmin=0, vmax=1)
            if immune_im is None:
                immune_im = im
            immune_axes.append(ax_i)
            if j == 0:
                ax_i.set_ylabel("Immune", fontsize=12)
            ax_i.axis("off")

            # Plot Drug
            ax_d = fig.add_subplot(heatmap_gs[2, j])
            im = ax_d.imshow(drug_slice, cmap="Greens", vmin=0, vmax=1)
            if drug_im is None:
                drug_im = im
            drug_axes.append(ax_d)
            if j == 0:
                ax_d.set_ylabel("Drug", fontsize=12)
            ax_d.axis("off")
            if j == 0:
                bottom_axis = ax_d

        # Time-series panel spanning the three rows
        line_ax = fig.add_subplot(regimen_gs[0, 2])  # dedicated column for line plot
        line_axes.append(line_ax)
        time_axis = np.linspace(0, 1, total_steps)
        normal_curve = [s[0].sum() for s in states]
        tumor_curve = [s[1].sum() for s in states]
        immune_curve = [s[2].sum() for s in states]
        drug_curve = [s[3].sum() for s in states]

        line_ax.plot(time_axis, normal_curve, color="dimgray", label="Normal")
        line_ax.plot(time_axis, tumor_curve, color="firebrick", label="Tumor")
        line_ax.plot(time_axis, immune_curve, color="royalblue", label="Immune")
        line_ax.plot(time_axis, drug_curve, color="seagreen", label="Drug")

        for t_ratio in time_steps:
            line_ax.axvline(t_ratio, color="gray", linestyle="--", linewidth=1, alpha=0.6)

        line_ax.set_xlim(0, 1)
        line_ax.set_xlabel("Episode progress", fontsize=10)
        line_ax.set_ylabel("Total population / concentration", fontsize=10)
        line_ax.grid(alpha=0.3, linestyle=":")

        line_ax.legend(
            fontsize=9,
            loc="upper left",
            bbox_to_anchor=(1.18, 1.02),
            frameon=False,
            borderaxespad=0.0,
        )

        # Store axes to align regimen label after layout
        label_refs.append((regimen, top_axis, bottom_axis))

    plt.tight_layout(pad=0.15, rect=[0.038, 0.0, 0.98, 1])
    fig.canvas.draw()

    # Place the three shared colorbars in a dedicated vertical lane with fixed gaps.
    if cbar_slot_axes:
        lane_x0 = min(ax.get_position().x0 for ax in cbar_slot_axes)
        lane_x1 = max(ax.get_position().x1 for ax in cbar_slot_axes)
        lane_y0 = min(ax.get_position().y0 for ax in cbar_slot_axes)
        lane_y1 = max(ax.get_position().y1 for ax in cbar_slot_axes)

        lane_width = lane_x1 - lane_x0
        lane_height = lane_y1 - lane_y0
        bar_width = min(0.014, lane_width * 0.22)
        bar_height = lane_height * 0.18
        bar_gap = lane_height * 0.07
        total_stack_height = 3 * bar_height + 2 * bar_gap
        stack_y0 = lane_y0 + (lane_height - total_stack_height) / 2
        bar_x0 = lane_x0 + lane_width * 0.18
        label_x = bar_x0 + bar_width + lane_width * 0.22

        def _add_cbar(im, label, idx):
            y0 = stack_y0 + (2 - idx) * (bar_height + bar_gap)
            cax = fig.add_axes([bar_x0, y0, bar_width, bar_height])
            fig.colorbar(im, cax=cax)
            fig.text(
                label_x,
                y0 + bar_height / 2,
                label,
                rotation=270,
                ha="center",
                va="center",
                fontsize=10,
            )

        if tumor_im is not None:
            _add_cbar(tumor_im, "Tumor density", 0)
        if immune_im is not None:
            _add_cbar(immune_im, "Immune density", 1)
        if drug_im is not None:
            _add_cbar(drug_im, "Drug concentration", 2)

    # Add regimen labels aligned to the center of the three heatmap rows
    for regimen, top_ax, bottom_ax in label_refs:
        if top_ax is None or bottom_ax is None:
            continue
        top_bb = top_ax.get_position()
        bottom_bb = bottom_ax.get_position()
        mid_y = (top_bb.y1 + bottom_bb.y0) / 2
        fig.text(
            0.034,
            mid_y,
            regimen,
            ha="left",
            va="center",
            fontsize=14,
            fontweight="bold",
            rotation=90,
        )

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved 2D slices to {save_path}")

def plot_3d_render(state, title, save_path, threshold=0.2):
    """
    Creates 3D scatter plot renderings of tumor mass, immune cells, and drug
    concentration side by side.
    """
    tumor_vol = state[1, :, :, :]
    immune_vol = state[2, :, :, :]
    drug_vol = state[3, :, :, :]
    
    fig = plt.figure(figsize=(21, 7))
    
    # Plot Tumor
    ax_tumor = fig.add_subplot(131, projection='3d')
    z_t, y_t, x_t = np.where(tumor_vol > threshold)
    c_t = tumor_vol[tumor_vol > threshold]
    
    sc_tumor = ax_tumor.scatter(x_t, y_t, z_t, c=c_t, cmap='Reds', marker='o', alpha=0.6)
    fig.colorbar(sc_tumor, ax=ax_tumor, label='Tumor Density', shrink=0.5)
    
    ax_tumor.set_title(f"{title} - Tumor", fontsize=16)
    ax_tumor.set_xlim(0, 31)
    ax_tumor.set_ylim(0, 31)
    ax_tumor.set_zlim(0, 31)
    ax_tumor.set_xlabel('X', fontsize=14)
    ax_tumor.set_ylabel('Y', fontsize=14)
    ax_tumor.set_zlabel('Z', fontsize=14)
    
    # Plot Immune
    ax_immune = fig.add_subplot(132, projection='3d')
    z_i, y_i, x_i = np.where(immune_vol > threshold)
    c_i = immune_vol[immune_vol > threshold]

    sc_immune = ax_immune.scatter(x_i, y_i, z_i, c=c_i, cmap='Blues', marker='o', alpha=0.6)
    fig.colorbar(sc_immune, ax=ax_immune, label='Immune Density', shrink=0.5)

    ax_immune.set_title(f"{title} - Immune", fontsize=16)
    ax_immune.set_xlim(0, 31)
    ax_immune.set_ylim(0, 31)
    ax_immune.set_zlim(0, 31)
    ax_immune.set_xlabel('X', fontsize=14)
    ax_immune.set_ylabel('Y', fontsize=14)
    ax_immune.set_zlabel('Z', fontsize=14)

    # Plot Drug
    ax_drug = fig.add_subplot(133, projection='3d')
    z_d, y_d, x_d = np.where(drug_vol > threshold)
    c_d = drug_vol[drug_vol > threshold]
    
    sc_drug = ax_drug.scatter(x_d, y_d, z_d, c=c_d, cmap='Greens', marker='o', alpha=0.6)
    fig.colorbar(sc_drug, ax=ax_drug, label='Drug Concentration', shrink=0.5)
    
    ax_drug.set_title(f"{title} - Drug", fontsize=16)
    ax_drug.set_xlim(0, 31)
    ax_drug.set_ylim(0, 31)
    ax_drug.set_zlim(0, 31)
    ax_drug.set_xlabel('X', fontsize=14)
    ax_drug.set_ylabel('Y', fontsize=14)
    ax_drug.set_zlabel('Z', fontsize=14)
    
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
    mtd_states, mtd_actions, mtd_reward = run_episode(env, policy=mtd_policy)
    
    print("Running Metronomic Baseline (FixedSchedule, low dose)...")
    metronomic_policy = make_baseline_policy(
        "FIXEDSCHEDULE",
        env.action_space,
        duration=int(env.action_space.nvec[0] - 1),
        dose=2,
        drug_type=0,
    )
    metro_states, metro_actions, metro_reward = run_episode(env, policy=metronomic_policy)

    print("Running Proportional Control Baseline...")
    proportional_policy = make_baseline_policy(
        "PROPORTIONALCONTROL",
        env.action_space,
        duration_scale=4.0,
        dose_scale=8.0,
        drug_type=0,
    )
    proportional_states, proportional_actions, proportional_reward = run_episode(env, policy=proportional_policy)

    print("Running Random Policy Baseline...")
    random_policy = make_baseline_policy(
        "RANDOMPOLICY",
        env.action_space,
        seed=42,
    )
    random_states, random_actions, random_reward = run_episode(env, policy=random_policy)

    # 3. Generate 2D Comparison Mosaics
    states_dict = {
        "Optimized DRL": drl_states,
        "MTD Baseline": mtd_states,
        "Metronomic": metro_states,
        "Proportional Control": proportional_states,
        "Random Policy": random_states,
    }
    actions_dict = {
        "Optimized DRL": drl_actions,
        "MTD Baseline": mtd_actions,
        "Metronomic": metro_actions,
        "Proportional Control": proportional_actions,
        "Random Policy": random_actions,
    }
    # Plot at 0%, 33%, 66%, and 100% of the episode
    plot_2d_slices(
        states_dict,
        time_steps=time_points,
        save_path="treatment_comparison_2d.png",
        actions_dict=actions_dict,
    )
    
    # 4. Generate 3D Renderings across the episode
    plot_3d_timepoints(states_dict, time_steps=time_points, out_dir="3d_renders")
    
    print("Visualization generation complete.")
