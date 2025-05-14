import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium.utils import seeding  # Updated seeding import
from phi.field import CenteredGrid
from phi.flow import ZERO_GRADIENT, Box, math, iterate, batch
from simulator.dynamical_system import reaction_diffusion
from simulator.params import *

class ReactionDiffusionEnv(gym.Env):
    """
    A custom Gym environment simulating reaction diffusion.
    """

    metadata = {'render_modes': ['human']}  # Specify supported render modes here

    def __init__(self, render_mode=None):
        self.render_mode = render_mode
        super(ReactionDiffusionEnv, self).__init__()
        # Load GDSC drug sensitivity data
        gdsc_data_path = 'Filtered_GDSC2_No_Duplicates_Averaged.xlsx'
        self.gdsc_data = pd.read_excel(gdsc_data_path)
        # Initialize the Ansarizadeh model parameters
        self.ansarizadeh = {'dn': 0.0, 'dtu': 0.001, 'di': 0.001, 'du': 0.001,
                            'r1': 1.0, 'r2': 1.5,
                            'a1': 0.1, 'a2': 0.3, 'a3': 0.2,
                            'b1': 1.0, 'b2': 1.0,
                            'c1': 1.0, 'c2': 1.0, 'c3': 0.5, 'c4': 1.0,
                            'd1': 0.2, 'd2': 1.0,
                            's': 0.33,
                            'rho': 0.01,
                            'alpha': 0.3}
        self.cancer_cell_lines = self.gdsc_data['CELL_LINE_NAME'].unique()
        self.drug_names = self.gdsc_data['DRUG_NAME'].unique()
        self.action_space = gym.spaces.MultiDiscrete([10, 11, len(self.drug_names)])
        self.observation_space = gym.spaces.Dict({'variables': gym.spaces.Box(low=0.0, high=np.inf, shape=(4, s_disc, s_disc, s_disc), dtype=np.float32),
                                                  'cell_line': gym.spaces.Discrete(len(self.cancer_cell_lines))})
        self.max_time = episode_time  # Set maximum number of time allowed per episode to prevent infinite loops.
        self.noise_level = observation_noise_level  # Inject noise into observations
        self.reset() # The environment supports random initial states for robust learning.

    def seed(self, seed=None):
        """
        Seeds the environment's random number generator for reproducibility.
        """
        self.np_random, seed = seeding.np_random(seed)  # Updated seed handling
        return [seed]

    def reset(self, seed = None, options = None):
        """
        Resets the environment to a random initial state, and resets the step counter.

        Parameters:
        - seed (Optional[int]): The seed for the random number generator.
        - options (dict, Optional): Additional information for environment initialization.

        Returns:
        - np.array: The initial observation.
        """
        """
        Resets the environment to its initial state.
        """
        if seed is not None:
            self.seed(seed) # The environment supports reproducibility through explicit seed handling.
        else:
            seed = self.seed()
        # Select cancer cell line and drug array
        if options is not None:
            random_cell_line = options.get('cell_line', None)
            d_array = options.get('diffusion', None)
        else:
            random_cell_line, d_array = None, None
        # Use provided options or randomly select values
        if random_cell_line is None:
            # Randomly select a cell line from the GDSC2 data
            random_cell_line = np.random.randint(0, high=len(self.cancer_cell_lines))
        if d_array is None:
            # Initialize the diffusion rates
            d_array = np.random.uniform(low=0.0, high=0.001, size=(3,))
        self.cell_line = self.cancer_cell_lines[random_cell_line]
        # Update the Ansarizadeh model parameters
        self.ansarizadeh.update({'dtu': d_array[0], 'di': d_array[1], 'du': d_array[2]})
        n0 = CenteredGrid(lambda x: 0.2 * math.exp(-2 * math.mean(x)**2) + np.random.exponential(scale=noise_scale), 
                          boundary = ZERO_GRADIENT, bounds = Box['x,y,z', -2:2, -2:2, -2:2], x=s_disc, y=s_disc, z=s_disc)
        tu0 = CenteredGrid(lambda x: 1 - 0.75 * math.cosh(math.mean(x))**-1 + np.random.exponential(scale=noise_scale), 
                           boundary = ZERO_GRADIENT, bounds = Box['x,y,z', -2:2, -2:2, -2:2], x=s_disc, y=s_disc, z=s_disc)
        i0 = CenteredGrid(lambda x: 0.375 - 0.235 * math.cosh(math.mean(x))**-2 + np.random.exponential(scale=noise_scale), 
                          boundary = ZERO_GRADIENT, bounds = Box['x,y,z', -2:2, -2:2, -2:2], x=s_disc, y=s_disc, z=s_disc)
        u0 = CenteredGrid(lambda x: math.cosh(math.mean(x))**-1 + np.random.exponential(scale=noise_scale), 
                          boundary = ZERO_GRADIENT, bounds = Box['x,y,z', -2:2, -2:2, -2:2], x=s_disc, y=s_disc, z=s_disc)  # Initial drug concentration
        # Reset the environment's state variables
        self.n = n0
        self.tu = tu0
        self.i = i0
        self.u = u0
        self.state = {'variables': np.array([n0.numpy() + self.noise_level * np.random.randn(s_disc, s_disc, s_disc),
                                             tu0.numpy() + self.noise_level * np.random.randn(s_disc, s_disc, s_disc),
                                             i0.numpy() + self.noise_level * np.random.randn(s_disc, s_disc, s_disc),
                                             u0.numpy() + self.noise_level * np.random.randn(s_disc, s_disc, s_disc)], dtype=np.float32), 
                      'cell_line': int(random_cell_line)}
        self.time = 0  # Reset time for the new episode
        return self.state, {'cell_line': self.cell_line}
        
    def step(self, action):
        """
        Applies an action to the environment, updating the state based on the action taken and microbial growth dynamics,
        calculates the reward, and checks if the episode should end.
        """
        info = {}  # Reset info dictionary at the beginning of each step
        counter = 0
        reward = 0.0    # Initialize reward
        # Filter and extract IC50 value for the random cell line and drug
        sensitivity_data = self.gdsc_data[(self.gdsc_data['CELL_LINE_NAME'] == self.cell_line) &
                                    (self.gdsc_data['DRUG_NAME'] == self.drug_names[action[2]])]
        a2 = 1.0 - sensitivity_data['AUC'].values[0] if not sensitivity_data.empty else 0.3
        self.ansarizadeh.update({'a2': a2})
        # Apply the action to the environment
        n_trj, tu_trj, i_trj, u_trj = iterate(reaction_diffusion, batch(time=(action[0]+1)*PDE_number_of_steps),
                                            self.n,
                                            self.tu,
                                            self.i,
                                            self.u,
                                            dt=PDE_step_length, f_kwargs=self.ansarizadeh, uc = 0.1*action[1],
                                            substeps=PDE_number_of_substeps)
        # Update state variables
        self.n = n_trj.time[-1]
        self.tu = tu_trj.time[-1]
        self.i = i_trj.time[-1]
        self.u = u_trj.time[-1]
        self.state.update({'variables': np.array([self.n.numpy() + self.noise_level * np.random.randn(s_disc, s_disc, s_disc),
                                                  self.tu.numpy() + self.noise_level * np.random.randn(s_disc, s_disc, s_disc),
                                                  self.i.numpy() + self.noise_level * np.random.randn(s_disc, s_disc, s_disc),
                                                  self.u.numpy() + self.noise_level * np.random.randn(s_disc, s_disc, s_disc)], dtype=np.float32)})
        # Reward function
        for time_step in range((action[0]+1)*PDE_number_of_steps):
            tu_step = np.mean(tu_trj.time[time_step].numpy())
            u_step = np.mean(u_trj.time[time_step].numpy())
            reward -= k1 * tu_step + k2 * u_step + xi * 0.1*action[1] # Scaled action penalty
            if time_step % PDE_number_of_steps == 0:
                n_step = np.mean(n_trj.time[time_step].numpy())
                i_step = np.mean(i_trj.time[time_step].numpy())
                info.update({counter: [n_step, tu_step, i_step, u_step]})
                counter += 1
        reward /= PDE_number_of_steps
        # Increment step count
        self.time += action[0]+1 # Counter for the number of steps taken in the current episode.
        n_step = np.mean(self.n.numpy())
        tu_step = np.mean(self.tu.numpy())
        i_step = np.mean(self.i.numpy())
        u_step = np.mean(self.u.numpy())
        info.update({counter: [n_step, tu_step, i_step, u_step]})
        # End episode if max steps reached
        done = self.time >= self.max_time or l1 * tu_step + l2 * u_step < 0.01
        return self.state, reward, done, False, info

    def render(self, mode='human'):
        """
        Simple rendering that prints the current tumor and drug.
        """
        if self.render_mode == 'human':
            plot(stack([self.n, self.tu, self.i, self.u], batch(f"Step: {self.time}")))