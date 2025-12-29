import numpy as np

# --- Environment parameters ---
N_RAYS = 12                # Number of rays for sensors can't be changed
RAY_LENGTH = 1.6           # Length of each ray
NOISE_STD  = 0.01           # Standard deviation of noise added to ray sensors
MAX_STEPS = 600            # Maximum steps per episode
CUTOFF_VALUE = 1.0         # Sensor cutoff distace, when < cutoff, get negative reward
SWITCH_EVERY = 5          # Frequency to switch xml environments
X_MIN,X_MAX,Y_MIN,Y_MAX = -19.5, 19.5,-19.5,19.5 # Agent position boundry
MAX_DISTANCE = float(np.sqrt((X_MAX - X_MIN)**2 + (Y_MAX - Y_MIN)**2))          # Maximum distance in the layout

# --- Action space limits ---
MAX_X = 0.1                # Maximum movement per step in X direction
MAX_Y = 0.1                # Maximum movement per step in Y direction


# --- Reward parameters ---
TIME_PENALTY = -0.05       # Penalty per time step to encourage faster completion
STUCK_PENALTY = -0.5        # Penalty applied if agent is stuck


# --- Goal parameters ---
GOAL_THRESHOLD = 0.4       # Distance threshold for goal to be considered reached


# --- History length for stuck detection ---
POSITION_HISTORY_LEN = 20  # Number of previous positions to track for stuck detection