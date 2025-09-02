# thresholds.py

"""
Common threshold values used across different modules of the pick and place task.
Centralizing these values makes it easier to tune and maintain consistency.
"""

# Table and environment
TABLE_HEIGHT = 0.77  # Height of the table surface


# Cube dimensions
CUBE_HEIGHT = 0.0382  # Height of the cube
CUBE_WIDTH = 0.0286  # Width of the cube
CUBE_LENGTH = 0.0635  # Length of the cube

# Cube starting height (table + half cube height)
CUBE_START_HEIGHT = TABLE_HEIGHT + (CUBE_HEIGHT / 2)

# Cube target placement
PLACEMENT_POS_THRESHOLD = 0.05  # Max distance for cube to be considered at target position

# Gripper thresholds
GRIPPER_OPEN_THRESHOLD = 5.0    # Max position to consider gripper "open"
GRIPPER_CLOSED_THRESHOLD = 25.0  # Min position to consider gripper "closed"
GRIPPER_CLOSING_THRESHOLD = 10.0 # Min position to consider gripper "closing" for rewards

# End effector positioning
POSITION_THRESHOLD = 0.01       # Max position error for success conditions
ORIENTATION_THRESHOLD = 0.9     # Min orientation alignment for success (0-1)
CUBE_HOVER_HEIGHT = 0.3         # Height to maintain above the cube during alignment
PRE_GRASP_HEIGHT = 0.1          # Height just before grasping

# Joint control
VELOCITY_THRESHOLD = 0.05       # Max joint velocity for stability
TORQUE_THRESHOLD = 1.0          # Max joint torque for stability

# Reward settings
CUBE_MAX_HEIGHT = 1.0           # Height for maximum cube lifting reward
DISTANCE_SCALE = 0.1            # Standard deviation for tanh distance rewards