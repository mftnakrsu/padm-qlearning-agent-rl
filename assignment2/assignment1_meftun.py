"""
Custom Grid World Environment for PADM Assignment 1
====================================================

This environment is a rich grid world with obstacles, danger zones,
reward states, and pygame-based visualization.

Inspired by Ice Age theme - Agent navigates through a maze to reach the goal.

Author: Meftun
Date: December 2025
Course: Planning and Decision Making
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import os
from pathlib import Path

# Project directories
PROJECT_DIR = Path(__file__).parent
IMAGE_DIR = PROJECT_DIR / 'images'


class ChidEnv(gym.Env):
    """
    Custom Grid World Environment with Obstacles and Dangers

    The agent navigates through a 7x12 grid maze to reach the goal.
    The environment contains:
    - Obstacles (ice crystals) that block movement
    - Danger states (enemies) that give negative rewards
    - Reward states (bonuses) that give positive rewards
    - A special "lover" state that doubles the goal reward

    Observation Space:
        Type: Box(2)
        Agent's position on the grid in [row, col] format

    Action Space:
        Type: Discrete(3)
        0: Up
        1: Down
        2: Right

    Reward Structure:
        -1.0: Living cost for each step
        +100.0: Reaching the goal
        +600.0: Reaching the goal after visiting lover
        -100.0: Hitting a danger state
        -600.0: Hitting a danger state after visiting lover
        +1.0: Collecting a reward state
        +100.0: First time visiting lover
    """

    metadata = {"render_modes": ["human", "pygame", "ansi"], "render_fps": 10}

    def __init__(self, num_rows=7, num_cols=12, cell_size=80, render_mode="pygame"):
        """
        Initialize the environment.

        Parameters:
        -----------
        num_rows : int, optional (default=7)
            Number of rows in the grid

        num_cols : int, optional (default=12)
            Number of columns in the grid

        cell_size : int, optional (default=80)
            Size of each cell in pixels for pygame rendering

        render_mode : str, optional (default="pygame")
            Render mode: "pygame", "human" (text), or "ansi"
        """
        super().__init__()

        self.num_rows = num_rows
        self.num_cols = num_cols
        self.cell_size = cell_size
        self.render_mode = render_mode

        # For compatibility with Q-learning code
        self.grid_size = max(num_rows, num_cols)

        # Define the maze layout
        # '.' = empty, 'O' = obstacle, 'H' = hell/danger, 'R' = reward
        # 'A' = agent start, 'G' = goal, 'L' = lover (special reward)
        self.maze = np.array([
            ['.', '.', 'O', '.', 'O', '.', 'O', '.', 'O', 'R', '.', 'G'],
            ['.', '.', 'O', '.', 'O', '.', 'O', '.', 'O', '.', 'H', 'H'],
            ['.', '.', '.', 'H', 'O', 'L', '.', 'H', 'O', '.', '.', '.'],
            ['A', 'H', '.', '.', 'R', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', 'O', '.', '.', '.', 'O', '.', '.', 'H', '.', '.'],
            ['.', '.', 'O', '.', 'O', 'H', 'O', '.', 'O', '.', '.', 'G'],
            ['.', '.', 'O', '.', 'O', '.', 'O', '.', 'O', '.', '.', '.'],
        ])

        # Parse maze to find special positions
        self.agent_start = None
        self.goal_states = []
        self.danger_states = []
        self.obstacle_states = []
        self.reward_states = []
        self.lover_state = None

        for i in range(self.num_rows):
            for j in range(self.num_cols):
                cell = self.maze[i, j]
                if cell == 'A':
                    self.agent_start = np.array([i, j], dtype=np.int32)
                elif cell == 'G':
                    self.goal_states.append(np.array([i, j], dtype=np.int32))
                elif cell == 'H':
                    self.danger_states.append(np.array([i, j], dtype=np.int32))
                elif cell == 'O':
                    self.obstacle_states.append(np.array([i, j], dtype=np.int32))
                elif cell == 'R':
                    self.reward_states.append(np.array([i, j], dtype=np.int32))
                elif cell == 'L':
                    self.lover_state = np.array([i, j], dtype=np.int32)

        # Primary goal (for compatibility)
        self.goal = self.goal_states[0] if self.goal_states else np.array([0, 11], dtype=np.int32)

        # Reward values
        self.goal_reward = 100
        self.danger_penalty = 100
        self.living_cost = 1
        self.mini_reward = 1
        self.lover_multiplier = 6

        # State variables
        self.state = None
        self.has_lover = False
        self.lover_collected = False
        self.collected_rewards = set()
        self.step_count = 0
        self.max_steps = 200
        self.total_reward = 0

        # Action space: 3-way movement (Up, Down, Right)
        self.action_space = spaces.Discrete(3)

        # Observation space: (row, col, has_lover)
        # has_lover: 0 or 1
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0]),
            high=np.array([self.num_rows - 1, self.num_cols - 1, 1]),
            shape=(3,),
            dtype=np.int32
        )

        # Pygame setup
        self.window = None
        self.clock = None
        self.images_loaded = False
        self.images = {}

        # Colors for fallback rendering
        self.colors = {
            'background': (200, 220, 255),  # Light blue (ice)
            'grid': (150, 180, 220),
            'agent': (255, 165, 0),  # Orange
            'goal': (0, 255, 0),  # Green
            'danger': (255, 0, 0),  # Red
            'obstacle': (100, 150, 200),  # Blue-gray (ice)
            'reward': (255, 215, 0),  # Gold
            'lover': (255, 105, 180),  # Pink
            'empty': (240, 248, 255),  # Alice blue
            'text': (0, 0, 0)
        }

    def _load_images(self):
        """Load images for pygame rendering."""
        if self.images_loaded:
            return

        try:
            # Try to load images if they exist
            if IMAGE_DIR.exists():
                image_files = {
                    'agent': ['Scrat.png', 'agent.png'],
                    'goal': ['ScratsAcorn.webp', 'Acorn.png', 'goal.png'],
                    'danger': ['Enemy.png', 'Dinasour.png', 'danger.png'],
                    'obstacle': ['IceCrystal.png', 'obstacle.png'],
                    'reward': ['Reward.png', 'MiniAcorn.png', 'reward.png'],
                    'lover': ['Scratte.png', 'lover.png'],
                    'background': ['Background.png', 'background.png']
                }

                for key, filenames in image_files.items():
                    for filename in filenames:
                        img_path = IMAGE_DIR / filename
                        if img_path.exists():
                            try:
                                img = pygame.image.load(str(img_path))
                                if key == 'background':
                                    img = pygame.transform.scale(img,
                                        (self.num_cols * self.cell_size, self.num_rows * self.cell_size))
                                else:
                                    img = pygame.transform.scale(img, (self.cell_size, self.cell_size))
                                self.images[key] = img
                                break
                            except:
                                pass
        except Exception as e:
            print(f"Warning: Could not load images: {e}")

        self.images_loaded = True

    def reset(self, seed=None, options=None):
        """
        Reset the environment to initial state.

        Returns:
        --------
        observation : np.ndarray
            Initial state (row, col, has_lover)
        info : dict
            Additional information
        """
        super().reset(seed=seed)

        # Reset agent position
        self.position = self.agent_start.copy()

        # Reset state variables
        self.has_lover = False
        self.lover_collected = False
        self.collected_rewards = set()
        self.step_count = 0
        self.total_reward = 0

        # State includes position and has_lover flag
        self.state = np.array([self.position[0], self.position[1], 0], dtype=np.int32)

        info = {
            "distance_to_goal": float(np.linalg.norm(self.goal - self.position)),
            "danger_states": [d.tolist() for d in self.danger_states],
            "obstacle_states": [o.tolist() for o in self.obstacle_states]
        }

        return self.state.copy(), info

    def _is_valid_position(self, pos):
        """Check if a position is valid (within bounds and not an obstacle)."""
        row, col = pos

        # Check bounds
        if row < 0 or row >= self.num_rows or col < 0 or col >= self.num_cols:
            return False

        # Check obstacles
        for obs in self.obstacle_states:
            if np.array_equal(pos, obs):
                return False

        return True

    def step(self, action):
        """
        Take an action in the environment.

        Parameters:
        -----------
        action : int
            0: Up, 1: Down, 2: Right

        Returns:
        --------
        observation, reward, terminated, truncated, info
        """
        self.step_count += 1

        # Calculate new position
        new_pos = self.position.copy()

        if action == 0:  # Up
            new_pos[0] -= 1
        elif action == 1:  # Down
            new_pos[0] += 1
        elif action == 2:  # Right
            new_pos[1] += 1

        # Check if new position is valid
        if self._is_valid_position(new_pos):
            self.position = new_pos

        # Initialize reward and flags
        reward = 0
        terminated = False
        hit_danger = False
        reached_goal = False

        # Check for lover state
        if self.lover_state is not None and np.array_equal(self.position, self.lover_state):
            if not self.has_lover:
                self.has_lover = True
                self.lover_collected = True
                reward += self.goal_reward  # Bonus for finding lover

        # Check for danger states
        for danger in self.danger_states:
            if np.array_equal(self.position, danger):
                hit_danger = True
                if self.has_lover:
                    reward -= self.lover_multiplier * self.danger_penalty
                else:
                    reward -= self.danger_penalty
                terminated = True
                break

        # Check for reward states (mini rewards)
        if not hit_danger:
            for i, rew_state in enumerate(self.reward_states):
                if np.array_equal(self.position, rew_state) and i not in self.collected_rewards:
                    reward += self.mini_reward
                    self.collected_rewards.add(i)
                    break

        # Check for goal states
        if not hit_danger:
            for goal in self.goal_states:
                if np.array_equal(self.position, goal):
                    reached_goal = True
                    if self.has_lover:
                        reward += self.lover_multiplier * self.goal_reward
                    else:
                        reward += self.goal_reward
                    terminated = True
                    break

        # Apply living cost if nothing special happened
        if not hit_danger and not reached_goal and not self.lover_collected:
            reward -= self.living_cost

        # Reset lover_collected flag after first step
        self.lover_collected = False

        # Check for truncation
        truncated = self.step_count >= self.max_steps

        # Update total reward
        self.total_reward += reward

        # Update state with position and has_lover flag
        self.state = np.array([self.position[0], self.position[1], int(self.has_lover)], dtype=np.int32)

        # Info
        info = {
            "distance_to_goal": float(np.linalg.norm(self.goal - self.position)),
            "step_count": self.step_count,
            "hit_danger": hit_danger,
            "reached_goal": reached_goal,
            "has_lover": self.has_lover,
            "total_reward": self.total_reward,
            "success": reached_goal
        }

        return self.state.copy(), reward, terminated, truncated, info

    def render(self):
        """Render the environment."""
        if self.render_mode == "ansi" or self.render_mode == "human":
            return self._render_text()
        elif self.render_mode == "pygame":
            return self._render_pygame()
        return None

    def _render_text(self):
        """Render environment as text."""
        grid = []
        for i in range(self.num_rows):
            row = []
            for j in range(self.num_cols):
                if np.array_equal([i, j], self.position):
                    row.append('A')
                elif self.maze[i, j] == 'O':
                    row.append('#')
                elif self.maze[i, j] == 'H':
                    row.append('X')
                elif self.maze[i, j] == 'G':
                    row.append('G')
                elif self.maze[i, j] == 'R':
                    row.append('R')
                elif self.maze[i, j] == 'L':
                    row.append('L')
                else:
                    row.append('.')
            grid.append(' '.join(row))

        out = '\n'.join(grid)
        out += f"\n\nStep: {self.step_count} | Reward: {self.total_reward:.1f} | Has Lover: {self.has_lover}"

        if self.render_mode == "human":
            print(out)
            print("-" * 40)
            return None
        return out

    def _render_pygame(self):
        """Render environment using pygame."""
        if self.window is None:
            pygame.init()
            pygame.display.init()

            # Set window position
            os.environ['SDL_VIDEO_WINDOW_POS'] = "100,100"

            self.window = pygame.display.set_mode(
                (self.num_cols * self.cell_size, self.num_rows * self.cell_size + 50)
            )
            pygame.display.set_caption("Grid World Environment - Assignment 1")
            self.clock = pygame.time.Clock()
            self._load_images()

        # Clear screen
        self.window.fill(self.colors['background'])

        # Draw background image if available
        if 'background' in self.images:
            self.window.blit(self.images['background'], (0, 0))

        # Draw grid
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                x = j * self.cell_size
                y = i * self.cell_size
                rect = pygame.Rect(x, y, self.cell_size, self.cell_size)

                cell = self.maze[i, j]

                # Draw cell background
                if cell == 'O':
                    if 'obstacle' in self.images:
                        self.window.blit(self.images['obstacle'], (x, y))
                    else:
                        pygame.draw.rect(self.window, self.colors['obstacle'], rect)
                        # Draw ice crystal pattern
                        pygame.draw.polygon(self.window, (150, 200, 255), [
                            (x + self.cell_size//2, y + 5),
                            (x + self.cell_size - 5, y + self.cell_size//2),
                            (x + self.cell_size//2, y + self.cell_size - 5),
                            (x + 5, y + self.cell_size//2)
                        ])
                elif cell != '.':
                    pygame.draw.rect(self.window, self.colors['empty'], rect)

                # Draw grid lines
                pygame.draw.rect(self.window, self.colors['grid'], rect, 1)

        # Draw reward states
        for i, rew_state in enumerate(self.reward_states):
            if i not in self.collected_rewards:
                x = rew_state[1] * self.cell_size
                y = rew_state[0] * self.cell_size
                if 'reward' in self.images:
                    self.window.blit(self.images['reward'], (x, y))
                else:
                    center = (x + self.cell_size//2, y + self.cell_size//2)
                    pygame.draw.circle(self.window, self.colors['reward'], center, self.cell_size//3)
                    pygame.draw.circle(self.window, (200, 150, 0), center, self.cell_size//3, 2)

        # Draw lover state
        if self.lover_state is not None and not self.has_lover:
            x = self.lover_state[1] * self.cell_size
            y = self.lover_state[0] * self.cell_size
            if 'lover' in self.images:
                self.window.blit(self.images['lover'], (x, y))
            else:
                center = (x + self.cell_size//2, y + self.cell_size//2)
                pygame.draw.circle(self.window, self.colors['lover'], center, self.cell_size//3)
                # Draw heart shape
                font = pygame.font.Font(None, self.cell_size//2)
                heart = font.render("L", True, (255, 255, 255))
                self.window.blit(heart, (x + self.cell_size//3, y + self.cell_size//3))

        # Draw danger states
        for danger in self.danger_states:
            x = danger[1] * self.cell_size
            y = danger[0] * self.cell_size
            if 'danger' in self.images:
                self.window.blit(self.images['danger'], (x, y))
            else:
                rect = pygame.Rect(x + 5, y + 5, self.cell_size - 10, self.cell_size - 10)
                pygame.draw.rect(self.window, self.colors['danger'], rect)
                # Draw X
                pygame.draw.line(self.window, (255, 255, 255),
                               (x + 15, y + 15), (x + self.cell_size - 15, y + self.cell_size - 15), 3)
                pygame.draw.line(self.window, (255, 255, 255),
                               (x + self.cell_size - 15, y + 15), (x + 15, y + self.cell_size - 15), 3)

        # Draw goal states
        for goal in self.goal_states:
            x = goal[1] * self.cell_size
            y = goal[0] * self.cell_size
            if 'goal' in self.images:
                self.window.blit(self.images['goal'], (x, y))
            else:
                center = (x + self.cell_size//2, y + self.cell_size//2)
                pygame.draw.circle(self.window, self.colors['goal'], center, self.cell_size//3)
                font = pygame.font.Font(None, self.cell_size//2)
                text = font.render("G", True, (255, 255, 255))
                self.window.blit(text, (x + self.cell_size//3, y + self.cell_size//3))

        # Draw agent
        x = self.position[1] * self.cell_size
        y = self.position[0] * self.cell_size
        if 'agent' in self.images:
            self.window.blit(self.images['agent'], (x, y))
        else:
            center = (x + self.cell_size//2, y + self.cell_size//2)
            pygame.draw.circle(self.window, self.colors['agent'], center, self.cell_size//3)
            # Draw face
            pygame.draw.circle(self.window, (0, 0, 0),
                             (center[0] - 8, center[1] - 5), 4)
            pygame.draw.circle(self.window, (0, 0, 0),
                             (center[0] + 8, center[1] - 5), 4)
            pygame.draw.arc(self.window, (0, 0, 0),
                          (center[0] - 10, center[1], 20, 10), 3.14, 0, 2)

        # Draw status bar
        status_y = self.num_rows * self.cell_size + 5
        font = pygame.font.Font(None, 30)

        status_text = f"Step: {self.step_count}  |  Reward: {self.total_reward:.1f}  |  Lover: {'Yes' if self.has_lover else 'No'}"
        text_surface = font.render(status_text, True, self.colors['text'])
        self.window.blit(text_surface, (10, status_y))

        # Update display
        pygame.display.flip()

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()

        if self.clock:
            self.clock.tick(self.metadata["render_fps"])

        return None

    def close(self):
        """Close the environment and pygame window."""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None


def create_env(render_mode="pygame"):
    """
    Create and return an instance of the environment.

    Parameters:
    -----------
    render_mode : str, optional (default="pygame")
        Render mode for the environment

    Returns:
    --------
    ChidEnv
        Environment instance
    """
    return ChidEnv(render_mode=render_mode)


# ============================================================================
# DEMO / TEST CODE
# ============================================================================

def demo_random_agent():
    """Test the environment with a random agent."""
    print("=" * 60)
    print("RANDOM AGENT DEMO - Enhanced Grid World")
    print("=" * 60)

    env = create_env(render_mode="pygame")
    obs, info = env.reset()

    print(f"\nEnvironment Info:")
    print(f"  Grid Size: {env.num_rows}x{env.num_cols}")
    print(f"  Obstacles: {len(env.obstacle_states)}")
    print(f"  Danger States: {len(env.danger_states)}")
    print(f"  Reward States: {len(env.reward_states)}")
    print(f"  Goal States: {len(env.goal_states)}")
    print(f"  Lover State: {env.lover_state}")
    print(f"\nStarting position: {obs}")
    print("\nActions: 0=Up, 1=Down, 2=Right")
    print("=" * 60)

    done = False
    total_reward = 0
    action_names = ["UP", "DOWN", "RIGHT"]

    while not done:
        env.render()

        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        print(f"Action: {action_names[action]:5} | Pos: {obs} | Reward: {reward:+.1f} | Total: {total_reward:.1f}")

        done = terminated or truncated

        pygame.time.wait(200)  # Slow down for visualization

    print("\n" + "=" * 60)
    if info.get("reached_goal"):
        print("SUCCESS! Agent reached the goal!")
    elif info.get("hit_danger"):
        print("FAILED! Agent hit a danger zone!")
    else:
        print("TIME UP! Maximum steps reached.")
    print(f"Final Reward: {total_reward:.1f}")
    print(f"Steps: {info['step_count']}")
    print("=" * 60)

    pygame.time.wait(2000)
    env.close()


def demo_manual_control():
    """Test with manual keyboard control."""
    print("=" * 60)
    print("MANUAL CONTROL - Enhanced Grid World")
    print("=" * 60)
    print("Controls: W=Up, S=Down, D=Right, Q=Quit")
    print("=" * 60)

    env = create_env(render_mode="pygame")
    obs, info = env.reset()

    done = False
    total_reward = 0

    while not done:
        env.render()

        # Get keyboard input
        action = None
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_w or event.key == pygame.K_UP:
                        action = 0
                        waiting = False
                    elif event.key == pygame.K_s or event.key == pygame.K_DOWN:
                        action = 1
                        waiting = False
                    elif event.key == pygame.K_d or event.key == pygame.K_RIGHT:
                        action = 2
                        waiting = False
                    elif event.key == pygame.K_q:
                        env.close()
                        return

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

    env.render()
    pygame.time.wait(2000)

    print(f"\nFinal Reward: {total_reward:.1f}")
    if info.get("reached_goal"):
        print("SUCCESS!")
    elif info.get("hit_danger"):
        print("FAILED!")

    env.close()


if __name__ == "__main__":
    print("Select demo mode:")
    print("1: Random Agent")
    print("2: Manual Control (Keyboard)")

    try:
        choice = input("\nChoice (1/2): ").strip() or "1"

        if choice == "1":
            demo_random_agent()
        elif choice == "2":
            demo_manual_control()
        else:
            demo_random_agent()
    except KeyboardInterrupt:
        print("\nExiting...")
