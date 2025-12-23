"""
PADM Assignment 1: Custom Grid World Environment
=================================================

A rich grid world environment built with OpenAI Gymnasium template.
The agent (Scrat from Ice Age) navigates through a maze to reach the goal (Acorn).

Features:
- 7x12 grid maze with obstacles, danger zones, and rewards
- Pygame-based visualization
- 3 actions: Up, Down, Right
- Special "Lover" state that multiplies goal reward by 6x

Author: Meftun Akarsu
Date: December 2025
Course: Planning and Decision Making (PADM)
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
    Custom Grid World Environment - Ice Age Theme

    The agent (Scrat) navigates through a 7x12 grid maze to reach the goal (Acorn).

    Environment Components:
    -----------------------
    - Empty cells: Normal traversable cells
    - Obstacles (O): Ice crystals that block movement
    - Danger states (H): Hell states - large negative reward, episode ends
    - Reward states (R): Mini rewards (+1)
    - Lover state (L): Special bonus (+100), makes goal reward 6x
    - Goal states (G): Episode ends with positive reward

    Observation Space:
    ------------------
    Type: Box(3,)
    - observation[0]: Row position (0 to num_rows-1)
    - observation[1]: Column position (0 to num_cols-1)
    - observation[2]: Has lover flag (0 or 1)

    Action Space:
    -------------
    Type: Discrete(3)
    - 0: Move Up
    - 1: Move Down
    - 2: Move Right

    Reward Structure:
    -----------------
    - Step cost: -1 (living cost)
    - Goal reached: +100 (or +600 if has lover)
    - Danger hit: -100 (or -600 if has lover)
    - Lover found: +100 (first time only)
    - Mini reward: +1 (collectible)
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

        # For compatibility
        self.grid_size = max(num_rows, num_cols)

        # Define the maze layout
        # '.' = empty, 'O' = obstacle, 'H' = hell/danger, 'R' = reward
        # 'A' = agent start, 'G' = goal, 'L' = lover
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
        self.position = None
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
            'background': (200, 220, 255),
            'grid': (150, 180, 220),
            'agent': (255, 165, 0),
            'goal': (0, 255, 0),
            'danger': (255, 0, 0),
            'obstacle': (100, 150, 200),
            'reward': (255, 215, 0),
            'lover': (255, 105, 180),
            'empty': (240, 248, 255),
            'text': (0, 0, 0)
        }

    def _load_images(self):
        """Load images for pygame rendering."""
        if self.images_loaded:
            return

        try:
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
        observation : np.ndarray
            New state
        reward : float
            Reward received
        terminated : bool
            Whether episode ended (goal or danger)
        truncated : bool
            Whether episode was truncated (max steps)
        info : dict
            Additional information
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

        # Apply living cost
        if not hit_danger and not reached_goal and not self.lover_collected:
            reward -= self.living_cost

        # Reset lover_collected flag
        self.lover_collected = False

        # Check for truncation
        truncated = self.step_count >= self.max_steps

        # Update total reward
        self.total_reward += reward

        # Update state
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
            os.environ['SDL_VIDEO_WINDOW_POS'] = "100,100"

            self.window = pygame.display.set_mode(
                (self.num_cols * self.cell_size, self.num_rows * self.cell_size + 50)
            )
            pygame.display.set_caption("PADM Assignment 1 - Ice Age Grid World")
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

                # Draw cell
                if cell == 'O':
                    if 'obstacle' in self.images:
                        self.window.blit(self.images['obstacle'], (x, y))
                    else:
                        pygame.draw.rect(self.window, self.colors['obstacle'], rect)
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

        # Draw lover state
        if self.lover_state is not None and not self.has_lover:
            x = self.lover_state[1] * self.cell_size
            y = self.lover_state[0] * self.cell_size
            if 'lover' in self.images:
                self.window.blit(self.images['lover'], (x, y))
            else:
                center = (x + self.cell_size//2, y + self.cell_size//2)
                pygame.draw.circle(self.window, self.colors['lover'], center, self.cell_size//3)
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
            pygame.draw.circle(self.window, (0, 0, 0), (center[0] - 8, center[1] - 5), 4)
            pygame.draw.circle(self.window, (0, 0, 0), (center[0] + 8, center[1] - 5), 4)
            pygame.draw.arc(self.window, (0, 0, 0), (center[0] - 10, center[1], 20, 10), 3.14, 0, 2)

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
    """Create and return an instance of the environment."""
    return ChidEnv(render_mode=render_mode)


# ============================================================================
# DEMO FUNCTIONS
# ============================================================================

def demo_random_agent():
    """Test the environment with a random agent."""
    print("=" * 60)
    print("RANDOM AGENT DEMO")
    print("=" * 60)

    env = create_env(render_mode="pygame")
    obs, info = env.reset()

    print(f"\nEnvironment Info:")
    print(f"  Grid Size: {env.num_rows}x{env.num_cols}")
    print(f"  Observation Space: {env.observation_space}")
    print(f"  Action Space: {env.action_space}")
    print(f"  Obstacles: {len(env.obstacle_states)}")
    print(f"  Danger States: {len(env.danger_states)}")
    print(f"  Reward States: {len(env.reward_states)}")
    print(f"  Goal States: {len(env.goal_states)}")
    print(f"  Lover State: {env.lover_state}")
    print(f"\nStarting: {obs}")
    print("=" * 60)

    done = False
    total_reward = 0
    action_names = ["UP", "DOWN", "RIGHT"]

    while not done:
        env.render()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(f"Action: {action_names[action]:5} | Pos: [{obs[0]},{obs[1]}] | Reward: {reward:+.1f}")
        done = terminated or truncated
        pygame.time.wait(200)

    print("\n" + "=" * 60)
    if info.get("success"):
        print("SUCCESS!")
    elif info.get("hit_danger"):
        print("FAILED - Hit danger!")
    else:
        print("TIME UP!")
    print(f"Total Reward: {total_reward:.1f}")
    print("=" * 60)

    pygame.time.wait(2000)
    env.close()


def demo_keyboard_control():
    """Control the agent with keyboard."""
    print("=" * 60)
    print("KEYBOARD CONTROL")
    print("Controls: W=Up, S=Down, D=Right, Q=Quit")
    print("=" * 60)

    env = create_env(render_mode="pygame")
    obs, _ = env.reset()

    done = False
    total_reward = 0

    while not done:
        env.render()

        action = None
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key in [pygame.K_w, pygame.K_UP]:
                        action = 0
                        waiting = False
                    elif event.key in [pygame.K_s, pygame.K_DOWN]:
                        action = 1
                        waiting = False
                    elif event.key in [pygame.K_d, pygame.K_RIGHT]:
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
    print(f"\nTotal Reward: {total_reward:.1f}")
    env.close()


if __name__ == "__main__":
    print("Select demo:")
    print("1: Random Agent")
    print("2: Keyboard Control")

    try:
        choice = input("\nChoice (1/2): ").strip() or "1"
        if choice == "2":
            demo_keyboard_control()
        else:
            demo_random_agent()
    except KeyboardInterrupt:
        print("\nExiting...")
