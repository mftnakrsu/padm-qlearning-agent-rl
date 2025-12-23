# PADM Assignment 1: Custom Grid World Environment

**Author:** Meftun Akarsu
**Course:** Planning and Decision Making (PADM)
**Date:** December 2025

---

## Overview

This assignment implements a custom Grid World environment using the OpenAI Gymnasium framework. The environment features an Ice Age theme where the agent (Scrat) navigates through a maze to reach the goal (Acorn).

---

## Environment Description

### Grid Layout (7 x 12)

```
. . O . O . O . O R . G    <- Row 0 (Goal at [0,11])
. . O . O . O . O . H H    <- Row 1 (Danger zones)
. . . H O L . H O . . .    <- Row 2 (Lover at [2,5])
A H . . R . . . . . . .    <- Row 3 (Agent starts at [3,0])
. . O . . . O . . H . .    <- Row 4
. . O . O H O . O . . G    <- Row 5 (Goal at [5,11])
. . O . O . O . O . . .    <- Row 6

Legend:
  A = Agent Start
  G = Goal
  H = Hell (Danger Zone)
  O = Obstacle (Ice Crystal)
  R = Reward
  L = Lover
  . = Empty Cell
```

### Components

| Component | Count | Description |
|-----------|-------|-------------|
| Grid Size | 7 x 12 | 84 total cells |
| Obstacles | 20 | Ice crystals that block movement |
| Danger States | 7 | Hell states - episode ends with penalty |
| Goal States | 2 | Target locations |
| Reward States | 2 | Mini rewards (+1) |
| Lover State | 1 | Special bonus, multiplies goal reward |

---

## API Reference

### Observation Space
- **Type:** `Box(3,)` with dtype `int32`
- **Contents:** `[row, col, has_lover]`
  - `row`: 0 to 6 (agent's row position)
  - `col`: 0 to 11 (agent's column position)
  - `has_lover`: 0 or 1 (whether agent has visited lover)

### Action Space
- **Type:** `Discrete(3)`
- **Actions:**
  - `0`: Move Up
  - `1`: Move Down
  - `2`: Move Right

### Reward Structure

| Event | Reward |
|-------|--------|
| Step (living cost) | -1 |
| Goal reached | +100 |
| Goal with lover | +600 |
| Hit danger | -100 |
| Hit danger with lover | -600 |
| Find lover (first time) | +100 |
| Collect mini reward | +1 |

---

## Required Methods

### `__init__(self, num_rows=7, num_cols=12, cell_size=80, render_mode="pygame")`
Initialize the environment with grid dimensions and rendering settings.

### `reset(self, seed=None, options=None) -> (observation, info)`
Reset environment to initial state. Returns initial observation and info dict.

### `step(self, action) -> (observation, reward, terminated, truncated, info)`
Execute action and return results:
- `observation`: New state `[row, col, has_lover]`
- `reward`: Immediate reward
- `terminated`: True if goal reached or hit danger
- `truncated`: True if max steps reached
- `info`: Additional information dict

### `render(self) -> None`
Render current state using pygame or text mode.

### `close(self) -> None`
Close pygame window and cleanup resources.

---

## Installation

### Requirements
```
gymnasium>=0.29.0
numpy>=1.24.0
pygame>=2.5.0
```

### Install Dependencies
```bash
pip install gymnasium numpy pygame
```

---

## Usage

### Basic Usage
```python
from chid_env import ChidEnv, create_env

# Create environment
env = create_env(render_mode="pygame")

# Reset
obs, info = env.reset()

# Run episode
done = False
while not done:
    env.render()
    action = env.action_space.sample()  # Random action
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

env.close()
```

### Run Demo
```bash
python demo.py
```

### Run Environment Directly
```bash
python chid_env.py
```

---

## File Structure

```
assignment1/
├── chid_env.py     # Main environment class
├── demo.py         # Demo script
├── README.md       # This file
└── images/         # Optional: Custom images for pygame
    ├── Scrat.png
    ├── Acorn.png
    ├── Enemy.png
    ├── IceCrystal.png
    ├── Scratte.png
    └── Background.png
```

---

## Features

1. **OpenAI Gymnasium Compatible**
   - Follows standard Gymnasium API
   - Compatible with RL algorithms

2. **Multiple Render Modes**
   - `pygame`: Graphical visualization
   - `human`: Text-based output
   - `ansi`: String output

3. **Rich Environment**
   - Obstacles, dangers, rewards
   - Special "Lover" mechanic for strategy
   - Multiple goal states

4. **Customizable**
   - Grid size adjustable
   - Cell size for pygame
   - Reward values configurable

---

## Keyboard Controls (in pygame mode)

| Key | Action |
|-----|--------|
| W / UP | Move Up |
| S / DOWN | Move Down |
| D / RIGHT | Move Right |
| Q | Quit |

---

## Example Output

```
Environment Info:
  Grid Size: 7x12
  Observation Space: Box(0, [6, 11, 1], (3,), int32)
  Action Space: Discrete(3)

Starting: [3, 0, 0]
Step 1: RIGHT -> [3,1], Reward: -100 (Hit danger!)
FAILED!
```

---

## Notes for Exam

### Key Concepts

1. **Observation Space**: What the agent can see
   - Position (row, col) and lover status

2. **Action Space**: What the agent can do
   - 3 discrete actions (Up, Down, Right)

3. **Reward Signal**: Guides learning
   - Positive for goals, negative for dangers

4. **Done Flag**: Episode termination
   - `terminated`: Goal or danger
   - `truncated`: Max steps

5. **Info Dictionary**: Extra information
   - Distance to goal, success status, etc.

### Design Decisions

- **3 Actions Only**: Simplified action space (no Left) to make learning easier
- **Lover Mechanic**: Adds strategic depth - visit lover for 6x reward
- **Multiple Goals**: Agent can reach either goal
- **State Includes has_lover**: Enables different policies before/after lover
