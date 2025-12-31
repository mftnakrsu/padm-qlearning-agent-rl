# PADM Final Exam Preparation Guide
## Complete Study Material for Oral Exam

**Exam Format:**
- 20 minutes total: 10 minutes presentation + 10 minutes Q&A
- Code will be reviewed during Q&A, not in presentation
- Questions cover: your code, theory, hyperparameters and training behavior

---

## 📚 PART 1: CORE CONCEPTS (Theory Questions)

### 1. What is a Policy? (In Your Own Words)

**Answer:**
"A policy is a strategy or rule that tells the agent what action to take in each state. It's like a decision-making function: given the current situation (state), the policy determines which action the agent should perform. In Q-learning, we learn the optimal policy by finding the action with the highest Q-value in each state."

**Key Points:**
- Policy π(s) → a: maps states to actions
- Optimal policy: maximizes expected cumulative reward
- In Q-learning: π*(s) = argmax_a Q(s,a) (greedy policy)
- In epsilon-greedy: mix of exploration (random) and exploitation (greedy)

---

### 2. Q-Learning Algorithm

**What is Q-Learning?**
Q-learning is a model-free, off-policy reinforcement learning algorithm that learns the optimal action-value function Q(s,a).

**Key Formula:**
```
Q(s,a) = Q(s,a) + α * [r + γ * max Q(s',a') - Q(s,a)]
```

**Components:**
- **Q(s,a)**: Expected cumulative reward from state s, taking action a
- **α (alpha)**: Learning rate - how much we update Q-values
- **r**: Immediate reward
- **γ (gamma)**: Discount factor - importance of future rewards
- **max Q(s',a')**: Best Q-value in next state

**Why it works:**
- Updates Q-values using Bellman equation
- Off-policy: learns optimal policy while following exploration policy
- Model-free: doesn't need environment dynamics

---

### 3. DQN (Deep Q-Network)

**What is DQN?**
DQN uses a neural network to approximate Q-values instead of a Q-table. This allows handling continuous or large state spaces.

**Key Components:**

1. **Neural Network**: Approximates Q(s,a) for all actions
   - Input: state (2D: x, y coordinates)
   - Output: Q-values for each action (4 actions: up, down, left, right)

2. **Experience Replay Buffer**:
   - Stores (s, a, r, s', done) tuples
   - Samples random batches for training
   - **Why?** Breaks correlation between consecutive experiences, stabilizes training

3. **Target Network**:
   - Separate network for computing TD targets
   - Updated less frequently (every N steps)
   - **Why?** Prevents moving target problem, stabilizes learning

4. **Loss Function**:
   ```
   Loss = MSE(Q(s,a) - (r + γ * max Q_target(s',a')))
   ```
   - TD target: r + γ * max Q_target(s',a')
   - Current Q-value: Q(s,a)
   - Minimize difference (TD error)

---

### 4. Epsilon-Greedy Exploration

**What is it?**
Strategy that balances exploration (trying new actions) and exploitation (using learned knowledge).

**How it works:**
- With probability ε (epsilon): choose random action (exploration)
- With probability 1-ε: choose best action (exploitation)

**Epsilon Decay:**
- Start with ε = 1.0 (100% exploration)
- Gradually decrease to ε_min = 0.1 (10% exploration)
- **Why decay?** More exploration early, more exploitation later

**Decay Strategies:**
1. **Linear/Multiplicative**: ε = max(ε_min, ε * decay_rate)
2. **Reverse Sigmoid**: Smooth S-curve transition
   - Formula: ε = ε_min + (ε_initial - ε_min) / (1 + exp(k*(t - t0)))
   - Smoother than linear decay

---

### 5. Bellman Equation

**What is it?**
Fundamental equation in RL that expresses the relationship between Q-values of current and next states.

**Formula:**
```
Q*(s,a) = E[r + γ * max Q*(s',a')]
```

**Meaning:**
- Optimal Q-value = immediate reward + discounted future reward
- Recursive: Q-value depends on future Q-values
- Q-learning uses this to update Q-table/network

---

### 6. State Space vs Action Space

**State Space (Observation Space):**
- **Assignment 1 & 2**: Discrete, 3D: [row, col, has_lover]
  - row: 0-6 (7 rows)
  - col: 0-11 (12 columns)
  - has_lover: 0 or 1 (boolean)
  - Total: 7 × 12 × 2 = 168 states

- **Assignment 3**: Continuous, 2D: [x, y]
  - x, y ∈ [0, 1] (normalized coordinates)
  - Infinite states (continuous)

**Action Space:**
- **Assignment 1 & 2**: Discrete(3) - Up, Down, Right
- **Assignment 3**: Discrete(4) - Up, Down, Left, Right

---

### 7. Reward Structure

**Assignment 1 & 2:**
- Step cost: -1 (living cost)
- Goal: +100 (or +600 with lover)
- Danger: -100 (or -600 with lover)
- Lover: +100 (first time)
- Mini reward: +1

**Assignment 3:**
- Step penalty: -0.01 (encourages efficiency)
- Wall collision: -1.0
- Danger zone: -100.0 (episode ends)
- Goal: +100.0 (episode ends)
- Distance shaping: +0.1 * (1 - distance_to_goal)

**Why these rewards?**
- Negative for bad actions (walls, danger)
- Large positive for goal
- Shaping helps guide agent toward goal

---

## 📝 PART 2: CODE EXPLANATIONS

### Assignment 1: Environment (`chid_env.py`)

#### Key Methods:

**1. `reset(seed=None, options=None)`:**
```python
def reset(self, seed=None, options=None):
    # Reset agent to start position [3, 0]
    self.position = self.agent_start.copy()
    # Reset state variables
    self.has_lover = False
    self.state = [row, col, 0]  # has_lover = 0 initially
    return self.state, info
```
- **Purpose**: Initialize environment for new episode
- **Returns**: Initial state [row, col, has_lover] and info dict

**2. `step(action)`:**
```python
def step(self, action):
    # Calculate new position
    new_pos = position + action_delta
    # Check if valid (not obstacle, within bounds)
    if self._is_valid_position(new_pos):
        self.position = new_pos
    
    # Calculate reward based on:
    # - Lover state (if visited)
    # - Danger states (if hit)
    # - Goal states (if reached)
    # - Mini rewards (if collected)
    # - Living cost (-1 per step)
    
    # Update state: [row, col, has_lover]
    self.state = [position[0], position[1], int(self.has_lover)]
    
    return state, reward, terminated, truncated, info
```
- **Purpose**: Execute action, update state, calculate reward
- **Returns**: new_state, reward, done flags, info

**3. `_is_valid_position(pos)`:**
```python
def _is_valid_position(self, pos):
    # Check bounds
    if row < 0 or row >= num_rows or col < 0 or col >= num_cols:
        return False
    # Check obstacles
    for obs in self.obstacle_states:
        if np.array_equal(pos, obs):
            return False
    return True
```
- **Purpose**: Validate if position is legal (within bounds, not obstacle)

**Why `has_lover` in state?**
- Without it: agent might loop trying to re-visit lover
- With it: state distinguishes "has lover" vs "no lover"
- Allows learning different policies for each case

---

### Assignment 2: Q-Learning (`assignment2_qlearning.py`)

#### Key Methods:

**1. `update_q_value(state, action, reward, next_state, done)`:**
```python
def update_q_value(self, state, action, reward, next_state, done):
    state_idx = (row, col, has_lover)  # 3D index
    next_state_idx = (next_row, next_col, next_has_lover)
    
    current_q = self.q_table[state_idx][action]
    
    if done:
        target_q = reward  # Terminal: no future rewards
    else:
        max_next_q = np.max(self.q_table[next_state_idx])  # Best next action
        target_q = reward + gamma * max_next_q
    
    # Q-learning update
    self.q_table[state_idx][action] = current_q + alpha * (target_q - current_q)
```
- **Purpose**: Update Q-value using Bellman equation
- **Formula**: Q(s,a) = Q(s,a) + α[r + γ*max Q(s',a') - Q(s,a)]

**2. `choose_action(state, training=True)`:**
```python
def choose_action(self, state, training=True):
    if training and np.random.rand() < self.epsilon:
        return self.env.action_space.sample()  # Exploration: random
    else:
        return np.argmax(self.q_table[state_idx])  # Exploitation: best action
```
- **Purpose**: Epsilon-greedy action selection
- **Training=True**: uses epsilon (exploration)
- **Training=False**: greedy (exploitation only)

**3. `train(num_episodes)`:**
```python
def train(self, num_episodes):
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        
        while not done:
            action = self.choose_action(state, training=True)
            next_state, reward, done, info = env.step(action)
            
            # Update Q-value
            self.update_q_value(state, action, reward, next_state, done)
            
            state = next_state
        
        # Decay epsilon
        self.epsilon = reverse_sigmoid_decay(...)
```
- **Purpose**: Train agent over multiple episodes
- **Process**: Reset → Act → Update Q → Decay epsilon → Repeat

**4. Q-Table Structure:**
```python
# Shape: (num_rows, num_cols, has_lover, num_actions)
# Example: (7, 12, 2, 3)
q_table[3, 0, 0, 2]  # State [3,0] without lover, action Right
q_table[3, 0, 1, 2]  # State [3,0] with lover, action Right
```
- **4D array**: rows × cols × has_lover × actions
- Stores Q-value for each (state, action) pair

---

### Assignment 3: DQN (`main.py`, `DQN_model.py`, `utils.py`)

#### Key Components:

**1. DQN Network (`DQN_model.py`):**
```python
class DQN(nn.Module):
    def __init__(self, state_dim=2, action_dim=4, hidden_dims=[128, 128, 64]):
        # Input: state [x, y] (2D continuous)
        # Hidden: [128, 128, 64] with ReLU
        # Output: Q-values for 4 actions
    
    def forward(self, state):
        # Pass through network
        q_values = self.network(state)  # Shape: (batch, 4)
        return q_values
```
- **Purpose**: Approximate Q(s,a) for continuous states
- **Input**: 2D state [x, y]
- **Output**: Q-values for 4 actions

**2. Experience Replay (`utils.py`):**
```python
class ReplayBuffer:
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        # Convert to tensors
        return states, actions, rewards, next_states, dones
```
- **Purpose**: Store experiences, sample random batches
- **Why?** Breaks correlation, stabilizes training

**3. Training Step (`main.py`):**
```python
def train_step(self):
    # Sample batch from replay buffer
    states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
    
    # Current Q-values
    current_q = self.q_network(states).gather(1, actions)
    
    # TD targets using target network
    with torch.no_grad():
        next_q = self.target_network(next_states)
        max_next_q = torch.max(next_q, dim=1)[0]
        td_targets = rewards + gamma * max_next_q * ~dones
    
    # Compute loss
    loss = F.mse_loss(current_q, td_targets)
    
    # Backpropagation
    loss.backward()
    optimizer.step()
    
    # Update target network periodically
    if step_count % target_update_freq == 0:
        target_network.load_state_dict(q_network.state_dict())
```
- **Purpose**: Train network on batch of experiences
- **Process**: Sample → Compute Q-values → Compute TD targets → Update network

**4. Training Loop:**
```python
for episode in range(num_episodes):
    state, _ = env.reset()
    done = False
    
    while not done:
        # Select action (epsilon-greedy)
        action = agent.select_action(state, training=True)
        
        # Take step
        next_state, reward, done, _ = env.step(action)
        
        # Store in replay buffer
        agent.replay_buffer.push(state, action, reward, next_state, done)
        
        # Train (after learning_starts steps)
        if agent.step_count >= learning_starts:
            loss = agent.train_step()
        
        state = next_state
    
    # Decay epsilon
    agent.decay_epsilon()
    
    # Check for 100 consecutive successes (after epsilon reaches 0.1)
    if epsilon_reached_min and reached_goal:
        consecutive_successes += 1
```

---

## ⚙️ PART 3: HYPERPARAMETERS & TRAINING BEHAVIOR

### Q-Learning Hyperparameters

#### 1. Learning Rate (α / alpha)

**What it is:**
- Controls how much Q-values are updated each step
- Range: 0.0 to 1.0

**Effects:**
- **Too high (e.g., 0.5)**: 
  - Q-values change too quickly
  - May overshoot optimal values
  - Training unstable, oscillates
- **Too low (e.g., 0.01)**:
  - Q-values change very slowly
  - Takes many episodes to learn
  - May get stuck in suboptimal policy
- **Good value (e.g., 0.1)**:
  - Balanced learning speed
  - Stable convergence
  - **Your value: 0.1** ✓

**How to choose:**
- Start with 0.1, adjust based on training curves
- If rewards oscillate: decrease
- If learning too slow: increase slightly

---

#### 2. Discount Factor (γ / gamma)

**What it is:**
- Determines importance of future rewards vs immediate rewards
- Range: 0.0 to 1.0

**Effects:**
- **γ = 0.0**: Only cares about immediate reward (myopic)
- **γ = 0.9**: Values future rewards moderately
- **γ = 0.99**: Highly values future rewards (long-term planning)
- **γ = 1.0**: All future rewards equally important (infinite horizon)

**Your value: 0.99** ✓
- Good for long-term planning
- Agent considers future consequences

**How it affects training:**
- Higher γ: Agent learns to plan ahead, may take longer to converge
- Lower γ: Agent focuses on immediate rewards, faster convergence but may miss optimal long-term strategy

---

#### 3. Epsilon (ε) - Exploration Rate

**What it is:**
- Probability of choosing random action (exploration)
- Starts at 1.0 (100% exploration), decays to ε_min

**Effects:**
- **High epsilon (1.0)**: 
  - Lots of exploration
  - Discovers new states/actions
  - But wastes time on bad actions
- **Low epsilon (0.1)**:
  - Mostly exploitation
  - Uses learned knowledge
  - But may miss better strategies

**Epsilon Decay:**
- **Linear/Multiplicative**: ε = max(ε_min, ε * 0.995)
  - Simple, gradual decrease
- **Reverse Sigmoid**: Smooth S-curve
  - Smoother transition
  - **Your choice: Reverse Sigmoid** ✓

**Why decay?**
- Early: Need exploration to discover environment
- Later: Need exploitation to use learned knowledge
- Balance: Smooth transition between phases

---

#### 4. Epsilon Min

**What it is:**
- Minimum exploration rate
- **Assignment 2**: 0.05 (5% exploration)
- **Assignment 3**: 0.1 (10% exploration, FIXED by requirement)

**Why keep some exploration?**
- Prevents getting stuck in suboptimal policy
- Allows adapting to environment changes
- Small random actions can discover better paths

---

### DQN Hyperparameters

#### 1. Learning Rate

**Your value: 0.001** ✓
- Standard for neural networks (Adam optimizer)
- Lower than Q-learning because network updates are more complex

**Effects:**
- Similar to Q-learning but for neural network weights
- Too high: unstable training, loss oscillates
- Too low: slow convergence

---

#### 2. Batch Size

**What it is:**
- Number of experiences sampled from replay buffer for each training step

**Your value: 64** ✓
- Standard value, good balance

**Effects:**
- **Small (e.g., 16)**: 
  - More frequent updates
  - Less stable (high variance)
  - Faster per step but needs more steps
- **Large (e.g., 256)**:
  - More stable (low variance)
  - Slower per step
  - Better gradient estimates
- **Medium (64)**: Good balance ✓

---

#### 3. Replay Buffer Size

**What it is:**
- Maximum number of experiences stored

**Your value: 10,000** ✓

**Effects:**
- **Too small**: Not enough diversity, may forget old experiences
- **Too large**: Wastes memory, old experiences may be irrelevant
- **10,000**: Good for this environment size

---

#### 4. Target Network Update Frequency

**What it is:**
- How often target network is updated (copied from main network)

**Your value: Every 100 steps** ✓

**Effects:**
- **Too frequent (e.g., every 10 steps)**:
  - Target network changes too often
  - Unstable training (moving target problem)
- **Too infrequent (e.g., every 1000 steps)**:
  - Target network becomes outdated
  - Slower learning
- **Every 100 steps**: Good balance ✓

**Why target network?**
- Without it: Q(s,a) and Q(s',a') both change → unstable
- With it: Q(s',a') from fixed target → stable training

---

#### 5. Learning Starts

**What it is:**
- Number of steps before starting training

**Your value: 500** ✓

**Effects:**
- **Too small**: Not enough experiences in buffer
- **Too large**: Wastes time collecting data
- **500**: Good for buffer to have diverse experiences

---

## 🎯 PART 4: POTENTIAL EXAM QUESTIONS & ANSWERS

### Theory Questions

**Q1: "Describe what a policy is in your own words."**

**Answer:**
"A policy is a decision-making rule that tells the agent what action to take in each state. It's like a strategy guide: given the current situation, the policy determines the best action. In Q-learning, we learn the optimal policy by finding the action with the highest Q-value in each state. During training, we use an epsilon-greedy policy that balances exploration (trying random actions) and exploitation (using the best learned action)."

---

**Q2: "What is the difference between Q-learning and DQN?"**

**Answer:**
"Q-learning uses a Q-table to store Q-values for each state-action pair. This works well for discrete, small state spaces. DQN (Deep Q-Network) uses a neural network to approximate Q-values, which allows handling continuous or very large state spaces. DQN also uses experience replay to break correlation between consecutive experiences, and a target network to stabilize training. In my Assignment 2, I used Q-learning for the discrete 7×12 grid. In Assignment 3, I used DQN for the continuous maze environment."

---

**Q3: "Explain the Bellman equation."**

**Answer:**
"The Bellman equation expresses the relationship between Q-values of current and future states. It says: the optimal Q-value Q*(s,a) equals the immediate reward r plus the discounted maximum Q-value of the next state. Mathematically: Q*(s,a) = E[r + γ * max Q*(s',a')]. This is recursive - Q-values depend on future Q-values. Q-learning uses this to update Q-values: we compute the target Q-value using the Bellman equation, then move our current Q-value toward that target."

---

**Q4: "What is epsilon-greedy exploration?"**

**Answer:**
"Epsilon-greedy is a strategy that balances exploration and exploitation. With probability epsilon, we choose a random action (exploration) to discover new states and actions. With probability 1-epsilon, we choose the best action according to our Q-values (exploitation) to use what we've learned. During training, epsilon starts high (1.0) for lots of exploration, then decays to a minimum (0.1) for mostly exploitation. This ensures we explore early to learn the environment, then exploit our knowledge later."

---

**Q5: "Why do we need experience replay in DQN?"**

**Answer:**
"Experience replay stores past experiences (state, action, reward, next_state) in a buffer and samples random batches for training. This breaks the correlation between consecutive experiences - without it, we'd train on highly correlated sequential data, which makes training unstable. By sampling random batches, we get diverse, uncorrelated experiences that lead to more stable and efficient learning."

---

**Q6: "What is the purpose of the target network?"**

**Answer:**
"The target network is a separate copy of the main Q-network that's used to compute TD targets. It's updated less frequently (every 100 steps in my implementation). Without it, both Q(s,a) and Q(s',a') would change simultaneously, creating a 'moving target' problem that makes training unstable. The target network provides stable targets for a few steps, allowing the main network to learn more effectively."

---

### Code-Specific Questions

**Q7: "Why did you include `has_lover` in the state space?"**

**Answer:**
"Without `has_lover`, the agent couldn't distinguish between 'has visited lover' and 'hasn't visited lover' states. This caused the agent to get stuck in a loop, repeatedly trying to visit the lover state even after already visiting it. By including `has_lover` as a third dimension in the state [row, col, has_lover], the agent can learn different policies: one for when it hasn't visited the lover (go to lover first), and one for when it has (go directly to goal). This increased success rate from 0% to 100%."

---

**Q8: "Explain your Q-learning update function."**

**Answer:**
"In `update_q_value`, I implement the Q-learning Bellman update. First, I get the current Q-value for the state-action pair. Then I compute the target Q-value: if it's a terminal state, the target is just the reward. Otherwise, it's the reward plus the discounted maximum Q-value of the next state. Finally, I update the Q-value by moving it toward the target: Q(s,a) = Q(s,a) + α * (target - Q(s,a)). The learning rate α controls how much we move toward the target."

---

**Q9: "How does your DQN network architecture work?"**

**Answer:**
"My DQN network takes a 2D continuous state [x, y] as input. It has three hidden layers with 128, 128, and 64 neurons, each followed by ReLU activation. The output layer has 4 neurons, one for each action (up, down, left, right), representing Q-values. The network uses Xavier uniform initialization for stable training. During forward pass, the state goes through the layers and outputs Q-values for all actions, then we select the action with the highest Q-value."

---

**Q10: "What happens in your training loop?"**

**Answer:**
"For each episode, I reset the environment and run until done. In each step: I select an action using epsilon-greedy, take the step in the environment, store the experience in the replay buffer, and if we've collected enough experiences (learning_starts), I sample a batch and train the network. After each episode, I decay epsilon. Once epsilon reaches 0.1, I track consecutive successes. When the agent reaches the goal 100 times in a row, training is complete."

---

### Hyperparameter Questions

**Q11: "How does learning rate affect training?"**

**Answer:**
"Learning rate controls how much we update Q-values or network weights each step. Too high (e.g., 0.5) causes overshooting - Q-values change too quickly and oscillate, making training unstable. Too low (e.g., 0.01) causes slow learning - it takes many episodes to converge. I used 0.1 for Q-learning and 0.001 for DQN, which provided stable, efficient learning. You can see this in training curves: with good learning rate, rewards increase smoothly. With bad learning rate, they oscillate or increase very slowly."

---

**Q12: "What happens if you set gamma too high or too low?"**

**Answer:**
"Gamma (discount factor) determines how much we value future rewards. If gamma is too low (e.g., 0.5), the agent becomes myopic - it only cares about immediate rewards and doesn't plan ahead. This might cause it to take suboptimal short-term actions. If gamma is too high (e.g., 0.999), the agent values future rewards almost as much as immediate ones, which can slow convergence. I used 0.99, which balances immediate and future rewards, allowing the agent to plan ahead while still prioritizing near-term goals."

---

**Q13: "Why do you decay epsilon?"**

**Answer:**
"Epsilon decay transitions the agent from exploration to exploitation. Early in training, we need high epsilon (1.0) to explore the environment and discover which actions lead to rewards. Later, we need low epsilon (0.1) to exploit what we've learned and follow the optimal policy. Without decay, the agent would always explore randomly (if epsilon stays 1.0) or never explore (if epsilon stays 0.0). I used reverse sigmoid decay for a smooth transition, which worked better than linear decay."

---

**Q14: "How does batch size affect DQN training?"**

**Answer:**
"Batch size determines how many experiences we use for each training step. Small batches (e.g., 16) have high variance - each update is noisy and training is less stable, but updates happen more frequently. Large batches (e.g., 256) have low variance - more stable gradients but slower per step. I used 64, which is a standard value that balances stability and speed. You can see this in loss curves: with good batch size, loss decreases smoothly. With too small batch size, loss oscillates."

---

**Q15: "Why update the target network every 100 steps, not every step?"**

**Answer:**
"If we updated the target network every step, it would change as fast as the main network, creating a moving target problem. The TD target Q(s',a') would keep changing, making it hard for the main network to learn. By updating every 100 steps, the target network stays relatively stable, providing consistent targets for the main network to learn from. This stabilizes training. If we updated too infrequently (e.g., every 1000 steps), the target would become outdated and slow learning."

---

## 📊 PART 5: TRAINING BEHAVIOR ANALYSIS

### How to Read Training Curves

**Episode Rewards:**
- Should increase over time
- Early: negative (exploring, hitting walls/danger)
- Later: positive (reaching goal)
- Smooth increase = good learning
- Oscillations = learning rate too high or unstable

**Success Rate:**
- Percentage of episodes reaching goal
- Should increase from 0% to 100%
- Assignment 2: Reached 100% after ~300 episodes
- Assignment 3: Reached 100 consecutive successes after epsilon decayed to 0.1

**Epsilon Decay:**
- Starts at 1.0 (100% exploration)
- Decreases smoothly to 0.1 (10% exploration)
- Reverse sigmoid: smooth S-curve
- Linear: straight line decrease

**Loss (DQN):**
- Should decrease over time
- Early: high (random predictions)
- Later: low (accurate Q-value predictions)
- Oscillations = batch size too small or learning rate too high

---

### Common Training Issues

**1. Agent not learning (rewards stay negative):**
- **Causes**: Learning rate too low, epsilon too high, reward structure too sparse
- **Fix**: Increase learning rate, check reward values, ensure rewards are reachable

**2. Training unstable (oscillating rewards):**
- **Causes**: Learning rate too high, batch size too small
- **Fix**: Decrease learning rate, increase batch size

**3. Agent gets stuck in loop:**
- **Causes**: State space incomplete (missing `has_lover`), reward structure
- **Fix**: Expand state space, adjust rewards

**4. Slow convergence:**
- **Causes**: Learning rate too low, gamma too high, not enough exploration
- **Fix**: Increase learning rate slightly, adjust gamma, ensure epsilon decay is appropriate

---

## 🎤 PART 6: PRESENTATION TIPS

### 10-Minute Presentation Structure

**1. Introduction (1 min)**
- Name, project overview
- Three assignments: Environment, Q-Learning, DQN

**2. Assignment 1: Environment (2 min)**
- 7×12 grid world
- Key features: obstacles, danger zones, lover state
- State space: [row, col, has_lover]
- Reward structure

**3. Assignment 2: Q-Learning (3 min)**
- Q-learning algorithm
- 4D Q-table structure
- Epsilon-greedy with reverse sigmoid decay
- Results: 100% success rate
- Show visualizations (Q-table heatmap, policy arrows)

**4. Assignment 3: DQN (3 min)**
- Continuous state space challenge
- DQN architecture: neural network, experience replay, target network
- Training process: epsilon decay to 0.1, 100 consecutive successes
- Results: successful navigation

**5. Conclusion (1 min)**
- Key achievements
- What you learned
- Thank you

---

### What NOT to Include in Presentation

- ❌ Code snippets (save for Q&A)
- ❌ Too many technical details
- ❌ Long explanations of formulas
- ✅ High-level concepts
- ✅ Visualizations and results
- ✅ Key design decisions

---

### Q&A Preparation

**Be ready to:**
1. Open your code and explain specific functions
2. Draw diagrams (Bellman equation, network architecture)
3. Explain hyperparameter choices
4. Discuss training behavior
5. Answer theory questions

**Practice explaining:**
- Your code line by line
- Why you made certain design choices
- How hyperparameters affect training
- What you would change if starting over

---

## ✅ FINAL CHECKLIST

Before the exam, make sure you can:

- [ ] Explain what a policy is in your own words
- [ ] Describe Q-learning algorithm and Bellman equation
- [ ] Explain DQN architecture and components
- [ ] Discuss epsilon-greedy exploration
- [ ] Explain why `has_lover` is in state space
- [ ] Walk through your Q-learning update function
- [ ] Walk through your DQN training loop
- [ ] Explain how each hyperparameter affects training
- [ ] Read and interpret training curves
- [ ] Discuss your reward structure choices
- [ ] Explain experience replay and target network
- [ ] Compare Q-learning vs DQN

---

## 🎯 KEY FORMULAS TO REMEMBER

**Q-Learning Update:**
```
Q(s,a) = Q(s,a) + α * [r + γ * max Q(s',a') - Q(s,a)]
```

**Bellman Equation:**
```
Q*(s,a) = E[r + γ * max Q*(s',a')]
```

**TD Target (DQN):**
```
target = r + γ * max Q_target(s',a')
```

**Loss (DQN):**
```
Loss = MSE(Q(s,a) - target)
```

**Reverse Sigmoid Decay:**
```
ε = ε_min + (ε_initial - ε_min) / (1 + exp(k*(t - t0)))
```

---

## 💡 TIPS FOR SUCCESS

1. **Speak clearly and confidently** - You know your code!
2. **Use examples** - "For example, when the agent is at state [3,0] without lover..."
3. **Draw diagrams** - Bellman equation, network architecture
4. **Admit if unsure** - "I'm not 100% certain, but I believe..."
5. **Connect theory to code** - "This implements the Bellman equation by..."
6. **Show understanding** - Explain WHY, not just WHAT

---

**Good luck! You've got this! 🚀**

