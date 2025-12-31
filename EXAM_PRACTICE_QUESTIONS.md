# PADM Exam - Practice Questions
## Test Yourself Before the Exam

---

## 🎯 THEORY QUESTIONS

### Question 1: Policy
**Q: "Describe what a policy is in your own words."**

**Your Answer:**
_________________________________________________________
_________________________________________________________
_________________________________________________________

**Model Answer:**
"A policy is a decision-making rule that tells the agent what action to take in each state. It's like a strategy guide: given the current situation, the policy determines the best action. In Q-learning, we learn the optimal policy by finding the action with the highest Q-value in each state."

---

### Question 2: Q-Learning vs DQN
**Q: "What is the main difference between Q-learning and DQN?"**

**Your Answer:**
_________________________________________________________
_________________________________________________________
_________________________________________________________

**Model Answer:**
"Q-learning uses a Q-table to store Q-values for each state-action pair, which works for discrete, small state spaces. DQN uses a neural network to approximate Q-values, allowing it to handle continuous or very large state spaces. DQN also uses experience replay and a target network for stability."

---

### Question 3: Bellman Equation
**Q: "Explain the Bellman equation and how Q-learning uses it."**

**Your Answer:**
_________________________________________________________
_________________________________________________________
_________________________________________________________

**Model Answer:**
"The Bellman equation expresses that the optimal Q-value equals the immediate reward plus the discounted maximum Q-value of the next state: Q*(s,a) = r + γ*max Q*(s',a'). Q-learning uses this to update Q-values: we compute the target using the Bellman equation, then move our current Q-value toward that target."

---

### Question 4: Epsilon-Greedy
**Q: "What is epsilon-greedy exploration and why do we decay epsilon?"**

**Your Answer:**
_________________________________________________________
_________________________________________________________
_________________________________________________________

**Model Answer:**
"Epsilon-greedy balances exploration and exploitation. With probability epsilon, we choose a random action to explore. With probability 1-epsilon, we choose the best action to exploit. We decay epsilon from 1.0 to 0.1 because early training needs exploration to discover the environment, while later training needs exploitation to use learned knowledge."

---

### Question 5: Experience Replay
**Q: "Why do we need experience replay in DQN?"**

**Your Answer:**
_________________________________________________________
_________________________________________________________
_________________________________________________________

**Model Answer:**
"Experience replay stores past experiences and samples random batches for training. This breaks the correlation between consecutive experiences. Without it, we'd train on highly correlated sequential data, making training unstable. Random batches provide diverse, uncorrelated experiences for stable learning."

---

### Question 6: Target Network
**Q: "What is the purpose of the target network in DQN?"**

**Your Answer:**
_________________________________________________________
_________________________________________________________
_________________________________________________________

**Model Answer:**
"The target network is a separate copy of the main Q-network used to compute TD targets. It's updated less frequently (every 100 steps). Without it, both Q(s,a) and Q(s',a') would change simultaneously, creating a moving target problem that makes training unstable. The target network provides stable targets."

---

## 💻 CODE QUESTIONS

### Question 7: State Space
**Q: "Why did you include 'has_lover' in your state space?"**

**Your Answer:**
_________________________________________________________
_________________________________________________________
_________________________________________________________

**Model Answer:**
"Without has_lover, the agent couldn't distinguish between 'has visited lover' and 'hasn't visited lover' states. This caused the agent to get stuck in a loop, repeatedly trying to visit the lover even after already visiting it. By including has_lover, the agent learns different policies for each case, increasing success rate from 0% to 100%."

---

### Question 8: Q-Learning Update
**Q: "Walk me through your update_q_value function. What does each line do?"**

**Your Answer:**
_________________________________________________________
_________________________________________________________
_________________________________________________________

**Model Answer:**
"First, I convert the state to an index (row, col, has_lover). I get the current Q-value for that state-action pair. Then I compute the target: if it's a terminal state (done=True), the target is just the reward. Otherwise, it's reward plus gamma times the maximum Q-value of the next state. Finally, I update the Q-value using the Q-learning formula: Q = Q + alpha * (target - Q)."

---

### Question 9: Action Selection
**Q: "How does your choose_action function work?"**

**Your Answer:**
_________________________________________________________
_________________________________________________________
_________________________________________________________

**Model Answer:**
"If we're training and a random number is less than epsilon, we return a random action from the action space (exploration). Otherwise, we find the state index, look up all Q-values for that state, and return the action with the highest Q-value using argmax (exploitation)."

---

### Question 10: DQN Training
**Q: "Explain what happens in your DQN training step."**

**Your Answer:**
_________________________________________________________
_________________________________________________________
_________________________________________________________

**Model Answer:**
"First, I sample a batch of experiences from the replay buffer. I pass the states through the main Q-network to get current Q-values, then select the Q-values for the actions that were actually taken. I compute TD targets using the target network: for each experience, if done, target is just reward; otherwise, target is reward plus gamma times the maximum Q-value from the target network for the next state. I compute MSE loss between current Q-values and targets, then backpropagate and update the network weights."

---

### Question 11: Training Loop
**Q: "What happens in your training loop for each episode?"**

**Your Answer:**
_________________________________________________________
_________________________________________________________
_________________________________________________________

**Model Answer:**
"For each episode, I reset the environment to get the initial state. Then I loop until done: I select an action using epsilon-greedy, take a step in the environment, store the experience in the replay buffer, increment the step count, and if we've collected enough experiences (learning_starts), I sample a batch and train the network. After the episode, I decay epsilon and check if we've reached 100 consecutive successes (after epsilon reaches 0.1)."

---

## ⚙️ HYPERPARAMETER QUESTIONS

### Question 12: Learning Rate
**Q: "How does learning rate affect training? What happens if it's too high or too low?"**

**Your Answer:**
_________________________________________________________
_________________________________________________________
_________________________________________________________

**Model Answer:**
"Learning rate controls how much we update Q-values or network weights each step. Too high (e.g., 0.5) causes overshooting - values change too quickly and oscillate, making training unstable. Too low (e.g., 0.01) causes slow learning - takes many episodes to converge. I used 0.1 for Q-learning and 0.001 for DQN, which provided stable, efficient learning."

---

### Question 13: Discount Factor
**Q: "What happens if you set gamma too high or too low? Why did you choose 0.99?"**

**Your Answer:**
_________________________________________________________
_________________________________________________________
_________________________________________________________

**Model Answer:**
"Gamma determines how much we value future rewards. Too low (e.g., 0.5) makes the agent myopic - only cares about immediate rewards, doesn't plan ahead, might take suboptimal short-term actions. Too high (e.g., 0.999) values future rewards almost as much as immediate ones, which can slow convergence. I chose 0.99 to balance immediate and future rewards, allowing the agent to plan ahead while still prioritizing near-term goals."

---

### Question 14: Epsilon Decay
**Q: "Why do you decay epsilon? What happens if you don't?"**

**Your Answer:**
_________________________________________________________
_________________________________________________________
_________________________________________________________

**Model Answer:**
"Epsilon decay transitions from exploration to exploitation. Early training needs high epsilon (1.0) to explore and discover which actions lead to rewards. Later training needs low epsilon (0.1) to exploit learned knowledge and follow the optimal policy. Without decay, if epsilon stays 1.0, the agent always explores randomly and never uses what it learned. If epsilon stays 0.0, it never explores and might miss better strategies."

---

### Question 15: Batch Size
**Q: "How does batch size affect DQN training? Why did you choose 64?"**

**Your Answer:**
_________________________________________________________
_________________________________________________________
_________________________________________________________

**Model Answer:**
"Batch size determines how many experiences we use for each training step. Small batches (e.g., 16) have high variance - each update is noisy and training is less stable, but updates happen more frequently. Large batches (e.g., 256) have low variance - more stable gradients but slower per step. I chose 64, which is a standard value that balances stability and speed."

---

### Question 16: Target Network Update
**Q: "Why update the target network every 100 steps instead of every step?"**

**Your Answer:**
_________________________________________________________
_________________________________________________________
_________________________________________________________

**Model Answer:**
"If we updated every step, the target network would change as fast as the main network, creating a moving target problem. The TD target Q(s',a') would keep changing, making it hard for the main network to learn. By updating every 100 steps, the target network stays relatively stable, providing consistent targets for the main network to learn from, which stabilizes training."

---

## 📊 TRAINING BEHAVIOR QUESTIONS

### Question 17: Reading Curves
**Q: "Looking at your training curves, what do they tell you about the training process?"**

**Your Answer:**
_________________________________________________________
_________________________________________________________
_________________________________________________________

**Model Answer:**
"The episode rewards curve shows rewards increasing from negative (early exploration, hitting walls/danger) to positive (reaching goal). The success rate increases from 0% to 100%, showing the agent learned the optimal policy. The epsilon curve shows smooth decay from 1.0 to 0.1. The loss curve (DQN) decreases over time, showing the network is learning to predict Q-values accurately. All curves show smooth, stable learning without oscillations."

---

### Question 18: Training Issues
**Q: "What would you do if your agent's rewards were oscillating during training?"**

**Your Answer:**
_________________________________________________________
_________________________________________________________
_________________________________________________________

**Model Answer:**
"Oscillating rewards indicate unstable training. I would first check the learning rate - if it's too high, I'd decrease it. For DQN, I'd also check batch size - if it's too small, I'd increase it. I might also check if the target network is updating too frequently. The goal is to make training more stable by reducing the variance in updates."

---

### Question 19: Slow Convergence
**Q: "Your agent is learning but very slowly. What hyperparameters would you adjust?"**

**Your Answer:**
_________________________________________________________
_________________________________________________________
_________________________________________________________

**Model Answer:**
"If learning is too slow, I'd first check the learning rate - if it's too low, I'd increase it slightly. I'd also check if epsilon is decaying too quickly, leaving too little exploration time. For DQN, I might increase batch size slightly for more stable gradients, or check if the target network update frequency is appropriate. However, I'd be careful not to make changes too drastic, as that could destabilize training."

---

## 🎯 DESIGN DECISION QUESTIONS

### Question 20: Reward Structure
**Q: "Why did you choose these specific reward values?"**

**Your Answer:**
_________________________________________________________
_________________________________________________________
_________________________________________________________

**Model Answer:**
"For Assignment 1 & 2, I used -1 for step cost to encourage efficiency, +100 for goal to make it highly desirable, -100 for danger to strongly discourage it, and +100 for lover to incentivize visiting it. The 6x multiplier with lover (+600 goal reward) creates strategic depth. For Assignment 3, I used -0.01 step penalty, -1.0 wall collision, -100.0 danger zone, +100.0 goal, and distance-based shaping (+0.1*(1-distance)) to guide the agent toward the goal."

---

### Question 21: Network Architecture
**Q: "Why did you choose this specific DQN architecture (128-128-64)?"**

**Your Answer:**
_________________________________________________________
_________________________________________________________
_________________________________________________________

**Model Answer:**
"I chose a three-layer architecture with 128, 128, and 64 neurons. The first two layers (128) provide enough capacity to learn complex Q-value mappings for the continuous state space. The third layer (64) reduces dimensionality before the output layer. This architecture is large enough to learn the task but not so large that it overfits or trains too slowly. I used ReLU activations for non-linearity."

---

### Question 22: Epsilon Decay Strategy
**Q: "Why did you use reverse sigmoid decay instead of linear decay?"**

**Your Answer:**
_________________________________________________________
_________________________________________________________
_________________________________________________________

**Model Answer:**
"Reverse sigmoid decay provides a smoother S-curve transition between exploration and exploitation compared to linear decay. Linear decay has a constant rate of change, while reverse sigmoid has a gradual start, faster middle transition, and gradual end. This smooth transition helps the agent gradually shift from exploration to exploitation without abrupt changes that could destabilize learning."

---

## 🔄 COMPARISON QUESTIONS

### Question 23: Q-Learning vs DQN
**Q: "When would you use Q-learning vs DQN?"**

**Your Answer:**
_________________________________________________________
_________________________________________________________
_________________________________________________________

**Model Answer:**
"Use Q-learning for discrete, small state spaces (like my 7×12 grid with 168 states). It's simpler, faster to train, and the Q-table is interpretable. Use DQN for continuous or very large state spaces (like my continuous maze). DQN can handle infinite states but requires more hyperparameter tuning and computational resources. The choice depends on the state space size and complexity."

---

### Question 24: State Representations
**Q: "How does the state representation differ between Assignment 2 and 3?"**

**Your Answer:**
_________________________________________________________
_________________________________________________________
_________________________________________________________

**Model Answer:**
"Assignment 2 uses discrete states: [row, col, has_lover] where row and col are integers (0-6, 0-11) and has_lover is binary (0 or 1). This creates 168 discrete states. Assignment 3 uses continuous states: [x, y] where x and y are floats in [0, 1]. This creates infinite possible states, which is why we need DQN with a neural network instead of a Q-table."

---

## ✅ SELF-CHECK

After answering these questions, check:
- [ ] Can you explain each answer clearly?
- [ ] Do you understand the concepts, not just memorized?
- [ ] Can you connect theory to your code?
- [ ] Can you discuss hyperparameter effects?
- [ ] Can you read training curves?

**If yes to all, you're ready! 🚀**

