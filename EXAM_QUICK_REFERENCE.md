# PADM Exam - Quick Reference Card
## Cheat Sheet for Oral Exam

---

## 🎯 CORE CONCEPTS (30 seconds each)

### Policy
"A strategy that maps states to actions. Optimal policy chooses action with highest Q-value: π*(s) = argmax Q(s,a)"

### Q-Learning
"Model-free RL algorithm. Updates Q-values: Q(s,a) = Q(s,a) + α[r + γ*max Q(s',a') - Q(s,a)]"

### DQN
"Neural network approximates Q-values. Uses experience replay (random batches) and target network (stable targets) for stability."

### Epsilon-Greedy
"With probability ε: random action (explore). With probability 1-ε: best action (exploit). Decays from 1.0 to 0.1."

### Bellman Equation
"Q*(s,a) = r + γ*max Q*(s',a'). Recursive: current Q-value depends on future Q-values."

---

## 📝 CODE EXPLANATIONS (1 minute each)

### Assignment 1: `step()` method
"Calculates new position, checks validity, computes reward (lover/danger/goal), updates state [row, col, has_lover], returns state/reward/done."

### Assignment 2: `update_q_value()`
"Gets current Q-value, computes target (reward + γ*max next Q if not done, else just reward), updates: Q = Q + α*(target - Q)."

### Assignment 2: `choose_action()`
"If training and random < epsilon: return random action (explore). Else: return argmax Q(s,a) (exploit)."

### Assignment 3: DQN training step
"Sample batch from replay buffer. Compute current Q(s,a). Compute TD target using target network: r + γ*max Q_target(s',a'). Compute MSE loss. Backpropagate. Update target network every 100 steps."

---

## ⚙️ HYPERPARAMETERS (30 seconds each)

### Learning Rate (α)
- **Q-Learning: 0.1** - Controls Q-value update speed
- **DQN: 0.001** - Standard for neural networks
- **Too high**: Unstable, oscillates
- **Too low**: Slow convergence

### Discount Factor (γ)
- **0.99** - Values future rewards highly
- **Too low**: Myopic (only immediate rewards)
- **Too high**: Slow convergence

### Epsilon
- **Starts: 1.0** (100% exploration)
- **Ends: 0.1** (10% exploration)
- **Decay**: Reverse sigmoid (smooth transition)
- **Why decay**: Explore early, exploit later

### Batch Size (DQN)
- **64** - Standard value
- **Too small**: High variance, unstable
- **Too large**: Slow per step

### Target Network Update
- **Every 100 steps** - Prevents moving target problem
- **Too frequent**: Unstable
- **Too infrequent**: Outdated targets

---

## ❓ COMMON QUESTIONS & ANSWERS

**Q: "Describe what a policy is in your own words."**
A: "A policy is a decision-making rule that tells the agent what action to take in each state. In Q-learning, the optimal policy chooses the action with the highest Q-value: π*(s) = argmax Q(s,a)."

**Q: "Why did you include has_lover in the state?"**
A: "Without it, the agent couldn't distinguish 'has visited lover' from 'hasn't visited lover', causing it to loop trying to re-visit the lover. With has_lover, it learns different policies for each case."

**Q: "How does learning rate affect training?"**
A: "Too high (0.5): Q-values change too quickly, training oscillates. Too low (0.01): Slow convergence. I used 0.1 for Q-learning, which provided stable, efficient learning."

**Q: "What is experience replay?"**
A: "Stores past experiences in a buffer, samples random batches for training. Breaks correlation between consecutive experiences, making training more stable."

**Q: "Why use a target network?"**
A: "Provides stable TD targets. Without it, both Q(s,a) and Q(s',a') change simultaneously (moving target problem), making training unstable."

**Q: "Explain your Q-learning update."**
A: "I compute the target Q-value: if terminal, just reward; else reward + γ*max Q(s',a'). Then update: Q(s,a) = Q(s,a) + α*(target - Q(s,a))."

**Q: "How does gamma affect behavior?"**
A: "Gamma determines future reward importance. Low (0.5): myopic, only immediate rewards. High (0.99): long-term planning. I used 0.99 for strategic planning."

**Q: "Why decay epsilon?"**
A: "Early training needs exploration (ε=1.0) to discover environment. Later needs exploitation (ε=0.1) to use learned knowledge. Decay provides smooth transition."

---

## 📊 TRAINING BEHAVIOR

### Good Training Signs
- ✅ Rewards increase smoothly
- ✅ Success rate increases to 100%
- ✅ Loss decreases (DQN)
- ✅ Epsilon decays smoothly

### Bad Training Signs
- ❌ Rewards oscillate → Learning rate too high
- ❌ Rewards stay negative → Learning rate too low or reward structure
- ❌ Loss oscillates → Batch size too small
- ❌ Slow convergence → Learning rate too low

---

## 🔢 KEY NUMBERS TO REMEMBER

- **State space (A1/A2)**: 7 × 12 × 2 = 168 states
- **State space (A3)**: Continuous [x, y] ∈ [0,1]²
- **Actions (A1/A2)**: 3 (Up, Down, Right)
- **Actions (A3)**: 4 (Up, Down, Left, Right)
- **Q-table shape (A2)**: (7, 12, 2, 3)
- **DQN architecture**: 2 → 128 → 128 → 64 → 4
- **Learning rate (Q)**: 0.1
- **Learning rate (DQN)**: 0.001
- **Gamma**: 0.99
- **Epsilon min**: 0.1 (A3 fixed), 0.05 (A2)
- **Batch size**: 64
- **Buffer size**: 10,000
- **Target update**: Every 100 steps
- **Learning starts**: 500 steps

---

## 🎤 PRESENTATION STRUCTURE

1. **Intro** (1 min): Name, 3 assignments
2. **A1** (2 min): Environment, state space, rewards
3. **A2** (3 min): Q-learning, Q-table, results
4. **A3** (3 min): DQN, architecture, training
5. **Conclusion** (1 min): Achievements, thanks

---

## ✅ FINAL CHECKLIST

Before exam, can you explain:
- [ ] What is a policy?
- [ ] Q-learning update formula
- [ ] Why has_lover in state?
- [ ] Experience replay purpose
- [ ] Target network purpose
- [ ] How learning rate affects training
- [ ] How gamma affects behavior
- [ ] Why epsilon decay?
- [ ] Your code's key functions
- [ ] Your hyperparameter choices

---

**Remember: Speak clearly, use examples, draw diagrams, connect theory to code!**

**Good luck! 🚀**

