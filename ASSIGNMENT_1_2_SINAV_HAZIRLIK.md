# ASSIGNMENT 1 & 2 - SIFIRDAN SINAV HAZIRLIĞI

## 📚 İÇİNDEKİLER

1. [Assignment 1: Environment (chid_env.py)](#assignment-1-environment)
2. [Assignment 2: Q-Learning (assignment2_qlearning.py)](#assignment-2-q-learning)
3. [Kod Açıklamaları](#kod-açıklamaları)
4. [Sınav Soruları ve Cevapları](#sınav-soruları-ve-cevapları)

---

# ASSIGNMENT 1: ENVIRONMENT

## 🎯 NE YAPIYORUZ?

Bir **Grid World** (labirent) yapıyoruz. Agent (Scrat) bu labirentte dolaşıp goal'a (palamut) ulaşmaya çalışıyor.

## 📐 ENVIRONMENT YAPISI

### Grid Boyutu
- **7 satır × 12 sütun** = 84 hücre
- Her hücre bir durumu temsil ediyor

### Özel Hücreler

| Sembol | İsim | Ne Yapar? | Reward |
|--------|------|-----------|--------|
| `.` | Empty | Boş hücre, geçilebilir | -1 (living cost) |
| `O` | Obstacle | Buz kristali, geçilemez | -1 (çarparsan) |
| `H` | Hell/Danger | Düşman, ölürsün | -100 (episode biter) |
| `R` | Reward | Mini ödül | +1 |
| `L` | Lover | Scratte, bonus | +100 (ilk alışta) |
| `G` | Goal | Palamut, hedef | +100 (episode biter) |
| `A` | Agent Start | Başlangıç pozisyonu | - |

### Lover Multiplier (6x Çarpan)

**ÇOK ÖNEMLİ!** Lover alırsan:
- Normal Goal: +100 → **Lover ile: +600**
- Normal Danger: -100 → **Lover ile: -600**

**Neden var?** Risk-reward dengesi. Lover almak avantajlı ama riskli!

---

## 🔍 STATE (DURUM) - EN ÖNEMLİ KAVRAM

### State Nedir?

State = Agent'ın o anki durumunu tanımlayan bilgi

### Senin State'in

```python
state = [row, col, has_lover]
#        │    │     │
#        │    │     └── Lover aldın mı? (0 veya 1)
#        │    └── Sütun pozisyonu (0-11)
#        └── Satır pozisyonu (0-6)
```

### Kodda Tanımı

```python
self.observation_space = spaces.Box(
    low=np.array([0, 0, 0]),      # Minimum değerler
    high=np.array([6, 11, 1]),    # Maximum değerler
    shape=(3,),                   # 3 elemanlı vektör
    dtype=np.int32                 # Tam sayı
)
```

### Toplam Kaç State Var?

```
7 (satır) × 12 (sütun) × 2 (lover durumu) = 168 state
```

**Neden has_lover state'in parçası?**
- Aynı pozisyonda olsan bile, lover'lı ve lover'sız durumlar **farklı**!
- Lover alınca reward 6 katına çıkıyor → farklı davranış gerekiyor

---

## 🎮 ACTION SPACE (AKSİYON UZAYI)

### Aksiyonlar

```python
self.action_space = spaces.Discrete(3)
```

| Aksiyon | Değer | Ne Yapar? | Kod |
|---------|-------|-----------|-----|
| UP | 0 | Yukarı git | `row -= 1` |
| DOWN | 1 | Aşağı git | `row += 1` |
| RIGHT | 2 | Sağa git | `col += 1` |

**LEFT neden yok?**
- Tasarım tercihi
- Görevi zorlaştırır (geri dönüş yok)
- Agent daha dikkatli plan yapmalı

### Geçersiz Hareket

Duvara/obstacle'a çarparsan:
- Pozisyon değişmez (aynı yerde kalırsın)
- Ama step sayılır → **-1 living cost** alırsın

---

## 💰 REWARD STRUCTURE (ÖDÜL YAPISI)

### Reward Tablosu

| Durum | Normal Reward | Lover ile | Episode Biter? |
|-------|---------------|-----------|----------------|
| Her adım (living cost) | -1 | -1 | Hayır |
| Goal'a ulaşma | +100 | **+600** | Evet (terminated) |
| Danger'a düşme | -100 | **-600** | Evet (terminated) |
| Lover bulma | +100 | - | Hayır |
| Mini reward (R) | +1 | +1 | Hayır |
| Max step (200) | 0 | 0 | Evet (truncated) |

### Living Cost Neden Var?

**Olmasaydı:**
- 10 adımda goal = +100
- 1000 adımda goal = +100 (aynı!)
- Agent kısa yolu öğrenmez

**Var olduğunda:**
- 10 adımda goal = +100 - 10 = **+90 net**
- 50 adımda goal = +100 - 50 = **+50 net**
- Agent **EN KISA YOLU** öğrenir!

---

## 🔄 ENVIRONMENT METODLARI

### 1. `__init__()` - Başlatma

```python
def __init__(self, num_rows=7, num_cols=12, ...):
    # Maze tanımla
    self.maze = np.array([...])
    
    # Özel pozisyonları parse et
    self.goal_states = [...]
    self.danger_states = [...]
    self.obstacle_states = [...]
    self.lover_state = [...]
    
    # Observation space tanımla
    self.observation_space = spaces.Box(...)
    
    # Action space tanımla
    self.action_space = spaces.Discrete(3)
```

### 2. `reset()` - Episode Başlat

```python
def reset(self):
    # Agent pozisyonunu başlangıca al
    self.position = self.agent_start.copy()
    
    # Lover flag'ini sıfırla
    self.has_lover = False
    
    # State oluştur
    self.state = np.array([
        self.position[0],      # row
        self.position[1],       # col
        int(self.has_lover)     # has_lover (0 veya 1)
    ], dtype=np.int32)
    
    return self.state, {}
```

**Ne yapar?**
- Her yeni episode'da agent başlangıca döner
- Lover flag sıfırlanır
- State oluşturulur ve döndürülür

### 3. `step(action)` - Aksiyon Al

**EN ÖNEMLİ METOD!** Bu metod agent'ın aksiyonunu alır, sonucu hesaplar.

```python
def step(self, action):
    # 1. Yeni pozisyonu hesapla
    new_row = self.position[0]
    new_col = self.position[1]
    
    if action == 0:  # UP
        new_row -= 1
    elif action == 1:  # DOWN
        new_row += 1
    elif action == 2:  # RIGHT
        new_col += 1
    
    # 2. Geçerli pozisyon mu kontrol et
    if self._is_valid_position(new_row, new_col):
        self.position = [new_row, new_col]
    
    # 3. Reward hesapla
    reward = -self.living_cost  # Her adım -1
    
    # 4. Özel durumları kontrol et
    if self.position in self.goal_states:
        if self.has_lover:
            reward += self.goal_reward * self.lover_multiplier  # +600
        else:
            reward += self.goal_reward  # +100
        terminated = True
    elif self.position in self.danger_states:
        if self.has_lover:
            reward -= self.danger_penalty * self.lover_multiplier  # -600
        else:
            reward -= self.danger_penalty  # -100
        terminated = True
    elif self.position == self.lover_state:
        if not self.has_lover:
            reward += 100  # İlk alışta +100
            self.has_lover = True
    elif self.position in self.reward_states:
        reward += self.mini_reward  # +1
    
    # 5. State'i güncelle
    self.state = np.array([
        self.position[0],
        self.position[1],
        int(self.has_lover)
    ], dtype=np.int32)
    
    # 6. Episode bitti mi kontrol et
    truncated = (self.step_count >= self.max_steps)
    done = terminated or truncated
    
    return self.state, reward, terminated, truncated, {}
```

**Adım Adım Ne Oluyor?**

1. **Pozisyon Hesapla**: Aksiyona göre yeni pozisyon
2. **Geçerlilik Kontrolü**: Duvara/obstacle'a çarptı mı?
3. **Reward Hesapla**: Başta -1 (living cost)
4. **Özel Durumlar**:
   - Goal → +100 veya +600 (lover varsa)
   - Danger → -100 veya -600 (lover varsa)
   - Lover → +100 (ilk alışta)
   - Mini reward → +1
5. **State Güncelle**: has_lover değiştiyse state değişir
6. **Episode Bitti mi?**: terminated (goal/danger) veya truncated (max step)

### 4. `render()` - Görselleştir

```python
def render(self):
    if self.render_mode == "pygame":
        # Pygame ile görsel gösterim
    elif self.render_mode == "human":
        # Text tabanlı gösterim
    elif self.render_mode == "ansi":
        # ANSI kodları ile renkli gösterim
```

---

# ASSIGNMENT 2: Q-LEARNING

## 🎯 NE YAPIYORUZ?

Assignment 1'deki environment'ta agent'ı **Q-Learning** algoritması ile eğitiyoruz. Agent deneme-yanılma ile optimal policy'yi öğreniyor.

## 🧠 Q-LEARNING NEDİR?

### Basit Açıklama

**Q-Learning** = "Bu durumda bu aksiyonu yaparsam, uzun vadede ne kadar kazanırım?"

### Q-Function

```
Q(s, a) = Beklenen Toplam Ödül
```

- `s` = state (durum)
- `a` = action (aksiyon)
- `Q(s, a)` = Bu state-action çiftinin değeri

### Örnek

```
Q([3, 0, 0], RIGHT) = 85.5
```

Ne demek?
- State: Satır 3, Sütun 0, Lover yok
- Action: RIGHT (sağa git)
- Değer: 85.5 (bu aksiyonun beklenen toplam getirisi)

---

## 📊 Q-TABLE

### Q-Table Nedir?

Tüm state-action çiftleri için Q değerlerini tutan tablo.

### Senin Q-Table'ın

```python
self.q_table = np.zeros((7, 12, 2, 3))
#                        │   │  │  │
#                        │   │  │  └── 3 aksiyon (UP, DOWN, RIGHT)
#                        │   │  └── 2 lover durumu (0, 1)
#                        │   └── 12 sütun
#                        └── 7 satır
```

**Boyut:** 7 × 12 × 2 × 3 = **504 Q değeri**

### Başlangıç

```python
self.q_table = np.zeros(...)  # Tüm değerler = 0
```

Neden 0? Agent henüz hiçbir şey bilmiyor!

---

## 🔄 Q-LEARNING UPDATE (BELLMAN EQUATION)

### Formül

```
Q(s, a) ← Q(s, a) + α × [R + γ × max Q(s', a') - Q(s, a)]
```

### Parçaları

| Sembol | İsim | Ne Yapar? |
|--------|------|-----------|
| `Q(s, a)` | Current Q | Şu anki Q değeri |
| `α` (alpha) | Learning Rate | Ne kadar hızlı öğren |
| `R` | Reward | Anlık ödül |
| `γ` (gamma) | Discount Factor | Gelecek ödüllerin değeri |
| `max Q(s', a')` | Max Next Q | Sonraki state'teki en iyi Q |
| `R + γ × max Q(s', a')` | Target Q | Hedef Q değeri |
| `Target - Current` | TD Error | Hata miktarı |

### Kodda

```python
def update_q_value(self, state, action, reward, next_state, done):
    state_idx = self.get_state_index(state)
    next_state_idx = self.get_state_index(next_state)
    
    # Mevcut Q değeri
    current_q = self.q_table[state_idx][action]
    
    # Hedef Q değerini hesapla
    if done:
        # Terminal state: gelecek yok
        target_q = reward
    else:
        # Non-terminal: gelecek ödülü ekle
        max_next_q = np.max(self.q_table[next_state_idx])
        target_q = reward + self.discount_factor * max_next_q
    
    # Q-learning update
    self.q_table[state_idx][action] = current_q + self.learning_rate * (target_q - current_q)
```

### Örnek Hesaplama

**Senaryo:**
- State: [3, 5, 0]
- Action: RIGHT
- Reward: -1 (living cost)
- Next state: [3, 6, 0]
- Max next Q: 80
- Gamma: 0.99

**Hesaplama:**
```
target_q = -1 + 0.99 × 80 = -1 + 79.2 = 78.2
current_q = 50 (örnek)
TD error = 78.2 - 50 = 28.2
new_q = 50 + 0.08 × 28.2 = 50 + 2.26 = 52.26
```

---

## 🎲 EPSILON-GREEDY POLICY

### Ne Demek?

**Epsilon-greedy** = Exploration (keşif) ve Exploitation (sömürme) dengesi

### Nasıl Çalışır?

```python
def choose_action(self, state, training=True):
    if training and np.random.rand() < self.epsilon:
        # Exploration: Random aksiyon seç
        return self.env.action_space.sample()
    else:
        # Exploitation: En iyi aksiyonu seç
        return int(np.argmax(self.q_table[state_idx]))
```

### Epsilon Değerleri

| Epsilon | Ne Yapar? |
|---------|-----------|
| ε = 1.0 | %100 random (başlangıç) |
| ε = 0.5 | %50 random, %50 greedy |
| ε = 0.1 | %10 random, %90 greedy |
| ε = 0.0 | %100 greedy (test modu) |

### Neden Azalıyor?

- **Başta:** Çok explore et (ortamı tanı)
- **Sonra:** Exploit et (öğrendiğini kullan)

**Sabit kalsa ne olur?**
- Sürekli random hareketler
- Öğrendiğin policy'yi kullanamazsın
- Performans düşük kalır

---

## 📉 EPSILON DECAY (AZALMA)

### İki Yöntem

#### 1. Multiplicative Decay (Basit)

```python
self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
# Örnek: epsilon = 1.0 × 0.995 = 0.995
```

#### 2. Reverse Sigmoid Decay (Senin Kodun)

```python
def reverse_sigmoid_decay(t, epsilon_initial, epsilon_min, k, t0):
    return epsilon_min + (epsilon_initial - epsilon_min) / (1 + np.exp(k * (t - t0)))
```

**Neden Reverse Sigmoid?**
- Smooth geçiş (ani düşüş yok)
- S şeklinde azalma
- Daha kontrollü

**Parametreler:**
- `k`: Decay hızı (ne kadar hızlı azalacak)
- `t0`: Inflection point (en hızlı azalmanın olduğu episode)

---

## 🏋️ TRAINING LOOP

### Kod Yapısı

```python
def train(self, num_episodes=1000):
    for episode in range(num_episodes):
        # 1. Environment'ı reset et
        state, info = self.env.reset()
        done = False
        
        # 2. Episode boyunca döngü
        while not done:
            # 3. Aksiyon seç (epsilon-greedy)
            action = self.choose_action(state, training=True)
            
            # 4. Environment'ta aksiyonu al
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # 5. Q değerini güncelle (Bellman equation)
            self.update_q_value(state, action, reward, next_state, done)
            
            # 6. State'i güncelle
            state = next_state
        
        # 7. Epsilon'u azalt (decay)
        self.epsilon = reverse_sigmoid_decay(...)
```

### Adım Adım Ne Oluyor?

1. **Reset**: Agent başlangıca döner
2. **Action Seç**: Epsilon-greedy ile aksiyon seç
3. **Step**: Environment'ta aksiyonu al, reward al
4. **Update**: Q değerini güncelle (öğren!)
5. **Decay**: Epsilon'u azalt (daha az explore, daha çok exploit)

---

## 🧪 TEST MODU

### Ne Fark Var?

```python
def test(self, num_episodes=10):
    for episode in range(num_episodes):
        while not done:
            # Greedy policy kullan (epsilon = 0)
            action = self.choose_action(state, training=False)
            # ...
```

**Fark:**
- `training=False` → epsilon kullanılmaz
- Her zaman en iyi aksiyon seçilir (greedy)
- Random yok!

---

## 📈 GÖRSELLEŞTİRME

### 1. Q-Table Visualization

```python
visualize_q_table(q_table, env, save_path="q_table.png")
```

**Ne gösterir?**
- Her aksiyon için Q değerlerini heatmap olarak
- Yüksek değerler = sarı
- Düşük değerler = mor

### 2. Policy Visualization

```python
visualize_policy(q_table, env, save_path="policy.png")
```

**Ne gösterir?**
- Her state'te en iyi aksiyonu ok ile
- Optimal policy'yi görselleştirir

### 3. Training Curves

```python
plot_training_curves(training_stats, save_path="curves.png")
```

**Ne gösterir?**
- Episode reward'ları
- Success rate
- Epsilon değişimi

---

# KOD AÇIKLAMALARI

## assignment2_qlearning.py - Satır Satır

### 1. QLearningAgent Class

```python
class QLearningAgent:
    def __init__(self, env, learning_rate=0.1, ...):
        self.env = env
        self.learning_rate = learning_rate  # α (alpha)
        self.discount_factor = discount_factor  # γ (gamma)
        self.epsilon = epsilon  # Exploration rate
        self.q_table = np.zeros((7, 12, 2, 3))  # Q-table başlat
```

**Ne yapar?**
- Agent'ı başlatır
- Hyperparameter'leri ayarlar
- Q-table'ı sıfırlarla doldurur

### 2. choose_action()

```python
def choose_action(self, state, training=True):
    state_idx = self.get_state_index(state)
    
    if training and np.random.rand() < self.epsilon:
        return self.env.action_space.sample()  # Random
    else:
        return int(np.argmax(self.q_table[state_idx]))  # Greedy
```

**Ne yapar?**
- Epsilon olasılıkla random aksiyon
- 1-epsilon olasılıkla en iyi aksiyon

### 3. update_q_value()

```python
def update_q_value(self, state, action, reward, next_state, done):
    current_q = self.q_table[state_idx][action]
    
    if done:
        target_q = reward  # Terminal state
    else:
        max_next_q = np.max(self.q_table[next_state_idx])
        target_q = reward + self.discount_factor * max_next_q
    
    # Update
    self.q_table[state_idx][action] = current_q + self.learning_rate * (target_q - current_q)
```

**Ne yapar?**
- Bellman equation'ı uygular
- Q değerini günceller

### 4. train()

```python
def train(self, num_episodes=1000):
    for episode in range(num_episodes):
        state, info = self.env.reset()
        done = False
        
        while not done:
            action = self.choose_action(state, training=True)
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            self.update_q_value(state, action, reward, next_state, done)
            state = next_state
        
        # Epsilon decay
        self.epsilon = reverse_sigmoid_decay(...)
```

**Ne yapar?**
- Episode'ları çalıştırır
- Her adımda Q değerini günceller
- Epsilon'u azaltır

---

# SINAV SORULARI VE CEVAPLARI

## ASSIGNMENT 1 SORULARI

### SORU 1: "Environment'ını anlat."

**CEVAP:**
"7x12 boyutunda bir grid world yaptım. Ice Age temalı - Scrat isimli sincap agent, palamuta (goal) ulaşmaya çalışıyor. Engellerden kaçınması, tehlikeli bölgelerden uzak durması lazım. Bir de Scratte var, lover - onu alırsa reward 6 katına çıkıyor ama risk de 6 kat artıyor."

### SORU 2: "State neyi içeriyor?"

**CEVAP:**
"3 elemanlı bir vektör: row (satır 0-6), column (sütun 0-11), has_lover (0 veya 1). Toplam 7×12×2 = 168 farklı state var."

### SORU 3: "Neden has_lover state'in parçası?"

**CEVAP:**
"Çünkü lover alındığında reward 6 katına çıkıyor. Yani agent aynı pozisyonda olsa bile, lover'lı ve lover'sız durumlar farklı değerlere sahip. Agent'ın farklı davranması gerekiyor - mesela lover aldıysan risk almaman lazım çünkü ölürsen 6 kat ceza alırsın."

### SORU 4: "Living cost neden -1?"

**CEVAP:**
"0 olsa agent acele etmezdi. 10 adımda da 1000 adımda da aynı reward'ı alırdı. -1 sayesinde her adım maliyetli, agent en kısa yolu öğreniyor."

### SORU 5: "step() metodunda ne oluyor?"

**CEVAP:**
"Önce yeni pozisyonu hesaplıyorum. Geçerli mi kontrol ediyorum. Reward hesaplıyorum - başta -1 living cost. Sonra özel durumları kontrol ediyorum - goal, danger, lover, mini reward. State'i güncelliyorum, özellikle has_lover değiştiyse. Episode bitti mi kontrol ediyorum."

---

## ASSIGNMENT 2 SORULARI

### SORU 1: "Q-Learning algoritmasını anlat."

**CEVAP:**
"Off-policy TD control algoritması. Agent epsilon-greedy ile aksiyon seçiyor - bazen random bazen en iyi. Her adımda Q değerini güncelliyor: Q(s,a) = Q(s,a) + α × [R + γ × maxQ(s') - Q(s,a)]. Zamanla epsilon azalıyor, daha çok exploit ediyor."

### SORU 2: "Q-table boyutu ne?"

**CEVAP:**
"7 × 12 × 2 × 3 = 504 değer. 7 satır, 12 sütun, 2 lover durumu, 3 aksiyon."

### SORU 3: "Bellman equation'ı yaz ve açıkla."

**CEVAP:**
"Q(s, a) = R + γ × max Q(s', a'). Yani bir state-action'ın değeri, anlık reward artı indirimli gelecek değerine eşit. R anlık reward, γ discount factor, max Q(s', a') sonraki state'teki en iyi Q değeri."

### SORU 4: "Epsilon-greedy ne demek?"

**CEVAP:**
"Exploration-exploitation dengesi için kullandığım strateji. Epsilon olasılıkla random aksiyon yapıyorum (explore), 1-epsilon olasılıkla en iyi aksiyonu (exploit). Başta epsilon=1 yani full random, sonra yavaş yavaş azalıyor."

### SORU 5: "Off-policy ne demek?"

**CEVAP:**
"Davrandığım policy ile öğrendiğim policy farklı. Ben epsilon-greedy ile davranıyorum ama öğrendiğim greedy/optimal policy. Update'te max Q kullanıyorum, bu greedy policy. Davranırken bazen random yapıyorum, bu epsilon-greedy. İkisi farklı olduğu için off-policy."

### SORU 6: "Alpha değerin ne? Neden bu değeri seçtin?"

**CEVAP:**
"0.08 kullandım. Düşük bir değer - yavaş ama stabil öğrenme sağlıyor. Çok yüksek olsa Q değerleri salınım yapardı, converge etmezdi. 0.08 ile yavaş yavaş ama güvenli öğreniyor."

### SORU 7: "Gamma 0 olsa ne olurdu?"

**CEVAP:**
"Agent miyop olurdu, sadece anlık reward'a bakardı. Her adım -1 olduğu için hiçbir yere gitmek istemezdi. Goal'ın +100 olduğunu göremezdi çünkü gelecek değeri sıfır sayardı."

### SORU 8: "Reverse sigmoid decay ne?"

**CEVAP:**
"Normal decay'de epsilon = epsilon × 0.995 gibi çarparak azaltıyorsun, üstel azalma. Reverse sigmoid'de S şeklinde azalıyor - başta yavaş, ortada hızlı, sonda tekrar yavaş. Daha smooth bir geçiş sağlıyor, ani düşüşler yok."

### SORU 9: "Test'te epsilon kaç?"

**CEVAP:**
"Sıfır. Test'te exploration yapmıyorum, sadece öğrendiğim policy'yi kullanıyorum. Her state'te argmax Q alıyorum, yani en iyi aksiyonu seçiyorum. Random yok."

### SORU 10: "update_q_value() metodunda ne oluyor?"

**CEVAP:**
"Önce mevcut Q değerini alıyorum. Sonra hedef Q değerini hesaplıyorum - eğer terminal state ise sadece reward, değilse reward + gamma × max next Q. Sonra TD error'ü hesaplayıp (target - current), bunu learning rate ile çarpıp mevcut Q'ya ekliyorum."

---

## 🎯 HIZLI REFERANS

### Formüller

| Formül | Açıklama |
|--------|----------|
| `Q(s,a) ← Q(s,a) + α[R + γ max Q(s',a') - Q(s,a)]` | Q-Learning Update |
| `ε-greedy: P(random) = ε, P(greedy) = 1-ε` | Exploration Policy |
| `V(s) = max Q(s, a)` | Value Function |

### Değerler

| Parametre | Değer | Amaç |
|-----------|-------|------|
| Grid | 7×12 | Environment boyutu |
| State | (row, col, has_lover) | 3 boyutlu observation |
| Actions | 3 (UP, DOWN, RIGHT) | Hareket seçenekleri |
| α | 0.08 | Yavaş-stabil öğrenme |
| γ | 0.995 | Uzun vadeli planlama |
| ε | 1.0 → 0.1 | Azalan exploration |
| Living cost | -1 | Kısa yol teşviki |
| Goal reward | +100 (×6 with lover) | Hedefe ulaşma |

---

## ✅ SINAV ÖNCESİ KONTROL LİSTESİ

- [ ] Environment yapısını anladım (7×12, özel hücreler)
- [ ] State'in ne olduğunu biliyorum (row, col, has_lover)
- [ ] Action space'i biliyorum (3 aksiyon: UP, DOWN, RIGHT)
- [ ] Reward yapısını biliyorum (living cost, goal, danger, lover)
- [ ] Q-table'ın boyutunu biliyorum (7×12×2×3 = 504)
- [ ] Bellman equation'ı yazabilirim
- [ ] Epsilon-greedy'yi açıklayabilirim
- [ ] Off-policy kavramını biliyorum
- [ ] Alpha, gamma, epsilon'u açıklayabilirim
- [ ] Training loop'u anladım
- [ ] Test modunun farkını biliyorum

---

**BAŞARILAR! 🚀**

