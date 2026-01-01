# EPSILON-GREEDY, EPSILON ve EPSILON_DECAY - DETAYLI AÇIKLAMA

## 🎯 EPSILON-GREEDY NEDİR?

**Epsilon-greedy** = Exploration (keşif) ve Exploitation (sömürme) dengesini sağlayan bir strateji.

### Basit Açıklama

Agent iki şey yapabilir:
1. **Exploration (Keşif)**: Random aksiyon seç, yeni şeyler öğren
2. **Exploitation (Sömürme)**: En iyi aksiyonu seç, öğrendiğini kullan

**Epsilon-greedy** bu ikisini dengeler!

---

## 📊 EPSILON NEDİR?

**Epsilon (ε)** = Random aksiyon seçme olasılığı

### Değer Aralığı

```
ε ∈ [0, 1]
```

- **ε = 1.0** → %100 random (tamamen keşif)
- **ε = 0.5** → %50 random, %50 en iyi
- **ε = 0.1** → %10 random, %90 en iyi
- **ε = 0.0** → %100 en iyi (tamamen sömürme, random yok)

### Kodda

```python
self.epsilon = 1.0  # Başlangıç değeri
self.epsilon_min = 0.1  # Minimum değer (asla bu değerin altına düşmez)
```

---

## 🎲 EPSILON-GREEDY NASIL ÇALIŞIR?

### Kod İncelemesi

```python
def choose_action(self, state, training=True):
    state_idx = self.get_state_index(state)
    
    if training and np.random.rand() < self.epsilon:
        # Exploration: Random aksiyon seç
        return self.env.action_space.sample()
    else:
        # Exploitation: En iyi aksiyonu seç
        return int(np.argmax(self.q_table[state_idx]))
```

### Adım Adım Ne Oluyor?

1. **Random sayı üret**: `np.random.rand()` → 0 ile 1 arası sayı
2. **Karşılaştır**: `random < epsilon` mi?
   - **EVET** → Random aksiyon seç (exploration)
   - **HAYIR** → En iyi aksiyonu seç (exploitation)

### Örnek Senaryolar

#### Senaryo 1: ε = 1.0 (Başlangıç)

```python
random = 0.7
epsilon = 1.0
0.7 < 1.0 → TRUE → Random aksiyon seç
```

**Sonuç:** %100 random (her zaman exploration)

#### Senaryo 2: ε = 0.5 (Orta)

```python
random = 0.3
epsilon = 0.5
0.3 < 0.5 → TRUE → Random aksiyon seç

random = 0.7
epsilon = 0.5
0.7 < 0.5 → FALSE → En iyi aksiyonu seç
```

**Sonuç:** %50 random, %50 en iyi

#### Senaryo 3: ε = 0.1 (Sonraki Aşamalar)

```python
random = 0.05
epsilon = 0.1
0.05 < 0.1 → TRUE → Random aksiyon seç

random = 0.15
epsilon = 0.1
0.15 < 0.1 → FALSE → En iyi aksiyonu seç
```

**Sonuç:** %10 random, %90 en iyi

#### Senaryo 4: ε = 0.0 (Test Modu)

```python
random = 0.5
epsilon = 0.0
0.5 < 0.0 → FALSE → En iyi aksiyonu seç
```

**Sonuç:** %100 en iyi (hiç random yok)

---

## 📉 EPSILON DECAY (AZALMA) NEDİR?

**Epsilon Decay** = Epsilon değerinin zamanla azalması

### Neden Azalıyor?

1. **Başlangıçta (ε = 1.0)**:
   - Agent hiçbir şey bilmiyor
   - Çok keşif yapmalı (exploration)
   - Ortamı tanımalı

2. **Sonraları (ε → 0.1)**:
   - Agent öğrenmeye başladı
   - Daha çok sömürmeli (exploitation)
   - Öğrendiğini kullanmalı

### İki Yöntem

#### 1. Multiplicative Decay (Basit)

```python
self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
```

**Nasıl Çalışır?**

```python
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.1

Episode 0:  epsilon = 1.0
Episode 1:  epsilon = 1.0 × 0.995 = 0.995
Episode 2:  epsilon = 0.995 × 0.995 = 0.990
Episode 3:  epsilon = 0.990 × 0.995 = 0.985
...
Episode 100: epsilon = 0.605
Episode 200: epsilon = 0.366
Episode 300: epsilon = 0.221
Episode 400: epsilon = 0.134
Episode 500: epsilon = 0.100 (epsilon_min'e ulaştı)
```

**Grafik:**
```
ε
1.0 |████
    |  ████
    |     ████
    |        ████
0.1 |              ████████████████
    └────────────────────────────────→ Episode
    0    100   200   300   400   500
```

**Avantaj:** Basit, hızlı
**Dezavantaj:** Ani düşüşler olabilir

#### 2. Reverse Sigmoid Decay (Senin Kodun)

```python
def reverse_sigmoid_decay(t, epsilon_initial, epsilon_min, k, t0):
    return epsilon_min + (epsilon_initial - epsilon_min) / (1 + np.exp(k * (t - t0)))
```

**Parametreler:**
- `t`: Mevcut episode numarası
- `epsilon_initial`: Başlangıç epsilon (1.0)
- `epsilon_min`: Minimum epsilon (0.1)
- `k`: Decay hızı (ne kadar hızlı azalacak)
- `t0`: Inflection point (en hızlı azalmanın olduğu episode)

**Nasıl Çalışır?**

```python
epsilon_initial = 1.0
epsilon_min = 0.1
k = 0.01
t0 = 25

Episode 0:   epsilon = 1.0 / (1 + exp(0.01 × (0 - 25))) ≈ 0.94
Episode 10:  epsilon = 1.0 / (1 + exp(0.01 × (10 - 25))) ≈ 0.82
Episode 25:  epsilon = 1.0 / (1 + exp(0.01 × (25 - 25))) = 0.5  ← Inflection point!
Episode 40:  epsilon = 1.0 / (1 + exp(0.01 × (40 - 25))) ≈ 0.18
Episode 50:  epsilon = 1.0 / (1 + exp(0.01 × (50 - 25))) ≈ 0.12
Episode 100: epsilon ≈ 0.1 (epsilon_min'e yaklaştı)
```

**Grafik (S Şeklinde):**
```
ε
1.0 |████
    |  ████
    |     ████
    |        ████
    |           ████
0.5 |              ●  ← Inflection point (t0)
    |                 ████
    |                    ████
0.1 |                       ████████████████
    └────────────────────────────────────────→ Episode
    0    10   20   30   40   50   100
```

**Avantaj:** Smooth geçiş, kontrollü azalma
**Dezavantaj:** Biraz daha karmaşık

---

## 🔄 KODDA NASIL KULLANILIYOR?

### Training Sırasında

```python
def train(self, num_episodes=1000):
    self.epsilon = self.epsilon_initial  # Başlangıç: 1.0
    
    for episode in range(num_episodes):
        # Episode boyunca epsilon-greedy kullan
        while not done:
            action = self.choose_action(state, training=True)
            # epsilon-greedy ile aksiyon seçildi
        
        # Episode sonunda epsilon'u azalt
        if self.use_reverse_sigmoid:
            self.epsilon = reverse_sigmoid_decay(
                episode, self.epsilon_initial, self.epsilon_min,
                self.sigmoid_k, self.sigmoid_t0
            )
        else:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
```

### Test Sırasında

```python
def test(self, num_episodes=10):
    for episode in range(num_episodes):
        while not done:
            # training=False → epsilon kullanılmaz!
            action = self.choose_action(state, training=False)
            # Her zaman en iyi aksiyonu seçer (greedy)
```

**Fark:**
- `training=True` → Epsilon-greedy kullan (random olabilir)
- `training=False` → Sadece greedy (her zaman en iyi)

---

## 📊 EPSILON DEĞERLERİNİN ETKİSİ

### Tablo

| Episode | Epsilon | Davranış | Açıklama |
|---------|---------|----------|----------|
| 0-10 | 1.0 → 0.9 | %90-100 random | Çok keşif, ortamı tanı |
| 10-50 | 0.9 → 0.5 | %50-90 random | Hala keşif, öğrenmeye başla |
| 50-100 | 0.5 → 0.2 | %20-50 random | Daha çok sömür, az keşif |
| 100+ | 0.2 → 0.1 | %10-20 random | Çoğunlukla sömür, az keşif |

---

## ❓ SIK SORULAN SORULAR

### SORU 1: "Epsilon neden 1.0'dan başlıyor?"

**CEVAP:**
"Agent başlangıçta hiçbir şey bilmiyor. Q-table tüm sıfırlar. Eğer epsilon düşük olsaydı, agent hep aynı aksiyonu seçerdi (çünkü tüm Q değerleri 0, hepsi eşit). Bu yüzden başta %100 random yapmalı, ortamı keşfetmeli."

### SORU 2: "Epsilon hiç azalmasa ne olur?"

**CEVAP:**
"Agent sürekli random hareketler yapar. Öğrendiği optimal policy'yi kullanamaz. Mesela 1000 episode sonra optimal yolu öğrenmiş olsa bile, hala %ε olasılıkla random yapıyor. Bu yüzden performans düşük kalır, tutarsız olur."

### SORU 3: "Epsilon çok hızlı azalırsa ne olur?"

**CEVAP:**
"Yetersiz exploration olur. Agent ortamı yeterince keşfedemeden exploit etmeye başlar. Local optimum'a takılabilir, optimal policy'yi bulamayabilir."

### SORU 4: "Epsilon çok yavaş azalırsa ne olur?"

**CEVAP:**
"Eğitim çok uzun sürer. Agent çok fazla random yapar, öğrendiğini kullanamaz. Convergence yavaş olur."

### SORU 5: "Reverse sigmoid neden daha iyi?"

**CEVAP:**
"Smooth geçiş sağlıyor. Başta yavaş azalır (çok keşif), ortada hızlı azalır (hızlı öğrenme), sonda tekrar yavaş azalır (stabil). Ani düşüşler yok, daha kontrollü."

---

## 🎯 ÖZET

### Epsilon-Greedy
- **Ne?** Exploration ve exploitation dengesi
- **Nasıl?** ε olasılıkla random, 1-ε olasılıkla en iyi

### Epsilon
- **Ne?** Random aksiyon seçme olasılığı
- **Değer:** 0.0 (hiç random) ile 1.0 (tamamen random) arası

### Epsilon Decay
- **Ne?** Epsilon'un zamanla azalması
- **Neden?** Başta keşif, sonra sömürme
- **Yöntemler:** Multiplicative veya Reverse Sigmoid

### Kod Özeti

```python
# Başlangıç
epsilon = 1.0  # %100 random

# Her episode'da
if random() < epsilon:
    action = random_action()  # Exploration
else:
    action = best_action()    # Exploitation

# Episode sonunda
epsilon = decay(epsilon)  # Azalt
```

---

**BAŞARILAR! 🚀**

