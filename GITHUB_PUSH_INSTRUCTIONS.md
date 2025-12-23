# GitHub'a Push Etme Talimatları

## 1. Git Repository'yi Hazırla

```bash
cd /Users/suleakarsu/Desktop/padm/github_repo

# Git durumunu kontrol et
git status

# Tüm dosyaları ekle
git add .

# İlk commit
git commit -m "Initial commit: Assignment 1 & 2 complete

- Custom Grid World Environment (Assignment 1)
- Q-Learning Agent with 100% success rate (Assignment 2)
- Comprehensive documentation and visualizations
- Demo scripts and examples"

# Remote repository'yi ekle (eğer yoksa)
git remote add origin https://github.com/mftnakrsu/padm-qlearning-agent-rl.git

# Push et
git push -u origin main
```

## 2. Eğer Branch Kullanıyorsan

```bash
# Master branch'e geç
git checkout -b main
git push -u origin main
```

## 3. GitHub'da Kontrol Et

1. https://github.com/mftnakrsu/padm-qlearning-agent-rl adresine git
2. README.md'nin düzgün göründüğünü kontrol et
3. Dosyaların yüklendiğini kontrol et

## 4. Görselleri Eklemek İçin (Opsiyonel)

Görselleri eklemek için:

```bash
# Görsel oluşturma scriptini çalıştır
python generate_demo_images.py

# Görselleri commit et
git add assets/
git commit -m "Add demo images and visualizations"
git push
```

## 5. README'deki Görsel Linklerini Güncelle

README.md'de görsel linkleri şu formatta olmalı:
- `![Description](assets/images/filename.png)`
- GitHub otomatik olarak görselleri gösterecek

## Notlar

- `.gitignore` dosyası gereksiz dosyaları (__pycache__, *.pyc) hariç tutar
- `q_table_final_4d.npy` dosyası zaten commit edilmiş (pre-trained model)
- Görselleri eklemek istersen `generate_demo_images.py` scriptini çalıştır
