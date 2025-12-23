# Setup Guide

## Quick Setup

```bash
# 1. Clone repository
git clone https://github.com/mftnakrsu/padm-qlearning-agent-rl.git
cd padm-qlearning-agent-rl

# 2. Install dependencies
pip install -r requirements.txt

# 3. Test Assignment 1
cd assignment1
python demo.py

# 4. Train Assignment 2 (optional - pre-trained model included)
cd ../assignment2
python assignment2_main.py

# 5. Run trained agent
python run_trained_agent.py
```

## Detailed Setup

### Python Version

Requires Python 3.8 or higher.

Check your version:
```bash
python --version
```

### Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Verify Installation

```bash
python -c "import gymnasium; import numpy; import pygame; print('✓ All dependencies installed')"
```

## Troubleshooting

### Pygame Issues

If pygame doesn't work:
```bash
# macOS
brew install sdl2 sdl2_image sdl2_mixer sdl2_ttf

# Linux (Ubuntu/Debian)
sudo apt-get install python3-pygame

# Windows
pip install pygame --upgrade
```

### Import Errors

If you get import errors, make sure you're in the correct directory:
```bash
# For Assignment 1
cd assignment1
python demo.py

# For Assignment 2
cd assignment2
python assignment2_main.py
```

## Next Steps

- Read the main [README.md](README.md) for detailed usage
- Check [assignment1/README.md](assignment1/README.md) for environment details
- Check [assignment2/README.md](assignment2/README.md) for Q-learning details

