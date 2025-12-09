# Troubleshooting Guide

## PyTorch DLL Error on Windows

If you see: `OSError: [WinError 1114] A dynamic link library (DLL) initialization routine failed`

**Good News:** The data collection system will still work! The backend will start and you can collect training data. Only the ML prediction feature won't work until PyTorch is fixed.

### Quick Fixes (try in order):

#### Option 1: Reinstall PyTorch (Recommended)
```bash
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

#### Option 2: Install CPU-only version
```bash
pip uninstall torch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### Option 3: Install Visual C++ Redistributables
Download and install from Microsoft:
https://aka.ms/vs/17/release/vc_redist.x64.exe

#### Option 4: Try a different PyTorch version
```bash
pip uninstall torch
pip install torch==2.0.1
```

#### Option 5: Deactivate and recreate virtual environment
```bash
deactivate
rmdir /s .venv
python -m venv .venv
.venv\Scripts\activate
pip install flask flask-cors torch
```

### Verify It Works
After trying a fix, run:
```bash
python backend/app.py
```

You should see:
```
✓ ML model loaded successfully
Running on http://127.0.0.1:5000
```

Or if model loading fails but data collection works:
```
Warning: Could not load ML model (...). Data collection features will still work.
Running on http://127.0.0.1:5000
```

Both are fine! The second just means ML predictions won't work, but you can still collect training data.

