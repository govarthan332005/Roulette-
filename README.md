# 🎰 Roulette AI Predictor

A modern, mobile-first roulette prediction web app with on-device machine learning.
Works fully offline. Mirrors the look & feel of a polished Android app.

## How to use

1. Open `index.html` in any modern browser (Chrome / Safari / Firefox).
2. Go to the **Train** tab.
   - Upload a CSV / TXT file containing previous rounds (numbers 0–36 separated by commas, spaces or newlines), **or**
   - Paste numbers directly, **or**
   - Add numbers one by one.
3. Tap **Train Model** (or it auto-trains as you add data).
4. Go to the **Predict** tab.
5. Enter the last 10 rounds (minimum 3 required).
6. Tap **PREDICT NEXT** — see top prediction, confidence, color, parity, dozen, range, plus the top-5 most likely numbers.
7. After the real result comes in, tap **Record Actual Result** to log it. The model retrains automatically.
8. Check the **History** tab to monitor accuracy and recent numbers.

## How the AI works

The engine is a hybrid statistical model trained ONLY on your uploaded data:
- Order-1, Order-2, Order-3 **Markov transition tables** (what tends to follow given the last 1 / 2 / 3 numbers)
- **Global frequency prior**
- **Pattern transition tables** for color, parity, dozen, and range (low/high)
- Recency-weighted blending of all signals into one probability distribution over 0–36

The top number is the most probable; the confidence bar reflects how strongly the model leans toward it.

## Features
- Modern dark theme with neon gradients & glassmorphism
- Splash screen, tabbed navigation, animated prediction wheel
- Works great on phones, tablets, **split-screen** (landscape & narrow heights handled)
- Installable as PWA (Add to Home Screen)
- Export / Import full model & history as JSON
- 100% offline, no server calls, your data never leaves the device

## Reality check
Roulette is designed to be a fair random game. No prediction model can guarantee outcomes on a truly random wheel. This tool is educational — use responsibly.
