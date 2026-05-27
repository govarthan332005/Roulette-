# 🎯 Roulette AI Predictor — Enhanced Edition

A modern, mobile-first roulette prediction web app with on-device AI.

## ✨ What's New in This Version

### 🎨 White & Blue Theme
Clean, professional Material-inspired design. Looks advanced but not arrogant.
Soft blues, crisp whites, gentle shadows — easy on the eyes.

### ⚡ Lightning-Fast Number Entry
- **Tap-based keypad** for predictions — tap a slot, tap a number, auto-advance to next
- **Mini keypad** in Training tab — every tap instantly adds a number to your dataset
- Color-coded keys (red / black / green) matching the roulette wheel
- Backspace and clear actions for fast corrections
- No typing required — perfect for live use

### 🔊 Sound Effects (Zero Latency)
Generated in real-time via Web Audio API — no audio files to download, no lag:
- Tap clicks for keys
- Tab-switch sweeps
- Success chime on save
- Predict sequence (ascending tones)
- Error buzz for invalid input
- Triumph fanfare on exact prediction hit
- Toggle via the 🔊 button in the top bar

### 📳 Haptic Feedback
Subtle vibrations on supported devices for a tactile, app-like feel.

### 🚀 Performance Optimized
- DocumentFragment-based rendering for heatmap
- Debounced model rebuilds (350ms) — no UI freeze when bulk-adding numbers
- `requestAnimationFrame` for smooth animations
- Passive event listeners where possible
- Deferred heavy operations after toast feedback

### 📱 Android-App Feel
- Fixed viewport — **no pinch zoom, no double-tap zoom**
- Material Design ripple effects on every button
- Safe-area-inset support for notched devices
- Responsive from 320px to 600px+
- Smooth tab transitions
- Sticky top bar with gradient
- Bottom-safe-area handling
- `touch-action: manipulation` for instant tap response

## 📦 Files
- `index.html` — UI skeleton
- `style.css` — full white & blue theme
- `app.js` — prediction engine, keypad, sound, UI logic
- `manifest.json` — PWA installable

## 🚀 Usage
Just open `index.html` in any modern mobile browser (Chrome / Safari / Edge).
Add to home screen for the full standalone app experience.

## 🧠 Prediction Engine
Hybrid Markov chain (orders 1, 2, 3) + frequency prior + pattern features (color, parity, dozen, range) — all running on-device. No data leaves your phone.

## ⚠️ Disclaimer
This is a probabilistic tool that learns from your training data. No model can guarantee outcomes in a fair random game. Use responsibly.
