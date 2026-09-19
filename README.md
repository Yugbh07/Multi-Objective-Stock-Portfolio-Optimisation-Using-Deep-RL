# Multi-Objective Portfolio Optimization Using Reinforcement Learning

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.127-teal.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.45-FF4B4B.svg)](https://streamlit.io/)
[![Stable--Baselines3](https://img.shields.io/badge/SB3-PPO-orange.svg)](https://stable-baselines3.readthedocs.io/)
[![Java](https://img.shields.io/badge/Java-17-red.svg)](https://openjdk.org/)

An academic and production-ready implementation of **Multi-Objective Portfolio Optimization** using Deep Reinforcement Learning (PPO), classical quantitative finance benchmarks (Equal Weight, Buy & Hold, Markowitz Mean-Variance Optimization), a custom Gymnasium market environment, a realistic virtual trading engine in Java, and an interactive Streamlit analytics dashboard.

---

## 🏛️ Project Information
- **Institution**: IILM University, Greater Noida
- **Degree**: Bachelor of Technology in Computer Science & Engineering
- **Session**: 2026–27
- **Team**:
  - Yug Bhandari (Roll No: 2410031103)
  - Nimisha Mishra (Roll No: 2410030123)
  - Yojit Bhatt (Roll No: 2410031104)
  - Jahanvi Jha (Roll No: 2410030454)
  - Sunny Ranjan (Roll No: 2410030457)
- **Faculty Guide**: Ms. Vishakha Agarwal (Assistant Professor, SCSE)

---

## 📊 Complete Workflow Architecture

```
                                USER / INVESTOR
                                       │
                         Capital, Risk Profile, Assets
                                       ▼
                              ┌─────────────────┐
                              │    Streamlit    │  (Interactive UI)
                              │    Dashboard    │
                              └────────┬────────┘
                                       │ REST API
                                       ▼
                              ┌─────────────────┐
                              │  FastAPI Server │  (Orchestrator & DB)
                              └────────┬────────┘
                                       │
                                       ▼
                              ┌─────────────────┐
                              │  Data Pipeline  │  (yfinance + Data Cleaner)
                              └────────┬────────┘
                                       │
                                       ▼
                              ┌─────────────────┐
                              │ Feature Engine  │  (Return, Volatility,
                              │ & Regime Detect │   Covariance, Bull/Bear)
                              └────────┬────────┘
                                       │
                   ┌───────────────────┴───────────────────┐
                   ▼                                       ▼
       ┌────────────────────────┐              ┌────────────────────────┐
       │ Traditional Baselines  │              │ Gymnasium Environment  │
       │ Equal Weight, Buy&Hold,│              │ PortfolioEnv + Softmax │
       │ Markowitz MVO          │              └───────────┬────────────┘
       └───────────┬────────────┘                          │
                   │                                       ▼
                   │                           ┌────────────────────────┐
                   │                           │    Trained PPO Agent   │
                   │                           └───────────┬────────────┘
                   │                                       │
                   │                                       ▼
                   │                           ┌────────────────────────┐
                   │                           │  Risk & Bounds Engine  │
                   │                           └───────────┬────────────┘
                   │                                       │
                   └───────────────────┬───────────────────┘
                                       ▼
                              ┌─────────────────┐
                              │  Backtesting &  │  (Transaction Costs,
                              │ Java Simulator  │   Virtual Trade Exec)
                              └────────┬────────┘
                                       │
                                       ▼
                              ┌─────────────────┐
                              │ Metrics Engine  │  (CAGR, Sharpe, MDD,
                              │ & Comparison    │   Volatility, Turnover)
                              └────────┬────────┘
                                       │
                                       ▼
                              Streamlit Analytics
```

---

## 🚀 Quickstart Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Data Pipeline & Train PPO
```bash
python src/data/collector.py
python src/data/cleaner.py
python src/features/engineer.py
python src/models/train_ppo.py
```

### 3. Run Experiments & Evaluation
```bash
python src/evaluation/generate_research_tables.py
```

### 4. Run Java Simulator & Backend
```bash
# Terminal 1: Compile & Start Java Trading Simulator
cd simulator/java/src
javac *.java
java SimulatorServer

# Terminal 2: Run FastAPI Backend
uvicorn backend.main:app --reload --port 8000

# Terminal 3: Launch Streamlit Dashboard
streamlit run dashboard/app.py
```
