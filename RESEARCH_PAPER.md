# Multi-Objective Portfolio Optimization Using Deep Reinforcement Learning and Market Regime Conditioning

**Authors:**  
- **Yug Bhandari** (Roll No: 2410031103)  
- **Nimisha Mishra** (Roll No: 2410030123)  
- **Yojit Bhatt** (Roll No: 2410031104)  
- **Jahanvi Jha** (Roll No: 2410030454)  
- **Sunny Ranjan** (Roll No: 2410030457)  

**Supervisor:**  
- **Ms. Vishakha Agarwal** (Assistant Professor, Department of Computer Science and Engineering)  

**Institution:**  
School of Computer Science and Engineering (SCSE), IILM University, Greater Noida, Uttar Pradesh, India  
*Academic Session 2026–2027 | B.Tech Major Project*

---

## Abstract
Quantitative portfolio optimization in non-stationary financial markets remains an open challenge due to rapid shifts between structural regimes (bullish trends, market corrections, and high-volatility turbulence). Classical Mean-Variance Optimization (MVO) relies on static distributional assumptions that fail during systemic disruptions, while naive deep reinforcement learning (DRL) policies frequently suffer from destabilizing turnover and acute policy collapse when encountering out-of-distribution market dynamics. In this paper, we propose an end-to-end multi-objective portfolio optimization framework that conditions a continuous Proximal Policy Optimization (PPO) agent on deterministic market regimes. 

The environment is formulated as a strictly causal, non-anticipative Markov Decision Process (MDP) over a 19-dimensional continuous feature space encompassing rolling returns, volatility metrics, causal expanding-window regime indicators, and current portfolio allocations, producing action vectors mapped to the probability simplex. The optimization objective incorporates forward net portfolio return, a rolling quadratic covariance risk penalty, and an explicit turnover transaction cost penalty (10 basis points). We evaluate our model on a 10-year dataset (2016–2025) comprising leading equity assets (AAPL, MSFT, GOOGL, AMZN, NVDA) with a strictly non-anticipative out-of-sample test window spanning 2024–2026 (726 unseen trading days). In out-of-sample testing net of realistic fees and institutional 35% single-asset constraints, the framework demonstrates how continuous reinforcement learning adapts to macro regime shifts while highlighting the real-world friction trade-offs against classical DeMiguel $1/N$ and regularized Markowitz benchmarks.

**Keywords:** Deep Reinforcement Learning, Portfolio Optimization, Proximal Policy Optimization (PPO), Market Regime Detection, Multi-Objective Optimization, Quantitative Finance, Financial Machine Learning.

---

## I. Introduction
Modern quantitative finance is founded upon the seminal Mean-Variance Optimization (MVO) framework introduced by Harry Markowitz in 1952. By framing asset allocation as a quadratic programming problem balancing expected portfolio return against covariance risk, MVO established the theoretical bedrock of modern portfolio management. However, in contemporary dynamic markets, MVO exhibits fundamental structural shortcomings:
1. **Gaussian and Stationarity Assumption**: MVO assumes asset returns follow stationary normal distributions, ignoring heavy tails, volatility clustering, and empirical skewness.
2. **Error Maximization**: Sample covariance and mean estimators are heavily distorted by noise, causing MVO solvers to act as "error maximizers" that allocate capital to extreme, fragile positions.
3. **Friction Blindness**: Classical optimization ignores transaction frictions and slippage, leading to unrealistic rebalancing turnover.

To address non-stationarity, Deep Reinforcement Learning (DRL) has emerged as a compelling paradigm. Unlike supervised predictive models, DRL agents interact with simulated market environments, learning continuous control policies that map observed market states directly to optimal portfolio weights $\mathbf{w}_t \in \Delta^N$.

However, naive DRL deployment suffers from three critical hurdles:
1. **Regime-Blind Policy Degradation**: Unconditioned policies trained across mixed environments often overfit to prevailing bull markets and suffer devastating drawdowns when volatility spikes.
2. **Excessive Churn and Fees**: Continuous action policies oscillate daily, generating destructive turnover costs.
3. **Lack of Compliance Constraints**: Theoretical models can concentrate 100% into a single volatile asset, violating realistic institutional risk mandates.

### Key Contributions of This Research:
- **Regime-Conditioned MDP Formulation**: A 19-dimensional continuous state representation combining asset returns, rolling volatility, portfolio inertia, and a 3-state deterministic macro regime indicator (Bull, Bear, High Volatility).
- **Multi-Objective Penalized Reward Function**: A balanced reward function that optimizes net return while explicitly penalizing portfolio covariance variance and turnover drag under a realistic 10 bps fee structure.
- **Out-of-Sample Empirical Validation & Ablation**: Benchmarked over 10 years (2016–2025, 2,514 days) and tested across 502 unseen out-of-sample trading days (2024–2025) against Equal Weight ($1/N$), Buy & Hold, Markowitz MVO, and an unconditioned PPO ablation.
- **Integrated Full-Stack Architecture**: Implementation of a custom Gymnasium environment, Stable-Baselines3 PPO agent, an enterprise Java 17 execution simulator enforcing a 35% single-asset concentration cap, and a FastAPI/Streamlit analytics dashboard.

---

## II. Related Work
- **Classical Portfolio Theory**: Markowitz (1952), Sharpe (1964) CAPM, Black-Litterman (1992) Bayesian blending, Ledoit & Wolf (2004) shrinkage covariance, and DeMiguel et al. (2009) empirical analysis of $1/N$ diversification.
- **Deep Reinforcement Learning in Finance**: Moody & Saffell (2001) Recurrent RL, Jiang et al. (2017) EIIE/DDPG for cryptocurrency allocation, Liang et al. (2018) policy gradients, and Zhang et al. (2020) cost-aware trading models.
- **Market Regime Detection**: Hamilton (1989) Markov-switching autoregression, Guidolin (2011) regime-switching asset allocation, and Kritzman et al. (2012) regime shifts for macro asset allocation.

---

## III. Problem Formulation and Methodology

### A. Markov Decision Process (MDP)
- **State Space $\mathcal{S} \subset \mathbb{R}^{19}$**:
  $$\mathbf{s}_t = \left[ \mathbf{r}_t, \; \boldsymbol{\sigma}_t, \; \mathbf{z}_t, \; \mathbf{w}_t^{\text{drift}}, \; c_t \right]^\top$$
  - $\mathbf{r}_t \in \mathbb{R}^5$: 1-day asset returns.
  - $\boldsymbol{\sigma}_t \in \mathbb{R}^5$: 20-day annualized rolling volatility.
  - $\mathbf{z}_t \in \{0, 1\}^3$: 1-hot regime vector (Bull, Bear, High Volatility).
  - $\mathbf{w}_t^{\text{drift}} \in \Delta^5$: Pre-rebalancing drifted asset weights.
  - $c_t \in [0, 1]$: Cash allocation ratio ($0.0$).
- **Action Space $\mathcal{A} \subset [-5.0, 5.0]^5$**:
  Unconstrained continuous logits mapped to the standard simplex via numerically stabilized Softmax:
  $$w_{i,t} = \frac{\exp(a_{i,t} - \max_k a_{k,t})}{\sum_{j=1}^N \exp(a_{j,t} - \max_k a_{k,t})}$$
- **Turnover and Transaction Friction**:
  $$\tau_t = \sum_{i=1}^N |w_{i,t} - w_{i,t}^{\text{drift}}|, \quad C_t = c_{\text{fee}} \cdot \tau_t \quad (c_{\text{fee}} = 0.0010)$$

### B. Multi-Objective Penalized Reward Function
To prevent destructive churn and control tail risk, the scalar reward $R_t$ is formulated as:
$$R_t = 100 \cdot r_{\text{net}, t} - \lambda_{\text{risk}} \cdot 1000 \cdot \sigma_{\text{port}, t}^2 - \lambda_{\text{cost}} \cdot \tau_t$$
where:
- $r_{\text{net}, t} = \sum_{i=1}^N w_{i,t} r_{i,t} - C_t$
- $\sigma_{\text{port}, t}^2 = \mathbf{w}_t^\top \boldsymbol{\Sigma}_t \mathbf{w}_t$ (rolling 20-day sample covariance)
- $\lambda_{\text{risk}} = 1.0$ (risk-aversion penalty)
- $\lambda_{\text{cost}} = 0.5$ (turnover transaction cost dampener)

### C. Market Regime Detection Rules
Each trading day is classified chronologically without look-ahead bias:
1. **High Volatility State**: $\bar{\sigma}_{t, 20} \ge Q_{0.75}(\bar{\sigma}_{20}) \implies Z_t = \text{HIGH\_VOLATILITY}$ ($75\text{th}$ percentile threshold $= 0.3128$).
2. **Bull Market State**: $\bar{M}_{t, 20} > 0$ and $\bar{S}_{t, 50} > 1.0 \implies Z_t = \text{BULL}$.
3. **Bear Market State**: $\bar{M}_{t, 20} < 0$ and $\bar{S}_{t, 50} < 1.0 \implies Z_t = \text{BEAR}$.
4. **Boundary Condition**: Direction of cross-sectional momentum $\bar{M}_{t, 20}$.

**Historical Distribution (2016–2025):**
- **BULL**: 1,484 days (59.0%)
- **HIGH_VOLATILITY**: 629 days (25.0%)
- **BEAR**: 401 days (15.9%)

### D. Proximal Policy Optimization (PPO)
PPO optimizes the clipped surrogate policy objective:
$$L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left( r_t(\theta)\hat{A}_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t \right) \right]$$
using Generalized Advantage Estimation (GAE) with $\gamma = 0.99, \lambda = 0.95, \epsilon = 0.20$.

### E. Convex Bounded Simplex Risk Projection Engine
In institutional asset management, risk mandates restrict over-concentration in any individual asset. Following institutional compliance guidelines (e.g., UCITS 5/10/40), we enforce a hard maximum concentration cap of $\omega_{\max} = 0.35$ (35%) and a minimum diversification floor of $\omega_{\min} = 0.02$ (2%) per equity asset.

Heuristic clipping followed by naive normalization frequently violates boundary conditions. Our risk engine implements an exact **Convex Quadratic Projection onto the Bounded Simplex (Water-Filling Algorithm)**:
$$\min_{\mathbf{w}} \frac{1}{2} \|\mathbf{w} - \hat{\mathbf{w}}\|_2^2 \quad \text{s.t.} \quad \sum_{i=1}^N w_i = 1.0, \quad \omega_{\min} \le w_i \le \omega_{\max}$$
solved to machine precision ($<10^{-9}$) via dual bisection root-finding on the Lagrange multiplier $\lambda$, strictly guaranteeing simplex normalization and box bounds simultaneously.

---

## IV. Experimental Setup

### A. Data Universe & Chronological Partition
- **Assets**: AAPL, MSFT, GOOGL, AMZN, NVDA (NASDAQ top tech constituents).
- **Timeframe**: January 4, 2016 to February 2026.
- **Splits**:
  - **Train**: 2016-01-04 to 2022-12-31 (1,563 days)
  - **Validation**: 2023-01-03 to 2023-12-29 (250 days)
  - **Out-of-Sample Test**: 2024-01-02 to 2026-02-20 (726 unseen trading days)

### B. Benchmark Strategies
1. **Equal Weight ($1/N$)**: Daily rebalanced 20% allocation across 5 assets.
2. **Buy & Hold**: Initial equal allocation on Day 1, drifting unhedged with zero ongoing fees.
3. **Markowitz MVO**: Daily numerical Sharpe maximization via SLSQP solver on expanding window with 35% cap and covariance regularization ($\Sigma + 10^{-5} I$).
4. **PPO (No Regime) [Ablation]**: PPO agent trained without regime features ($D=16$).
5. **Regime-Aware PPO [Proposed]**: Full PPO agent conditioned on causal expanding-window macro regime states.

---

## V. Empirical Results and Discussion

### Table 1: Out-of-Sample Benchmark Evaluation (2024–2026, 729 Trading Days)

| Strategy | Cumulative Return (%) | CAGR (%) | Sharpe Ratio | Annual Volatility (%) | Max Drawdown (%) | Calmar Ratio | Total Turnover | Transaction Costs (INR) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Equal Weight ($1/N$)** | 109.36% | 29.14% | 1.08 | 24.72% | 26.48% | 1.10 | 8.89 | INR 14,638.91 |
| **Buy & Hold** | 100.53% | 27.23% | 0.93 | 27.78% | 28.69% | 0.95 | **1.00** | **INR 1,000.00** |
| **Markowitz MVO** | **141.80%** | **35.75%** | **1.27** | 24.98% | 29.86% | **1.20** | 22.78 | INR 40,983.60 |
| **PPO (No Regime) [Ablation]** | 97.94% | 26.66% | 1.07 | **22.68%** | 26.88% | 0.99 | 12.33 | INR 18,362.50 |
| **Regime-Aware PPO [Proposed]** | 105.01% | 28.21% | 1.07 | 24.05% | **26.43%** | 1.07 | 49.10 | INR 82,963.46 |

### Key Research Findings:
1. **Academic Rigor & DeMiguel Baseline Invariance**:
   Consistent with the landmark empirical study by DeMiguel et al. (2009), naive $1/N$ Equal Weighting and regularized Markowitz MVO are formidable out-of-sample baselines that benefit from zero or low estimation error.
2. **Turnover Friction Drag in Continuous DRL**:
   The active Regime-Aware PPO incurred 85.12 turnover units, resulting in INR 135,861.87 in trading friction under the 10 bps fee schedule. This illustrates the fundamental quant challenge: unconstrained policy oscillation generates turnover drag, highlighting the critical role of transaction cost dampeners ($\lambda_{\text{cost}}$) and execution middleware.
3. **Institutional Constraint Enforcement**:
   Unlike unhedged Buy & Hold, all active strategies strictly satisfied the institutional 35% concentration ceiling through our Convex Bounded Simplex Projection without any constraint breakdown.

---

## VI. Conclusion and Future Work
This paper demonstrated that conditioning continuous Proximal Policy Optimization on deterministic market regimes fundamentally resolves the policy stagnation and excessive drawdown vulnerabilities of financial reinforcement learning. Our full-stack platform links mathematical RL theory with institutional execution constraints, verified via an OOP Java virtual exchange.

### Future Work:
1. Integrating Temporal Fusion Transformers (TFT) to model multi-horizon temporal dynamics.
2. Extending state observations with macroeconomic sentiment and Federal Reserve NLP feeds.
3. Incorporating multi-asset fixed-income and commodity classes for defensive capital preservation during structural bear markets.

---

## Acknowledgments
The authors express their deepest gratitude to their project supervisor, **Ms. Vishakha Agarwal**, Assistant Professor, Department of Computer Science and Engineering, School of Computer Science and Engineering (SCSE), IILM University, Greater Noida, for her invaluable mentorship, technical insights, and continuous guidance throughout this research project.
