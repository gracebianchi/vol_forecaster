# Markov Regime-Switching Volatility Forecaster

Forecasting S&P 500 (SPY) realized volatility, and testing a specific question: does modeling the market as switching between distinct volatility regimes improve on standard single-regime approaches? The project builds a full pipeline — from classical econometric baselines through machine learning to a Markov regime-switching framework — and evaluates each stage with rigorous out-of-sample and risk-management tests.

The short answer is nuanced, and the nuance is the result: regime-switching does not improve point-forecast accuracy (simple models are hard to beat), but it provides a richer, more interpretable, and better-calibrated account of how volatility behaves — an early-warning stress index, regime-dependent forecast distributions, and cleaner tail behavior.

## Key Findings

Simple models win on point forecasting. HAR-RV beat EWMA and GARCH out-of-sample (4.90 vs. 7.39 / 8.70 RMSE). No machine-learning model (random forest, gradient boosting) improved on it, and CV-scored stepwise selection independently recovered the exact HAR specification from a 7-feature set.
Regime-switching adds descriptive, not point-forecast, value. A 4-state hidden Markov model (calm / normal / stressed / panic, selected by BIC) recovered an interpretable ordered-ladder regime structure and a continuous early-warning stress index. The regime-switching HAR was statistically indistinguishable from single-regime HAR on point accuracy (Diebold–Mariano p = 0.35), matching the literature's "less clear-cut at the daily horizon" finding.
Regime-dependent uncertainty. Monte Carlo forecasts produce tight, symmetric predictive distributions in calm markets and wide, right-skewed ones in stress — uncertainty a single-regime model cannot express.
Well-calibrated tail risk via Extreme Value Theory. A Generalized Pareto (GPD) tail on standardized residuals yielded well-calibrated 99% and 95% Value-at-Risk for both models (passing Kupiec and Christoffersen tests), where a Gaussian assumption failed badly. The regime model left a thinner residual tail (better conditional-volatility modeling) but under-predicted Expected Shortfall — the flip side of that thin tail.
Methodology

## Stage 1 — Baselines. 

Realized volatility is estimated with the Garman–Klass range estimator. Classical models (EWMA, GARCH(1,1), HAR-RV) and machine-learning models (linear regression, random forest, histogram gradient boosting) are compared on a chronological out-of-sample split (train 2015–2022, test 2023+), scored on RMSE and QLIKE with time-series cross-validation.

## Stage 2 — Regime-Switching Extension.

Regime detection: a Gaussian hidden Markov model identifies latent volatility regimes; the number of states is selected by BIC, and a severity-weighted stress index serves as an early-warning indicator.
Forecasting: a regime-conditional HAR (Markov-switching HAR) lets the forecasting relationship differ by regime, with forecasts blended by the HMM's one-step-ahead regime probabilities.
Probabilistic forecasting: Monte Carlo simulation through the regime chain produces full predictive distributions (fan charts).
Tail-risk evaluation: conditional Extreme Value Theory (McNeil & Frey, 2000) fits a GPD to the tail of standardized residuals; VaR and Expected Shortfall are backtested for calibration.

## Results Summary

Stage	Model	Test RMSE	Notes

Classical	EWMA	7.39	

Classical	GARCH(1,1)	8.70	

Classical	HAR-RV	4.90	Best baseline

ML	Linear / RF / Gradient Boosting	≈ 4.9–5.4	None beat HAR-RV

Regime	MS-HAR	≈ 4.9	DM test p = 0.35 vs. HAR-RV (tie)

Tail risk (99% VaR, EVT-calibrated): both models pass Kupiec and Christoffersen coverage tests; GPD shape parameters ξ = 0.128 (single-regime) vs. −0.061 (regime-switching).

## Repository Structure

vol-forecaster/

├── notebooks/

│   └── vol_forecaster.ipynb     # full analysis, Stage 1 + Stage 2

├── src/

│   ├── data.py                  # SPY download + Garman-Klass realized vol

│   └── features.py              # feature engineering (lags, rolling stats)

├── models/

│   └── hmm_4state.joblib        # fitted 4-state regime detector

├── requirements.txt

└── README.md


## Setup

bash

python -m venv venv && source venv/bin/activate

pip install -r requirements.txt

jupyter notebook notebooks/vol_forecaster.ipynb

## Selected References

Corsi (2009), A Simple Approximate Long-Memory Model of Realized Volatility.

Ding, Kambouroudis & McMillan (2025), Forecasting Realised Volatility Using Regime-Switching Models.

McNeil & Frey (2000), Estimation of Tail-Related Risk Measures for Heteroscedastic Financial Time Series.

Cont (2001), Empirical Properties of Asset Returns: Stylized Facts and Statistical Issues.

Built with Python (pandas, numpy, statsmodels, arch, hmmlearn, scikit-learn, scipy).
