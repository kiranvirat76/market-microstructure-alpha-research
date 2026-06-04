# Market Microstructure Signal Discovery and Backtesting Framework

Independent Quant Research Project using the IMC Prosperity Trading Challenge Round 1 dataset.

## Overview

This project investigates short-horizon alpha generation from limit order book dynamics and trade-flow behavior. The framework combines feature engineering, predictive validation, execution-cost modelling and risk analysis into a complete quantitative research pipeline.

## Dataset

* IMC Prosperity Trading Challenge – Round 1
* 60,000+ limit order book snapshots
* 2,200+ executed trades
* 2 products across 3 trading days

## Research Pipeline

Raw Order Book Data
→ Feature Engineering
→ Signal Generation
→ Predictive Validation
→ Backtesting
→ Capacity Analysis
→ Alpha Attribution

## Signals

* Order Book Imbalance (OBI)
* Liquidity Shock
* Trade Flow Imbalance
* Lee–Ready Trade Classification

## Validation Framework

* Information Coefficient (IC) Analysis
* Quintile Analysis
* Signal Decay Studies
* Walk-Forward Validation

## Execution & Risk Modelling

* Bid-Ask Spread Costs
* Square-Root Market Impact Model
* Position Limits
* Capacity Analysis
* Drawdown Analysis
* Alpha Attribution

## Technology Stack

Python • Pandas • NumPy • SciPy • Matplotlib

## Project Structure

market-microstructure-alpha-research/

├── data/
├── main.py
├── data_loader.py
├── signals.py
├── metrics.py
├── validation.py
├── backtest.py
├── risk.py
├── capacity.py
├── walk_forward.py
├── attribution.py
└── strategies/

## Key Research Components

* Market Microstructure Analytics
* Order Book Feature Engineering
* Statistical Signal Validation
* Execution Cost Modelling
* Capacity & Robustness Testing
* Quantitative Strategy Research
