# 🐄 Cow Behavior Sorting Regressor

This repository contains Python scripts and data for developing a **behavior sorting regression model** for cows.  
The model integrates **daily diet summaries** and **behavioral data** to predict or classify behavioral patterns using machine learning and regression analysis.

---

## 📘 Overview

This project aims to understand and predict cow behaviors based on their diet and daily summaries.  
It processes per-cow data, merges multiple data sources, and evaluates performance using **Leave-One-Cow-Out Cross Validation (LOCO-CV)** for reliable generalization.

**Applications include:**
- Automated behavior recognition  
- Diet–behavior relationship modeling  
- Livestock welfare and productivity analytics  

---

## Repository Structure

| File | Description |
|------|--------------|
| **`raw data.py`** | Loads and cleans cow-wise behavioral and daily summary data. Prepares features for model training. |
| **`check_diets.py`** | Verifies and processes dietary intake data for each cow and day. Ensures alignment and consistency. |
| **`cow_day_diet_summary.csv`** | Contains daily diet summaries (feed intake, nutrient composition, feeding behavior, etc.) per cow. |
| **`cowwise_results.csv`** | Stores model predictions and regression metrics (e.g., predicted vs. true values, R², RMSE). |
| **`NEW.py`** | Main training and evaluation script implementing the behavior sorting regressor. Defines the regression model, trains it, and saves outputs. |
| **`general_for_all_cows_one_out.py`** | Runs **Leave-One-Cow-Out Cross Validation (LOCO-CV)** — testing one cow at a time to evaluate model generalization. |
| **`agg.py`** | Aggregates and summarizes validation results from all cows, computing final metrics (e.g., mean R², RMSE) and generating visual summaries. |
| **`pyvenv.cfg`** | Configuration file for your Python virtual environment (auto-generated). |

---

## Objectives

- Train a regression model for **cow behavior prediction**
- Combine **dietary and behavioral features** as inputs  
- Validate generalization using **LOCO cross-validation**  
- Summarize and visualize performance metrics  

---

## Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/<repo-name>.git
   cd <repo-name>

2. **Create and activate a virtual environment
   ```bash
      python -m venv venv
      source venv/bin/activate     # macOS/Linux
      venv\Scripts\activate        # Windows
