#  March Madness 2026 - ML Prediction Pipeline

[![Train Model](https://github.com/youssoufabdouramane/march-madness-ml-2026/actions/workflows/train.yml/badge.svg)](https://github.com/youssoufabdouramane/march-madness-ml-2026/actions/workflows/train.yml)
[![Data Drift Detection](https://github.com/youssoufabdouramane/march-madness-ml-2026/actions/workflows/data-drift.yml/badge.svg)](https://github.com/youssoufabdouramane/march-madness-ml-2026/actions/workflows/data-drift.yml)

##  Overview

This project implements an **AI Agent pipeline** for predicting NCAA March Madness tournament outcomes using Google's Agent Development Kit (ADK). It features automated **data drift detection** and **CI/CD pipelines** with GitHub Actions.

##  Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download data
python scripts/download_data.py

# Run pipeline
python scripts/run_pipeline.py --all