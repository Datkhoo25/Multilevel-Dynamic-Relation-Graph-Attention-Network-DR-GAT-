# Multilevel DR-GAT with TCN for Stock Ranking Prediction

## Overview

This repository contains the implementation of the research project "Multilevel DR-GAT with Novel Top-K Ranking Loss for Stock Ranking Prediction" by Lucas Khoo Kar Hoo. The project aims to improve stock ranking prediction by integrating advanced similarity metrics, Temporal Convolutional Networks (TCN), and technical and fundamental indicators into Dynamic Routing Graph Attention Networks (DR-GAT).

## Table of Contents

1. [Introduction](#introduction)
2. [Project Structure](#project-structure)
3. [Data Preprocessing](#data-preprocessing)
4. [Model Checkpoints](#model-checkpoints)


## Introduction

The stock market prediction is a challenging task due to high randomness and noisy data. This project proposes a novel approach using DR-GAT with TCN to capture both temporal and spatial dependencies among stocks. The model aims to improve the prediction of stock rankings by leveraging dynamic adjacency matrices and advanced feature extraction techniques.

## Project Structure

The repository is organized as follows:

- `GAT Data Preprocessing + Sliding Window`: Jupyter notebook for data preprocessing, window slicing, DTW variance calculation and adjacency matrices calculation.
- `Base Model 1_GRU w/o GAT + HP(WS10).ipynb`: Jupyter notebook for GRU model.
- `Base Model 2_TCN Tutorial and Trial (Using most simple TCN + normalization).ipynb`: Jupyter notebook for TCN model.
- `Base_Model_3_(With_Residual)_GRU_w_Fixed_GAT_+_HP(WS10)__(IRR_MRR20).ipynb`: Jupyter notebook for fixed GAT with GRU model.
- `Base_Model_4_PW_GRU_(Residual)w_Dynamic_+_Fixed_GAT_+_HP(WS10)(IRR_MRR_20).ipynb
`: Jupyter notebook for pairwise trend generated dynamic GAT with short and long-term adjacency matrix.
- `Base_Model_4_MASS_GRU_(Residual)w_Dynamic_+_Fixed_GAT_+_HP(WS10)(IRR_MRR_20).ipynb
`: Jupyter notebook for MASS (Mueen's Algorithm for Similarity Search)
 generated dynamic GAT with short and long-term adjacency matrix------> Research Objective 1
- `Base_Model_4_DTW_GRU_(Residual)w_Dynamic_+_Fixed_GAT_+_HP(WS10)(IRR_MRR_20).ipynb
`: Jupyter notebook for Custom Dynamic Time Wrapping generated dynamic GAT with short and long-term adjacency matrix------> Research Objective 1 (Best model)




- `README.md`: This readme file.

## Data Preprocessing

The `GAT Data Preprocessing + Sliding Window` notebook contains the steps for data preprocessing, including:

1. Loading the dataset from NASDAQ (551 companies).
2. Calculating technical indicators such as Moving Average (MA), Relative Strength Index (RSI), Commodity Channel Index (CCI), and Bollinger Bands (BB).
3. Calculating fundamental indicators such as Gross Margin (GM), Earnings Per Share (EPS), Price-to-Earnings (PE), Gross Profit Growth (GPG), Asset-Liability Ratio (ALR), and Leverage (L).
4. Constructing adjacency matrices based on industry relationships, Wiki relationships, and dynamic stock price similarity.

### Additional Data Processing Functions

The repository includes additional data processing functions to handle missing dates, process news sentiment, and compute similarity metrics:

- **add_missing_dates_to_tickers**: Adds rows for missing dates and forward fills NaN values for each ticker.
- **process_news_sentiment**: Processes news sentiment data and normalizes sentiment scores.
- **calculate_dwt_variances_for_price**: Calculates DWT variances for price data to determine volatility.
- **compute_similarity1**: Computes similarity metrics (DTW, pairwise movement) for long-term and short-term data.
 **compute_similarity2**: Computes similarity metrics (MASS similiarity) for long-term and short-term data.

## Model Checkpoints

The `checkpoints/` directory contains the saved model checkpoints for different stages of training. These checkpoints can be used to resume training or for evaluation purposes.
