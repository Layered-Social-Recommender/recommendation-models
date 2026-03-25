# Layered Social Recommendation System (NCF Scoring Layer)

## Phase 1: NCF Implementation

This module contains the Neural Collaborative Filtering (NCF) recommendation model that learns user-item interaction patterns.

### Project Structure
- `data/`: Contains raw and processed Movielens datasets.
- `src/preprocessing/`: Data loading, preprocessing, and negative sampling scripts.
- `src/ncf/`: Core PyTorch dataset and NCF model definitions, alongside training and evaluation logic.
- `outputs/`: Model weights, logs, and inference outputs.
