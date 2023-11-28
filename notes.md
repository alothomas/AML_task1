# Discussion 26.10.23
1. Splitting task:
- Outlier detection: Andreas
- Feature selection: Khue
- Imputation of missing values: Alois

2. Framework:
- Env: conda
- Pre: 
    - outlier_detection.py
    - feature_seclection.py
    - imputation.py
    - Each method takes and output pd.DataFrame
- Config parsing

3. Baseline evaluation: linear regression?

4. Next meeting:
- Initial implementation of preprocessing
- Config parsing?
- Time: Some time beginning next week (from 30th Oct)

# Discussion 02.11.23
1. Current status and observations:
- Preprocessing (mostly) done, need some more feature selection.
- Current score 0.54 with mice, z-score, lasso and linear.
- SVM doesn't really work.
- Currently using rounded r2 score and rounded predictions.

2. Splitting task:
- Kernelized regression -> Andreas
- Parameter tune for mice -> Aloïs
- Non rounded r2 -> Khue
- Multilabel classification -> Khue
- More feature selections -> Khue
