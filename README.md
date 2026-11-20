# Breast Cancer Data Exploration & Preprocessing (`B.Cancer.ipynb`)

A Jupyter notebook that loads the UCI Breast Cancer Wisconsin (Diagnostic) dataset, explores it, and prepares it for modeling. The notebook covers **data loading, cleaning, visualization, encoding, scaling, and outlier handling**. It does not train a model yet.

## Dataset

- **File expected:** `uci b.cancer.csv` (place it in the same folder as the notebook)
- **Size:** 569 rows, 33 columns in the raw file
- **Target:** `diagnosis` (`M` = malignant, `B` = benign)
- **Features:** 30 numeric measurements of cell nuclei, in three groups: `*_mean`, `*_se`, and `*_worst`
  (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension)
- **Columns dropped:** `id` (identifier) and `Unnamed: 32` (entirely empty)

## What the notebook does

1. **Imports and loading**: pandas, numpy, matplotlib, seaborn, and scikit-learn preprocessing tools.
2. **Inspection**: `head()`, missing-value counts (only `Unnamed: 32` is empty), column list, and duplicate check (0 duplicates).
3. **Cleaning**: drops `Unnamed: 32` and `id`, keeping a copy of the original (`df1`).
4. **Visualization (before preprocessing)**
   - Bar chart of the diagnosis class distribution
   - Correlation heatmap of the 30 features
5. **Preprocessing**
   - `LabelEncoder` on `diagnosis` (`B` -> 0, `M` -> 1)
   - `StandardScaler` on all 30 feature columns
6. **Outlier removal**: an IQR filter (k = 1.5) over every feature, followed by count plots, box plots, and histograms of the "cleaned" data.

## Requirements

```
python >= 3.9
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
```

Install with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

## How to run

1. Put `uci b.cancer.csv` next to `B.Cancer.ipynb`.
2. Launch Jupyter: `jupyter notebook B.Cancer.ipynb`
3. Run the cells from top to bottom (**Kernel -> Restart & Run All**).

## Known issues

- **The outlier-removal step removes every row.** In the saved output, `Before: (569, 32)` becomes `After: (0, 32)`, so the "after outlier removal" plots are empty.
  - **Cause:** the processed dataframe (`df_proc`) still contains the all-NaN `Unnamed: 32` column. `Series.between()` is `False` for NaN, so every row fails the mask.
  - **Fix:** drop `Unnamed: 32` first, or build `df_proc` from `df1`, which is already clean. Also note that a 1.5 x IQR rule across all 30 features can still remove a large share of rows, so consider a looser `k` or per-feature capping.
- **Cell execution order is inconsistent.** Cells were run out of order (for example `execution_count` 31, 14, 3, ...), and the `df` / `df1` / `df_proc` variables overlap. Use **Restart & Run All** before trusting any output.
- **Scaling is done before splitting the data.** When you add a model, split into train and test first and fit the scaler on the training set only, to avoid data leakage.

## Suggested next steps

- Fix the outlier filter (see above) and compare results with and without it
- Add a train/test split and fit the scaler inside a `Pipeline`
- Train baseline classifiers (logistic regression, random forest, SVM) and evaluate with accuracy, recall, precision, and ROC-AUC. For cancer detection, recall on the malignant class matters most
- Consider dropping highly correlated features (for example radius/perimeter/area) or applying PCA

## Source

Dataset: [UCI Machine Learning Repository, Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
