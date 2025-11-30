# Data Mining Project - Flood Risk Prediction in Jabodetabek

## Project Overview
Proyek prediksi risiko banjir di wilayah Jabodetabek menggunakan data geografis, klimat, dan demografis.

## Workflow & Execution Order

### 1. main.ipynb
**Deskripsi:** Ekstraksi data demografis dari file GeoJSON

**Input: https://www.kaggle.com/datasets/afiskandr/jumlah-penduduk-per-kecamatan-di-indonesia-2023**
- `kecamatan.geojson` (7,287 rows, multiple columns)

**Output:**
- `kecamatan_filtered.csv` (7,287 rows, 21 columns)

**Kolom Output:**
- objectid, kode_kec_spatial, nama_prop, nama_kab, nama_kec
- jumlah_penduduk, kepadatan_penduduk, jumlah_kk, luas_wilayah
- pertumbuhan_2022, pertumbuhan_2021, pertumbuhan_2020
- perpindahan_pddk, jumlah_desa, jumlah_kelurahan
- perdagangan, nelayan, wiraswasta
- tidak_blm_sekolah, slta, s1

---

### 2. join_table.ipynb
**Deskripsi:** Join data banjir dengan data kecamatan menggunakan normalisasi nama

**Input:**
- `data_banjir_combine_final.csv` (18,048 rows)
- `kecamatan_filtered.csv` (7,287 rows, 21 columns)

**Output:**
- `data_banjir_joined.csv` (18,048 rows, 38 columns)
- `data_banjir_joined_clean.csv` (18,048 rows, 38 columns dengan rename)

**Proses:**
- Normalisasi nama kecamatan (lowercase, remove special chars, unicode normalization)
- Left join berdasarkan kecamatan_key
- Deduplikasi data kecamatan
- Rename kolom: NAME_2 → kabupaten_kota, NAME_3 → kecamatan
- Remove kolom objectid

**Join Success Rate:** ~88.2% (15,914 matched, 2,134 unmatched)

---

### 3. remove_header.ipynb
**Deskripsi:** Filter kolom relevan dan hapus data yang tidak ter-join

**Input:**
- `data_banjir_joined_clean.csv` (18,048 rows, 38 columns)

**Output:**
- `data_banjir_filtered.csv` (15,914 rows, 16 columns)

**Kolom Output:**
- kabupaten_kota, kecamatan
- avg_rainfall, max_rainfall, avg_temperature
- elevation, landcover_class, ndvi, slope, soil_moisture
- year, month
- banjir (target variable)
- lat, long, jumlah_penduduk

**Data Cleaning:**
- Removed 2,134 rows tanpa jumlah_penduduk (unmatched data)
- Retention rate: 88.2%

---

### 4. eda.ipynb
**Deskripsi:** Exploratory Data Analysis lengkap

**Input:**
- `data_banjir_filtered.csv` (15,914 rows, 16 columns)

**Output:**
- Visualisasi dan analisis statistik (tidak menghasilkan file CSV)

**Analisis:**
- Data quality check (missing values, duplicates)
- Descriptive statistics
- Target variable distribution
- Univariate analysis (numerical & categorical)
- Bivariate analysis (features vs target)
- Correlation analysis
- Geospatial analysis
- Statistical comparison (t-tests)

---

### 5. fixing_dataset.ipynb
**Deskripsi:** Data preprocessing dan feature engineering berdasarkan temuan EDA

**Input:**
- `data_banjir_filtered.csv` (15,914 rows, 16 columns)

**Output:**
- `data_banjir_processed.csv` (15,914 rows, 29 columns - scaled)
- `data_banjir_engineered.csv` (15,914 rows, 22 columns - unscaled)

**Proses:**
1. **Outlier Handling:** IQR clipping method untuk semua fitur numerik
2. **Feature Engineering:** 6 fitur baru
   - `rainfall_intensity`: max_rainfall / avg_rainfall
   - `is_rainy_season`: binary flag untuk bulan hujan (Nov-Mar)
   - `elevation_slope_ratio`: elevation / slope
   - `vegetation_moisture`: ndvi * soil_moisture
   - `population_density_proxy`: jumlah_penduduk / elevation
   - `extreme_rainfall`: binary flag untuk curah hujan ekstrem (Q3)
3. **Categorical Encoding:**
   - Label encoding untuk landcover_class
   - One-hot encoding untuk landcover features
4. **Feature Scaling:** StandardScaler untuk fitur numerik

**Output Details:**
- `data_banjir_processed.csv`: Dataset lengkap dengan scaling (untuk distance-based ML)
- `data_banjir_engineered.csv`: Dataset dengan fitur baru tanpa scaling (untuk tree-based ML)

---

### 6. DecisionTree.ipynb
**Deskripsi:** Model Decision Tree untuk prediksi banjir dengan class weighting

**Input:**
- `data_banjir_engineered.csv` (15,914 rows, 22 columns)

**Output:**
- `decision_tree_rules.txt` (aturan keputusan dalam format teks)
- `decision_tree_visualization.png` (visualisasi tree)
- Model performance metrics dan visualisasi

**Features Used (16 total):**
- **Numerical (12):** avg_rainfall, max_rainfall, avg_temperature, elevation, ndvi, slope, soil_moisture, jumlah_penduduk, rainfall_intensity, elevation_slope_ratio, vegetation_moisture, population_density_proxy
- **Categorical (4):** month, is_rainy_season, extreme_rainfall, landcover_encoded

**Workflow:**
1. Load & encode landcover_class
2. Exploratory visualization (scatter plots)
3. Train-test split (80/20, stratified)
4. Baseline Decision Tree training (class_weight='balanced')
5. Model evaluation (confusion matrix, classification report, ROC curve)
6. Entropy & Gini index analysis
7. Tree structure visualization
8. Feature importance ranking
9. Cost complexity pruning analysis
10. Optimized (pruned) tree training
11. GridSearchCV hyperparameter tuning
12. Decision rules extraction
13. Best model visualization
14. Final performance summary

**Model Configuration:**
- Algorithm: DecisionTreeClassifier
- Class weight: balanced (algorithmic approach to handle imbalance)
- Hyperparameter tuning: GridSearchCV with 5-fold CV (scoring='accuracy')
- Evaluation metrics: Accuracy, Precision, Recall, F1-Score, AUC-ROC

**Best Model Results:**
- Test Accuracy: 0.8711 (87.11%)
- Test Precision: 0.3184 (31.84%)
- Test Recall: 0.4359 (43.59%)
- Test F1-Score: 0.3678
- Test AUC-ROC: 0.6817

**Best Hyperparameters:**
- criterion: gini
- max_depth: 20
- min_samples_split: 10
- min_samples_leaf: 1

**Top 3 Important Features:**
1. month (18.24%)
2. avg_rainfall (13.60%)
3. rainfall_intensity (11.22%)

---

### 7. SMOTE_baru.ipynb
**Deskripsi:** Model Decision Tree dengan SMOTE untuk menangani class imbalance

**Input:**
- `data_banjir_engineered.csv` (15,914 rows, 22 columns)

**Output:**
- `data_train_smote.csv` (16,216 rows - balanced training data)
- `data_test_original.csv` (2,196 rows - original test data)
- `decision_tree_smote_rules.txt` (aturan keputusan)
- `decision_tree_smote_visualization.png` (visualisasi tree)
- Model performance metrics dan visualisasi

**Key Approach:**
- **SMOTE Implementation:** Synthetic Minority Over-sampling Technique
- **Critical Feature:** SMOTE applied ONLY on training data (no data leakage)
- **Test Data:** Remains unchanged with original distribution

**Workflow:**
1. Load dataset & encode landcover_class
2. Data preparation (features selection)
3. Train-test split (80/20, stratified) - BEFORE SMOTE
4. Apply SMOTE on training data only
5. Visualize SMOTE effect
6. Verify data integrity (no leakage)
7. Save SMOTE data for modeling
8. Train baseline Decision Tree
9. Model evaluation (confusion matrix, ROC curve)
10. Feature importance analysis
11. GridSearchCV hyperparameter tuning (scoring='f1')
12. Compare baseline vs tuned models
13. Best model visualization
14. Extract decision rules
15. Final summary

**Model Configuration:**
- Algorithm: DecisionTreeClassifier
- Class balance: SMOTE oversampling (data-level approach)
- Training data: 8,784 → 16,216 samples (balanced 1:1)
- Test data: 2,196 samples (unchanged, original distribution)
- Hyperparameter tuning: GridSearchCV with 5-fold CV (scoring='f1')
- Evaluation metrics: Accuracy, Precision, Recall, F1-Score, AUC-ROC

**Best Model Results:**
- Test Accuracy: 0.8711 (87.11%)
- Test Precision: 0.2889 (28.89%)
- Test Recall: 0.4615 (46.15%)
- Test F1-Score: 0.3554
- Test AUC-ROC: 0.6834

**Best Hyperparameters:**
- criterion: entropy
- max_depth: None
- min_samples_split: 2
- min_samples_leaf: 1

**Top 3 Important Features:**
1. avg_rainfall (16.67%)
2. month (12.86%)
3. max_rainfall (10.56%)

**SMOTE Statistics:**
- Original training distribution: 7,432 (class 0) vs 1,352 (class 1)
- After SMOTE: 8,108 (class 0) vs 8,108 (class 1) - perfectly balanced
- Synthetic samples generated: 6,756 new samples
- Test data: Unchanged (1,914 class 0, 282 class 1)

---

### 8. WithoutMonth.ipynb
**Deskripsi:** Eksperimen Decision Tree tanpa fitur 'month' untuk validasi feature importance

**Input:**
- `data_banjir_engineered.csv` (15,914 rows, 22 columns)

**Status:** Created, ready for execution

**Purpose:** 
- Test model performance tanpa fitur 'month' (top feature: 18.24%)
- Validasi apakah 'is_rainy_season' dapat menggantikan 'month'
- Features used: 15 (month removed from categorical_features)

**Expected Outcome:**
- Likely performance decrease jika month memang critical
- Empirical validation of feature importance findings

---

## Data Flow Diagram

```
kecamatan.geojson (7,287 rows)
         |
         v
    [main.ipynb]
         |
         v
kecamatan_filtered.csv (7,287 rows, 21 cols)
         |
         +---------------------------+
         |                           |
         v                           v
data_banjir_combine_final.csv   kecamatan_filtered.csv
    (18,048 rows)                (7,287 rows)
         |                           |
         +----------[join_table.ipynb]-----------+
                          |
                          v
            data_banjir_joined_clean.csv
                 (18,048 rows, 38 cols)
                          |
                          v
              [remove_header.ipynb]
                          |
                          v
            data_banjir_filtered.csv
                (15,914 rows, 16 cols)
                          |
                          +------------------+
                          |                  |
                          v                  v
                   [eda.ipynb]      [fixing_dataset.ipynb]
                          |                  |
                          v                  v
              Analysis & Viz       data_banjir_engineered.csv
                                        (15,914 rows, 22 cols)
                                              |
                                              +----------------------------+
                                              |                            |
                                              v                            v
                                     [DecisionTree.ipynb]        [SMOTE_baru.ipynb]
                                    (class_weight='balanced')    (SMOTE oversampling)
                                              |                            |
                                              v                            v
                                  Decision Tree Model              SMOTE Model
                                  - decision_tree_rules.txt        - data_train_smote.csv
                                  - decision_tree_visualization.png - data_test_original.csv
                                  - F1: 0.3678, AUC: 0.6817        - decision_tree_smote_rules.txt
                                                                    - decision_tree_smote_visualization.png
                                                                    - F1: 0.3554, AUC: 0.6834
```

## Dataset Summary

### Final Dataset Statistics
- Total Records: 15,914
- Total Features: 16
- Target Variable: banjir (binary: 0/1)
- Missing Values: 0
- Duplicate Rows: 0
- Class Balance: ~50-50 (relatively balanced)

### Feature Categories
- **Geographic (2):** kabupaten_kota, kecamatan
- **Climate (3):** avg_rainfall, max_rainfall, avg_temperature
- **Environmental (5):** elevation, landcover_class, ndvi, slope, soil_moisture
- **Temporal (2):** year, month
- **Demographic (1):** jumlah_penduduk
- **Coordinates (2):** lat, long
- **Target (1):** banjir

## Requirements

### Python Libraries
```
pandas
numpy
geopandas
matplotlib
seaborn
scipy
scikit-learn
imbalanced-learn  # For SMOTE
re
unicodedata
warnings
```

### Installation
```bash
pip install pandas numpy geopandas matplotlib seaborn scipy scikit-learn imbalanced-learn
```

## Usage Instructions

### Execution Order
1. Run `main.ipynb` - Extract demographic data
2. Run `join_table.ipynb` - Join flood data with demographic data
3. Run `remove_header.ipynb` - Filter columns and clean unmatched data
4. Run `eda.ipynb` - Perform exploratory data analysis
5. Run `fixing_dataset.ipynb` - Data preprocessing and feature engineering
6. Run `DecisionTree.ipynb` - Train Decision Tree with class weighting
7. Run `SMOTE_baru.ipynb` - Train Decision Tree with SMOTE oversampling
8. (Optional) Run `WithoutMonth.ipynb` - Experiment without month feature

### Notes
- Pastikan semua input files berada di directory yang sama dengan notebook
- Jalankan notebook secara berurutan sesuai workflow
- Setiap notebook akan generate file CSV sebagai input untuk notebook berikutnya
- File encoding: UTF-8 with BOM (utf-8-sig)
- DecisionTree.ipynb dan SMOTE_baru.ipynb dapat dijalankan secara parallel (tidak saling bergantung)
- SMOTE_baru.ipynb memerlukan library tambahan: `imbalanced-learn`

### Library Installation
```bash
# Standard libraries
pip install pandas numpy matplotlib seaborn scipy scikit-learn

# Additional for SMOTE
pip install imbalanced-learn

# For GeoJSON processing
pip install geopandas
```

## File Sizes
- kecamatan.geojson: ~varied (GeoJSON format)
- data_banjir_combine_final.csv: ~varied
- kecamatan_filtered.csv: ~small
- data_banjir_joined_clean.csv: ~medium
- data_banjir_filtered.csv: ~reduced size (~11.8% smaller)

## Key Findings

### From EDA
- No missing values in final dataset
- No duplicate rows, all 16 features complete
- Multiple Kabupaten/Kota across Jabodetabek with hundreds of unique Kecamatan
- Multi-year data with all 12 months represented
- Climate variables show strong relationship with flood occurrence
- T-test results show significant differences between flood and non-flood conditions

### From Feature Engineering
- 6 engineered features created based on domain knowledge
- Outliers handled using IQR clipping method
- Feature scaling applied for distance-based algorithms
- Unscaled version preserved for tree-based algorithms

### From Decision Tree Model (Class Weighting)
- **Best Model Performance:**
  - Test Accuracy: 0.8711 (87.11%)
  - Test Precision: 0.3184 (31.84%)
  - Test Recall: 0.4359 (43.59%)
  - Test F1-Score: 0.3678
  - AUC-ROC: 0.6817
- **Top Important Features:**
  1. month (18.24%)
  2. avg_rainfall (13.60%)
  3. rainfall_intensity (11.22%)
- **Model Insights:**
  - Class weighting handles imbalance algorithmically
  - More efficient (uses original data size)
  - Better precision (fewer false positives)
  - Training: 8,784 samples, Test: 2,196 samples

### From SMOTE Model (Data Oversampling)
- **Best Model Performance:**
  - Test Accuracy: 0.8711 (87.11%)
  - Test Precision: 0.2889 (28.89%)
  - Test Recall: 0.4615 (46.15%)
  - Test F1-Score: 0.3554
  - AUC-ROC: 0.6834
- **Top Important Features:**
  1. avg_rainfall (16.67%)
  2. month (12.86%)
  3. max_rainfall (10.56%)
- **Model Insights:**
  - SMOTE creates synthetic samples for minority class
  - Better recall (detects more actual floods)
  - Lower precision (more false positives)
  - Training: 16,216 samples (balanced), Test: 2,196 samples (original)
  - No data leakage - SMOTE only on training data

### Model Comparison: Class Weighting vs SMOTE

| Metric | Class Weighting | SMOTE | Winner |
|--------|----------------|-------|--------|
| Accuracy | 0.8711 | 0.8711 | TIE |
| Precision | 0.3184 | 0.2889 | Class Weight (+10%) |
| Recall | 0.4359 | 0.4615 | SMOTE (+5.9%) |
| F1-Score | 0.3678 | 0.3554 | Class Weight (+3.5%) |
| AUC-ROC | 0.6817 | 0.6834 | SMOTE (+0.2%) |
| Training Size | 8,784 | 16,216 | Class Weight (more efficient) |
| Computational Cost | Lower | Higher | Class Weight |

**Key Takeaways:**
- **Class Weighting** slightly better overall (higher F1-Score, better precision)
- **SMOTE** better for early warning systems (higher recall - detects more floods)
- **Both models below optimal threshold** (F1 target ≥ 0.70 for production)
- High accuracy (87%) misleading due to class imbalance
- Need improvements: ensemble methods, feature engineering, threshold tuning

**Recommendations:**
- Use **class_weight='balanced'** for production (more efficient, better F1)
- Use **SMOTE** if recall is critical (e.g., early warning systems)
- Consider hybrid approach: moderate SMOTE + class weights
- Implement Random Forest or XGBoost for better performance
- Tune classification threshold to optimize recall vs precision trade-off

## Model Output Files

### DecisionTree.ipynb Outputs
- `decision_tree_rules.txt` - Human-readable decision rules (class_weight approach)
- `decision_tree_visualization.png` - Tree structure visualization (class_weight)
- Performance metrics and confusion matrix visualizations

### SMOTE_baru.ipynb Outputs
- `data_train_smote.csv` - Balanced training data (16,216 rows)
- `data_test_original.csv` - Original test data (2,196 rows)
- `decision_tree_smote_rules.txt` - Human-readable decision rules (SMOTE approach)
- `decision_tree_smote_visualization.png` - Tree structure visualization (SMOTE)
- SMOTE effect visualizations and performance metrics

### Data Processing Outputs
- `data_banjir_processed.csv` - Processed dataset with scaling (for distance-based ML)
- `data_banjir_engineered.csv` - Engineered features without scaling (for tree-based ML)

---

## Future Work

### Immediate Improvements (Quick Wins)
- Threshold tuning for better recall vs precision balance
- Hybrid approach: Combine moderate SMOTE + class weights
- Cost-sensitive learning with custom class weights

### Model Enhancements
- Ensemble methods: Random Forest, XGBoost, Gradient Boosting
- Neural Networks for complex pattern recognition
- Stacking/Blending multiple models
- Advanced SMOTE variants: ADASYN, BorderlineSMOTE, SMOTE-ENN

### Feature Engineering
- Interaction features (e.g., avg_rainfall × is_rainy_season)
- Temporal aggregations (rolling averages, lag features)
- Geospatial clustering (flood-prone zones)
- Weather pattern features (consecutive rainy days)

### Data Collection
- Gather more flood occurrence data (increase minority class samples)
- Add external features: river levels, drainage system data
- Real-time weather data integration
- Historical flood damage data

### Deployment & Monitoring
- Real-time prediction system deployment
- Model monitoring and retraining pipeline
- API endpoint for flood risk queries
- Early warning dashboard with visualization

### Analysis & Validation
- Time-series cross-validation for temporal data
- Spatial cross-validation for geographic generalization
- External validation on different regions
- Cost-benefit analysis of false positives vs false negatives
