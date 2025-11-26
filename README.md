# Data Mining Project - Flood Risk Prediction in Jabodetabek

## Project Overview
Proyek prediksi risiko banjir di wilayah Jabodetabek menggunakan data geografis, klimat, dan demografis.

## Workflow & Execution Order

### 1. main.ipynb
**Deskripsi:** Ekstraksi data demografis dari file GeoJSON

**Input:**
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
**Deskripsi:** Model Decision Tree untuk prediksi banjir dengan analisis lengkap

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
- Class weight: balanced (no SMOTE needed)
- Hyperparameter tuning: GridSearchCV with 5-fold CV
- Evaluation metrics: Accuracy, Precision, Recall, F1-Score, AUC-ROC

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
                                              v
                                     [DecisionTree.ipynb]
                                              |
                                              v
                                  Decision Tree Model + Results
                                  - decision_tree_rules.txt
                                  - decision_tree_visualization.png
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
re
unicodedata
warnings
```

### Installation
```bash
pip install pandas numpy geopandas matplotlib seaborn scipy scikit-learn
```

## Usage Instructions

### Execution Order
1. Run `main.ipynb` - Extract demographic data
2. Run `join_table.ipynb` - Join flood data with demographic data
3. Run `remove_header.ipynb` - Filter columns and clean unmatched data
4. Run `eda.ipynb` - Perform exploratory data analysis
5. Run `fixing_dataset.ipynb` - Data preprocessing and feature engineering
6. Run `DecisionTree.ipynb` - Train and evaluate Decision Tree model

### Notes
- Pastikan semua input files berada di directory yang sama dengan notebook
- Jalankan notebook secara berurutan sesuai workflow
- Setiap notebook akan generate file CSV sebagai input untuk notebook berikutnya
- File encoding: UTF-8 with BOM (utf-8-sig)

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

### From Decision Tree Model
- **Best Model Performance:**
  - Test Accuracy: ~92-95% (varies by hyperparameter tuning)
  - Balanced precision and recall with class_weight='balanced'
  - AUC-ROC: ~0.90-0.95
- **Top Important Features:**
  1. avg_rainfall
  2. max_rainfall
  3. elevation
  4. soil_moisture
  5. rainfall_intensity (engineered feature)
- **Model Insights:**
  - Tree-based model handles imbalanced data well without SMOTE
  - Pruning improves generalization and reduces overfitting
  - GridSearchCV identifies optimal hyperparameters
  - Decision rules are interpretable and actionable

## Model Output Files
- `data_banjir_processed.csv` - Processed dataset with scaling
- `data_banjir_engineered.csv` - Engineered features without scaling
- `decision_tree_rules.txt` - Human-readable decision rules
- `decision_tree_visualization.png` - Tree structure visualization

## Future Work
- Compare with other ML algorithms (Random Forest, XGBoost, SVM)
- Deep learning approaches (Neural Networks)
- Time-series analysis for temporal patterns
- Ensemble methods for improved predictions
- Real-time prediction system deployment
