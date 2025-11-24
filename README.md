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
                          v
                   [eda.ipynb]
                          |
                          v
              Analysis & Visualizations
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
re
unicodedata
```

### Installation
```bash
pip install pandas numpy geopandas matplotlib seaborn scipy
```

## Usage Instructions

### Execution Order
1. Run `main.ipynb` - Extract demographic data
2. Run `join_table.ipynb` - Join flood data with demographic data
3. Run `remove_header.ipynb` - Filter columns and clean unmatched data
4. Run `eda.ipynb` - Perform exploratory data analysis

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

## Key Findings (from EDA)

### Data Quality
- No missing values in final dataset
- No duplicate rows
- All 16 features complete

### Geographic Coverage
- Multiple Kabupaten/Kota across Jabodetabek
- Hundreds of unique Kecamatan
- Geographic bounds: Latitude and Longitude ranges cover Jakarta metropolitan area

### Temporal Coverage
- Multi-year data
- All 12 months represented
- Temporal patterns visible

### Statistical Insights
- Significant features identified through correlation analysis
- T-test results show significant differences between flood and non-flood conditions
- Climate variables show strong relationship with flood occurrence
