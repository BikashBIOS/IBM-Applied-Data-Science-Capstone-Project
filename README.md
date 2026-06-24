# 🚀 IBM Applied Data Science Capstone Project
### SpaceX Falcon 9 First Stage Landing Prediction

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-green?logo=scikitlearn)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow)
![Plotly Dash](https://img.shields.io/badge/Plotly-Dash-blueviolet?logo=plotly)
![Folium](https://img.shields.io/badge/Folium-Maps-brightgreen)

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Business Problem & Scenario](#-business-problem--scenario)
- [Repository Structure](#-repository-structure)
- [Dataset Description](#-dataset-description)
- [Data Collection](#-data-collection)
- [Data Wrangling & Cleaning](#-data-wrangling--cleaning)
- [Exploratory Data Analysis (EDA)](#-exploratory-data-analysis-eda)
- [SQL-Based EDA](#-sql-based-eda)
- [Interactive Visual Analytics](#-interactive-visual-analytics)
- [Machine Learning Pipeline](#-machine-learning-pipeline)
- [TensorFlow Deep Learning Model](#-tensorflow-deep-learning-model)
- [Model Results & Evaluation](#-model-results--evaluation)
- [Confusion Matrix Analysis](#-confusion-matrix-analysis)
- [Key Findings & Conclusions](#-key-findings--conclusions)
- [Tech Stack](#-tech-stack)
- [How to Run](#-how-to-run)

---

## 🌐 Project Overview

This repository contains all hands-on lab assignments and the full capstone project completed as part of the **IBM Applied Data Science Professional Certificate** on Coursera. The capstone applies the complete data science lifecycle — from raw data ingestion through model deployment — to a real-world aerospace prediction problem using SpaceX Falcon 9 launch data.

---

## 💼 Business Problem & Scenario

**Context:** SpaceX advertises Falcon 9 rocket launches at **\$62 million USD**, while competitors charge upward of **\$165 million**. The primary reason for this cost advantage is SpaceX's ability to **reuse the first-stage booster** after each launch.

**Business Goal:** A competing startup (**SpaceY**) wants to bid against SpaceX for rocket launches. To estimate launch costs and make competitive bids, SpaceY needs to predict whether the Falcon 9's first stage will successfully land and be reusable.

**Core Question:**
> _"Given a set of Falcon 9 launch features (payload mass, orbit type, launch site, booster version, etc.), will the first stage of the rocket land successfully?"_

This is a **binary classification problem**: `1 = Successful Landing`, `0 = Unsuccessful Landing`.

---

## 📁 Repository Structure

```
IBM-Applied-Data-Science-Capstone-Project/
│
├── Data_Collection_API.ipynb      # SpaceX REST API data collection
├── Web_Scraping.ipynb             # Wikipedia scraping with BeautifulSoup
├── data_wrangling.ipynb           # Data cleaning & label engineering
├── eda_sql.ipynb                  # Exploratory analysis using SQL (Db2)
├── eda_visualization.ipynb        # EDA with Matplotlib & Seaborn
├── iva_folium.ipynb               # Interactive geographic maps with Folium
├── spacex_dash_app.py             # Plotly Dash interactive dashboard
├── ML_Prediction.ipynb            # ML models + TensorFlow deep learning
└── README.md                      # Project documentation (this file)
```

---

## 📊 Dataset Description

Data was collected from two primary sources:

### Source 1 — SpaceX REST API
**Endpoint:** `https://api.spacexdata.com/v4/launches`
Filtered to **Falcon 9** launches only.

### Source 2 — Wikipedia Web Scraping
**Page:** _List of Falcon 9 and Falcon Heavy launches_ (BeautifulSoup)

### Final Dataset Shape
| Property          | Value                        |
|-------------------|------------------------------|
| Total Records     | **90 rows** (Falcon 9 only)  |
| Total Features    | **17 columns**               |
| Target Variable   | `Class` (0 = Fail, 1 = Success) |
| Class Distribution | ~64% Success, ~36% Failure  |

### Feature Dictionary

| Feature         | Type        | Description                                                    |
|-----------------|-------------|----------------------------------------------------------------|
| `FlightNumber`  | Integer     | Sequential launch number                                       |
| `Date`          | DateTime    | Launch date                                                    |
| `BoosterVersion`| Categorical | Version of the Falcon 9 booster (e.g., FT, B4, B5)           |
| `PayloadMass`   | Float (kg)  | Mass of payload being carried                                 |
| `Orbit`         | Categorical | Target orbit (LEO, GTO, ISS, SSO, VLEO, HEO, PO, etc.)       |
| `LaunchSite`    | Categorical | Physical launch location (CCAFS SLC-40, KSC LC-39A, VAFB)    |
| `Outcome`       | String      | Raw landing outcome label                                      |
| `Flights`       | Integer     | Number of flights made with that core                          |
| `GridFins`      | Boolean     | Whether grid fins were deployed                               |
| `Reused`        | Boolean     | Whether the booster core was reused                           |
| `Legs`          | Boolean     | Whether landing legs were deployed                            |
| `LandingPad`    | Categorical | Specific landing pad identifier                               |
| `Block`         | Float       | Core version block number                                     |
| `ReusedCount`   | Integer     | Number of times that core had been previously reused          |
| `Serial`        | String      | Unique core serial identifier                                 |
| `Longitude`     | Float       | Launch site longitude                                         |
| `Latitude`      | Float       | Launch site latitude                                          |
| `Class`         | Integer     | **Target** — 1 = Successful landing, 0 = Unsuccessful         |

---

## 🔌 Data Collection

### Method 1 — SpaceX REST API (`Data_Collection_API.ipynb`)

```python
import requests
import pandas as pd
from pandas import json_normalize

url = "https://api.spacexdata.com/v4/launches"
response = requests.get(url)
data = json_normalize(response.json())

# Filter to Falcon 9 only
falcon9 = data[data['rocket'] == '5e9d0d95aca009351e68af46']

# Additional endpoint calls to resolve IDs
# - /v4/rockets/   → Rocket name
# - /v4/launchpads/ → Launchpad name, longitude, latitude
# - /v4/payloads/  → Payload mass, target orbit
# - /v4/cores/     → Landing outcome, grid fins, reuse data, landing pad, block
```

**Missing Values Imputation:**
- `PayloadMass`: 5 missing values → replaced with **column mean**
- `LandingPad`: 26 missing values → handled in feature engineering

---

### Method 2 — Web Scraping (`Web_Scraping.ipynb`)

```python
from bs4 import BeautifulSoup
import requests

url = "https://en.wikipedia.org/wiki/List_of_Falcon_9_and_Falcon_Heavy_launches"
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# Parse launch tables
tables = soup.find_all('table', {'class': 'wikitable'})
```

Columns extracted: Date/Time, Version, Booster, Launch Site, Payload, Orbit, Customer, Launch Outcome, Booster Landing.

---

## 🧹 Data Wrangling & Cleaning (`data_wrangling.ipynb`)

### Steps Performed

**1. Outcome Label Engineering**

The raw `Outcome` column contained 9 distinct string values:

| Outcome Value         | Meaning                              | Class Label |
|-----------------------|--------------------------------------|-------------|
| `True ASDS`           | Successful drone ship landing        | **1**       |
| `True RTLS`           | Successful return-to-launch landing  | **1**       |
| `True Ocean`          | Successful ocean landing             | **1**       |
| `False ASDS`          | Unsuccessful drone ship landing      | **0**       |
| `False RTLS`          | Unsuccessful return-to-launch        | **0**       |
| `None ASDS`           | No landing attempt, drone ship       | **0**       |
| `None None`           | No landing attempt at all            | **0**       |
| `False Ocean`         | Failed controlled ocean landing      | **0**       |
| `None`                | Unclassified miss                    | **0**       |

```python
# Binary label assignment
landing_class = {
    'True ASDS': 1, 'True RTLS': 1, 'True Ocean': 1,
    'False ASDS': 0, 'False RTLS': 0, 'None ASDS': 0,
    'None None': 0, 'False Ocean': 0, 'None': 0
}
df['Class'] = df['Outcome'].map(landing_class)
```

**2. One-Hot Encoding**

Categorical columns were one-hot encoded before ML:
```python
features = df[['FlightNumber','PayloadMass','Orbit','LaunchSite',
               'Flights','GridFins','Reused','Legs','LandingPad',
               'Block','ReusedCount','Serial']]
features_one_hot = pd.get_dummies(features)
features_one_hot = features_one_hot.astype(float)
```

**3. Feature Standardization**

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(features_one_hot)
```

**Summary of Cleaning Steps:**

| Step                     | Action                                      |
|--------------------------|---------------------------------------------|
| Filter rocket type       | Kept only Falcon 9 launches                 |
| Resolve API IDs          | Mapped IDs to human-readable values         |
| Handle missing payload   | Imputed with column mean (5 values)         |
| Engineer target label    | Converted `Outcome` strings to binary Class |
| Encode categoricals      | One-hot encoding of 5 categorical features  |
| Standardize numerics     | StandardScaler applied to all features      |

---

## 📈 Exploratory Data Analysis (EDA)

### Visualizations (`eda_visualization.ipynb`)

Using **Matplotlib** and **Seaborn**, the following plots were generated:

#### 1. Flight Number vs. Payload Mass
Shows a positive correlation — higher flight numbers (more recent missions) had increasingly successful landings regardless of payload mass, confirming SpaceX's rapid improvement curve.

#### 2. Flight Number vs. Launch Site
Revealed that **CCAFS SLC-40** dominated early launches, with **KSC LC-39A** increasing in later missions.

#### 3. Payload Mass vs. Launch Site
- CCAFS SLC-40: Wide range of payloads, increasing success with heavier payloads
- VAFB SLC-4E: Consistently lighter payloads (Polar orbit missions)
- KSC LC-39A: Medium-to-heavy payloads

#### 4. Success Rate by Orbit Type

| Orbit | Success Rate |
|-------|-------------|
| HEO   | 100%         |
| SSO   | 100%         |
| ES-L1 | 100%         |
| VLEO  | ~88%         |
| ISS   | ~76%         |
| LEO   | ~68%         |
| GTO   | ~49%         |
| PO    | ~55%         |

GTO (Geostationary Transfer Orbit) had the lowest success rate due to high energy demands leaving less fuel for booster recovery.

#### 5. Success Rate by Year

```
2013: ~0%    2014: ~0%    2015: ~14%
2016: ~50%   2017: ~78%   2018: ~92%
2019: ~100%  2020: ~100%
```

A clear improvement trend — SpaceX's success rate climbed dramatically from 2017 onwards.

#### 6. Payload vs. Orbit (Scatter Plot)
- LEO, ISS, and Polar orbits: heavier payloads correlated with more success
- GTO: success was mixed regardless of payload mass

---

## 🗄️ SQL-Based EDA (`eda_sql.ipynb`)

Data was loaded into an **IBM Db2 cloud database** and queried with SQL for structured analysis.

### Key SQL Queries & Results

```sql
-- Total payload launched by NASA (CRS missions)
SELECT SUM(PAYLOAD_MASS__KG_) AS total_payload
FROM SPACEXTBL
WHERE CUSTOMER LIKE '%NASA (CRS)%';
-- Result: ~45,596 kg

-- Average payload mass for booster version F9 v1.1
SELECT AVG(PAYLOAD_MASS__KG_)
FROM SPACEXTBL
WHERE BOOSTER_VERSION LIKE 'F9 v1.1%';
-- Result: ~2,928 kg

-- First successful ground landing date
SELECT MIN(DATE)
FROM SPACEXTBL
WHERE LANDING_OUTCOME = 'Success (ground pad)';
-- Result: 2015-12-22

-- Drone ship successes with payloads 4000–6000 kg
SELECT BOOSTER_VERSION, LAUNCH_SITE
FROM SPACEXTBL
WHERE LANDING_OUTCOME = 'Success (drone ship)'
AND PAYLOAD_MASS__KG_ BETWEEN 4000 AND 6000;

-- Total successful vs failed mission outcomes
SELECT LANDING_OUTCOME, COUNT(*) AS count
FROM SPACEXTBL
WHERE DATE BETWEEN '2010-06-04' AND '2017-03-20'
GROUP BY LANDING_OUTCOME
ORDER BY count DESC;
```

**Summary Statistics from SQL:**
- **101 total launches** in the dataset
- **61 successful landings** (~60.4% success rate)
- **10 outright failures** (~9.9%)
- Remaining outcomes: controlled ocean landings, no-attempt missions
- Pre-2017: 8 successes | Post-2017: 53 successes

---

## 🗺️ Interactive Visual Analytics

### Folium Maps (`iva_folium.ipynb`)

Interactive geographic maps were built to analyze launch site characteristics:

**Launch Site Locations:**

| Site           | Location         | Coast  | Nearest City     | Distance to City |
|----------------|------------------|--------|------------------|------------------|
| CCAFS SLC-40   | Cape Canaveral, FL | East  | Titusville, FL   | ~23 km           |
| KSC LC-39A     | Merritt Island, FL | East  | Titusville, FL   | ~20 km           |
| VAFB SLC-4E    | Vandenberg, CA   | West   | Lompoc, CA       | ~14 km           |

**Proximity Analysis (Folium Lines & Markers):**
- ✅ All launch sites are adjacent to **coastlines** (for ocean abort paths)
- ✅ All sites are within range of **railways**
- ✅ CCAFS and KSC are near major highways; VAFB is near roads but no major highway
- ❌ All sites maintain safe distances from **populated cities**
- ✅ Sites cluster in Florida (East Coast) and California (West Coast)

**Folium Features Used:**
```python
import folium
from folium.plugins import MarkerCluster

map_spacex = folium.Map(location=[28.5623, -80.5774], zoom_start=5)
folium.CircleMarker(location=[lat, lon], radius=5,
                    color='green', popup='CCAFS SLC-40').add_to(map_spacex)
# Distance measurement lines between sites and nearby features
```

---

### Plotly Dash Dashboard (`spacex_dash_app.py`)

An interactive dashboard with:
- **Pie chart**: Success counts per launch site (filterable by site)
- **Scatter plot**: Payload mass vs. outcome, color-coded by booster version
- **Dropdown**: Filter by individual launch site or all sites
- **Range slider**: Filter launches by payload mass (0–10,000 kg)

**Key Dashboard Insights:**
- CCAFS SLC-40 + KSC LC-39A combined account for ~58.9% of all successful landings
- Payload mass range **2,000–3,700 kg** correlates most strongly with successful landings
- The **FT booster** version was dominant in successful missions
- Higher success rates are visible in the post-2017 data slice

---

## 🤖 Machine Learning Pipeline (`ML_Prediction.ipynb`)

### Data Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=2
)

# Dataset sizes
# Training set: 72 samples (80%)
# Test set:     18 samples (20%)
```

### Hyperparameter Tuning

**GridSearchCV** was applied to all four classical models:

```python
from sklearn.model_selection import GridSearchCV

parameters = {'C': [0.01, 0.1, 1], 'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
              'gamma': [0.1, 0.01]}
svm = SVC()
clf_svm = GridSearchCV(svm, parameters, cv=10, scoring='accuracy')
clf_svm.fit(X_train, y_train)
```

### Models Trained

| # | Model                  | Library        | Tuning Method  |
|---|------------------------|----------------|----------------|
| 1 | Logistic Regression    | Scikit-learn   | GridSearchCV   |
| 2 | Support Vector Machine | Scikit-learn   | GridSearchCV   |
| 3 | Decision Tree          | Scikit-learn   | GridSearchCV   |
| 4 | K-Nearest Neighbors    | Scikit-learn   | GridSearchCV   |
| 5 | Neural Network         | TensorFlow 2.x | Manual tuning  |

---

### Best Hyperparameters (GridSearchCV)

**Logistic Regression:**
```python
{'C': 0.01, 'penalty': 'l2', 'solver': 'lbfgs'}
```

**Support Vector Machine:**
```python
{'C': 1.0, 'gamma': 0.0316, 'kernel': 'sigmoid'}
```

**Decision Tree:**
```python
{'criterion': 'entropy', 'max_depth': 10, 'max_features': 'sqrt',
 'min_samples_leaf': 4, 'min_samples_split': 5, 'splitter': 'random'}
```

**K-Nearest Neighbors:**
```python
{'algorithm': 'auto', 'n_neighbors': 10, 'p': 1}
```

---

## 🧠 TensorFlow Deep Learning Model

A deep neural network was built using **TensorFlow 2.x / Keras** for comparison against classical ML models.

### Model Architecture

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),

    Dense(32, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),

    Dense(16, activation='relu'),

    Dense(1, activation='sigmoid')  # Binary classification output
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()
```

### Model Summary

```
Model: "sequential"
_________________________________________________________________
 Layer (type)             Output Shape         Param #
=================================================================
 dense (Dense)            (None, 64)           5,440
 batch_normalization       (None, 64)           256
 dropout (Dropout)         (None, 64)           0
 dense_1 (Dense)           (None, 32)           2,080
 batch_normalization_1     (None, 32)           128
 dropout_1 (Dropout)       (None, 32)           0
 dense_2 (Dense)           (None, 16)           528
 dense_3 (Dense)           (None, 1)            17
=================================================================
Total params: 8,449
Trainable params: 8,257
Non-trainable params: 192
_________________________________________________________________
```

### Training

```python
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=16,
    validation_split=0.1,
    verbose=1
)
```

### Training Configuration

| Parameter       | Value                    |
|-----------------|--------------------------|
| Optimizer       | Adam                     |
| Loss Function   | Binary Crossentropy      |
| Metric          | Accuracy                 |
| Epochs          | 100                      |
| Batch Size      | 16                       |
| Validation Split| 10% of training data     |
| Regularization  | Dropout (0.2–0.3) + BatchNorm |

### Evaluation

```python
loss, accuracy = model.evaluate(X_test, y_test)
print(f"TensorFlow Model Test Accuracy: {accuracy:.4f}")
```

---

## 📊 Model Results & Evaluation

### Training vs. Test Accuracy Summary

| Model                    | Train Accuracy | Test Accuracy | GridSearchCV Best Score |
|--------------------------|----------------|---------------|--------------------------|
| Logistic Regression      | 29.8%          | **83.33%**    | ~80.6%                   |
| Support Vector Machine   | 84.8%          | **83.33%**    | ~84.8%                   |
| Decision Tree            | 87.7%          | **83.33%**    | ~74.5%                   |
| K-Nearest Neighbors      | 84.8%          | **83.33%**    | ~84.8%                   |
| TensorFlow Neural Network| ~88.0%         | **83.33%**    | N/A (manual tuning)      |

> **Note:** All five models converged to the same **83.33% test accuracy**, indicating the dataset size (18 test samples) is the limiting factor. GridSearchCV cross-validation scores provide more differentiating power at this scale.

### Detailed Metrics per Model

#### Logistic Regression
| Metric      | Value  |
|-------------|--------|
| Accuracy    | 83.33% |
| Precision   | 0.83   |
| Recall      | 0.83   |
| F1-Score    | 0.83   |

#### Support Vector Machine (SVM)
| Metric      | Value  |
|-------------|--------|
| Accuracy    | 83.33% |
| Precision   | 0.83   |
| Recall      | 0.83   |
| F1-Score    | 0.83   |

#### Decision Tree
| Metric      | Value  |
|-------------|--------|
| Accuracy    | 83.33% |
| Precision   | 0.83   |
| Recall      | 0.83   |
| F1-Score    | 0.83   |

#### K-Nearest Neighbors
| Metric      | Value  |
|-------------|--------|
| Accuracy    | 83.33% |
| Precision   | 0.83   |
| Recall      | 0.83   |
| F1-Score    | 0.83   |

#### TensorFlow Neural Network
| Metric      | Value  |
|-------------|--------|
| Accuracy    | 83.33% |
| Loss        | ~0.38  |
| Precision   | 0.83   |
| Recall      | 0.83   |
| F1-Score    | 0.83   |

---

## 🔲 Confusion Matrix Analysis

All classical models and the TensorFlow model shared the same confusion matrix on the test set:

```
              Predicted: 0    Predicted: 1
Actual: 0   [     3               2    ]   ← True Negatives / False Positives
Actual: 1   [     1               12   ]   ← False Negatives / True Positives
```

### Confusion Matrix Breakdown

| Metric                       | Value | Interpretation                               |
|------------------------------|-------|----------------------------------------------|
| True Positives (TP)          | 12    | Correctly predicted successful landings       |
| True Negatives (TN)          | 3     | Correctly predicted failed landings           |
| False Positives (FP)         | 2     | Predicted success but actually failed         |
| False Negatives (FN)         | 1     | Predicted failure but actually succeeded      |
| **Total Test Samples**       | **18**|                                              |
| **Accuracy**                 | **83.33%** | (TP + TN) / Total = 15/18              |
| **Precision** (Class 1)      | 0.857 | TP / (TP + FP) = 12/14                      |
| **Recall** (Class 1)         | 0.923 | TP / (TP + FN) = 12/13                      |
| **F1-Score** (Class 1)       | 0.889 | 2 × (Precision × Recall) / (Precision + Recall) |
| **Specificity** (Class 0)    | 0.600 | TN / (TN + FP) = 3/5                        |

### Visualization Code

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

y_pred = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=['Landing Failure', 'Landing Success'])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix — Falcon 9 Landing Prediction')
plt.tight_layout()
plt.show()
```

### Interpretation

- The model is **better at predicting successes** (high recall of 0.923) than failures (specificity of 0.600).
- False Negatives (1) are preferable to False Positives (2) from a **cost-estimation standpoint** — under-predicting successes costs SpaceY less than over-predicting them.
- The class imbalance (~64% success in the dataset) contributes to the model's bias toward predicting Class 1.

---

## 🔑 Key Findings & Conclusions

### 1. Temporal Trend is the Strongest Signal
SpaceX's landing success rate increased dramatically after **March 2017**:
- Pre-March 2017: **8 successes** out of ~45 launches
- Post-March 2017: **53 successes** out of ~56 launches

Time (as a proxy for SpaceX's engineering improvements) is the dominant predictor.

### 2. Launch Site Characteristics
- All 3 launch sites are located on **US coastlines** (for ocean abort safety)
- All sites maintain distances of **14–23 km from nearby cities**
- All sites are near **railways**; East Coast sites are near major highways

### 3. Orbit Type Matters
- **HEO, SSO, ES-L1**: 100% success rate (but very few missions)
- **LEO, ISS, VLEO**: High success rates (~76–88%)
- **GTO**: Lowest success rate (~49%) due to high fuel consumption leaving less for booster recovery

### 4. Payload Mass Sweet Spot
- Payload mass in the range **2,000–3,700 kg** correlates most strongly with successful landings
- VAFB SLC-4E (Vandenberg) consistently carried lighter payloads

### 5. Machine Learning Conclusions
- All four classical models + TensorFlow achieved **identical 83.33% test accuracy**
- **GridSearchCV cross-validation** is the differentiator: SVM and KNN scored ~84.8% CV accuracy (best)
- **Decision Tree** achieved the highest training accuracy (87.7%) but converged to the same test score
- **Logistic Regression** had the lowest training accuracy (29.8%) — it underfits but generalizes well
- No single model is a clear winner at this dataset scale; more data would create meaningful differentiation
- The TensorFlow model demonstrates that deep learning adds **complexity without accuracy gains** on small datasets

### 6. Recommended Model for Production
**Support Vector Machine (SVM)** or **KNN** are recommended based on highest cross-validation scores (~84.8%), indicating better generalization potential on unseen data.

---

## 🛠️ Tech Stack

| Category              | Tools & Libraries                                      |
|-----------------------|-------------------------------------------------------|
| Language              | Python 3.9+                                           |
| Data Collection       | `requests`, `BeautifulSoup4`, SpaceX REST API         |
| Data Processing       | `pandas`, `numpy`                                     |
| Database & SQL        | IBM Db2, `ibm_db_sa`, `SQLAlchemy`                    |
| Visualization         | `matplotlib`, `seaborn`, `plotly`, `folium`           |
| Dashboard             | `Plotly Dash`                                         |
| Machine Learning      | `scikit-learn` (LR, SVM, DT, KNN, GridSearchCV)      |
| Deep Learning         | `TensorFlow 2.x`, `Keras`                             |
| Environment           | Jupyter Notebook, IBM Watson Studio                   |

---

## ▶️ How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/BikashBIOS/IBM-Applied-Data-Science-Capstone-Project.git
cd IBM-Applied-Data-Science-Capstone-Project
```

### 2. Install Dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn \
            tensorflow requests beautifulsoup4 \
            folium plotly dash ibm_db_sa sqlalchemy
```

### 3. Run Notebooks in Order
```bash
jupyter notebook
```

Execute notebooks in this sequence:
1. `Data_Collection_API.ipynb`
2. `Web_Scraping.ipynb`
3. `data_wrangling.ipynb`
4. `eda_sql.ipynb`
5. `eda_visualization.ipynb`
6. `iva_folium.ipynb`
7. `ML_Prediction.ipynb`

### 4. Run the Dash App
```bash
python spacex_dash_app.py
# Open http://127.0.0.1:8050 in your browser
```

---

## 📜 Certificate & Course

This project was completed as part of the **IBM Data Science Professional Certificate** on Coursera.

**Course:** Applied Data Science Capstone (Course 10 of 10)

---

## 🤝 Acknowledgements

- **IBM Skills Network** for course design and IBM Watson Studio access
- **SpaceX** for the publicly available REST API (`api.spacexdata.com`)
- **Wikipedia** for the Falcon 9 launch history tables
- **Coursera** for the platform and peer review structure

---

*Made with ❤️ by BikashBIOS as part of the IBM Applied Data Science Professional Certificate Capstone*
