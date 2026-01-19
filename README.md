# Spotify Chart Performance Analysis

A comprehensive data mining and machine learning project analyzing Spotify chart performance metrics to uncover patterns in song popularity and predict streaming success.

## 📊 Project Overview

This project explores aggregated chart performance and streaming metrics for individual songs using various data mining techniques, predictive modeling, and visualization methods. The analysis reveals insights into what drives listener engagement and helps predict which songs will become hits.

## 🎯 Objectives

- Analyze streaming patterns and chart performance metrics
- Identify factors that contribute to song popularity
- Predict total streams and Top-10 chart success
- Segment songs into distinct performance archetypes
- Visualize trends and relationships in the data

## 📁 Dataset Description

**Source:** Spotify chart data (11,084 records)

**Key Features:**
- `Position`: Current chart rank
- `Artist Name`: Performing artist
- `Song Name`: Track title
- `Days`: Total days on chart
- `Top 10 (xTimes)`: Number of Top-10 entries
- `Peak Position`: Best chart position achieved
- `Peak Position (xTimes)`: Days at peak position
- `Peak Streams`: Highest single-day stream count
- `Total Streams`: Cumulative streams across all days

## 🔧 Technologies & Libraries

```python
- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
```

## 📈 Methodology

### 1. Data Preparation
- **Data Cleaning**: Handled 4 missing song names, parsed text fields
- **Feature Engineering**: 
  - `streams_per_day`: Total Streams / Days
  - `ever_top10`: Binary flag for Top-10 entry
  - `peak_bucket`: Categorized peak positions
- **Outlier Detection**: IQR-based boxplot analysis
- **Data Splitting**: 75/25 train-test split

### 2. Exploratory Data Analysis
- **Descriptive Statistics**: Distribution analysis of all numeric features
- **Correlation Analysis**: Pearson correlation matrix revealing:
  - Days ↔ Total Streams: 0.93 (strong positive)
  - Peak Streams ↔ Total Streams: 0.45 (moderate positive)
  - Peak Position ↔ Total Streams: -0.38 (moderate negative)

### 3. Data Mining Techniques

#### Frequency Analysis
- Identified top 20 artists by #1 peak positions
- Drake leads with 18 #1 hits

#### K-Means Clustering (k=4)
Segmented artists into four distinct hit profiles:
- **Cluster 0 (Blue)**: Long-tail artists (300+ days, <200M streams)
- **Cluster 1 (Orange)**: Flash-in-the-pan (100 days, <150M streams)
- **Cluster 2 (Green)**: Reliable performers (50-200 days, 50-200M streams)
- **Cluster 3 (Red)**: Superstars (50-150 days, 200-900M+ streams)

#### Regression Models
Predicting total streams:

| Model | R² Score | RMSE |
|-------|----------|------|
| Linear Regression | 0.899 | 17.4M streams |
| Random Forest | 0.959 | 11.1M streams |

#### Classification Models
Predicting Top-10 hits:

| Model | Accuracy | ROC AUC | Precision (Class 1) | Recall (Class 1) |
|-------|----------|---------|---------------------|------------------|
| Logistic Regression | ~0.87 | 0.36 | 0.37 | 0.30 |
| Random Forest | 0.95 | 0.97 | 0.80 | 0.73 |

## 📊 Key Visualizations

1. **Top 10 Songs by Total Streams** (Column Chart)
2. **Top 10 Artists by #1 Peaks** (Bar Chart)
3. **Top-10 Hit Distribution** (Pie Chart) - 89% hit Top-10
4. **Total Streams Distribution** (Boxplot)
5. **Average Streams by Chart Days** (Line Plot)
6. **Peak vs Total Streams Comparison** (Multi-line Plot)
7. **Peak Streams vs Total Streams** (Scatter Plot)
8. **Artist Hit-Profile Clusters** (Scatter Plot)
9. **ROC Curves** (Classification Performance)
10. **Confusion Matrix** (Random Forest)

## 🔍 Key Findings

1. **Longevity Matters**: Chart longevity (Days) is the strongest predictor of total streams (r=0.93)

2. **Viral ≠ Sustained Success**: High peak streams don't always translate to massive total streams

3. **Four Hit Archetypes**: Songs cluster into distinct performance patterns - from viral sensations to steady evergreens

4. **Random Forest Superiority**: Non-linear models significantly outperform linear models for both regression and classification tasks

5. **Top-10 Dominance**: 89% of charting songs reach Top-10, indicating dataset skew toward hits

6. **Artist Concentration**: A small number of superstar artists (Drake, Taylor Swift, Post Malone) dominate #1 positions

## 💡 Business Applications

- **Playlist Curation**: Target different cluster profiles for specific playlist types
- **Marketing Strategy**: Identify potential breakout hits early for promotional investment
- **A&R Decisions**: Predict long-term streaming potential from early performance metrics
- **Resource Allocation**: Prioritize marketing spend on songs predicted to reach Top-10

## 📂 Project Structure

```
spotify-analysis/
│
├── data/
│   └── Spotify_final_dataset.csv
│
├── notebooks/
│   ├── Data_Cleaning.py
│   ├── EDA.py
│   ├── Clustering.py
│   ├── Regression.py
│   └── Classification.py
│
├── visualizations/
│   ├── bar_chart.png
│   ├── pie_chart.png
│   ├── boxplot_total_streams.png
│   ├── line_plot.png
│   ├── scatter_plot.png
│   └── artist_clusters.png
│
├── models/
│   ├── regression_results.csv
│   └── classification_results.csv
│
├── README.md
└── requirements.txt
```

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Installation
```bash
git clone https://github.com/yourusername/spotify-chart-analysis.git
cd spotify-chart-analysis
pip install -r requirements.txt
```

### Usage
```python
# Load and explore the dataset
import pandas as pd
df = pd.read_csv('data/Spotify_final_dataset.csv')

# Run analysis scripts
python notebooks/Data_Cleaning.py
python notebooks/EDA.py
python notebooks/Clustering.py
python notebooks/Regression.py
python notebooks/Classification.py
```

## 📝 Future Enhancements

- [ ] Incorporate audio features (tempo, energy, valence)
- [ ] Time-series analysis of streaming trends
- [ ] Genre-based performance comparison
- [ ] Deep learning models for improved predictions
- [ ] Real-time chart position prediction
- [ ] Cross-platform analysis (Spotify, Apple Music, YouTube)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👏 Acknowledgments

- Dataset: [Kaggle Spotify Dataset](https://www.kaggle.com/)
- Reference: GeeksforGeeks tutorials on data mining and visualization
- Tools: Python, scikit-learn, pandas, matplotlib, seaborn

## 📧 Contact

For questions or feedback, please open an issue in the GitHub repository.

---

**Note**: This project is for educational and analytical purposes only. All data used is publicly available aggregated chart information.
