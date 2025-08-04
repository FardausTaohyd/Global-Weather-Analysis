# Global Weather Data Analysis

## Project Description
This project analyzes global weather data using Python in Google Colab. The analysis focuses on identifying climate trends, temperature variations and rainfall patterns across different regions. It utilizes popular Python libraries such as Pandas for data manipulation, and Matplotlib and Seaborn for data visualization. Various machine learning models are applied to predict weather conditions, and their performance is evaluated.

## Dataset
The dataset used in this project is the "Global Weather Repository," available on Kaggle.
Project dataset link: https://www.kaggle.com/datasets/nelgiriyewithana/global-weather-repository/data

## Project Structure
The project is structured as a Google Colab notebook, containing the following sections:

1.  **Setup and Data Loading:** Imports necessary libraries and loads the dataset.

2.  **Exploratory Data Analysis (EDA):** Visualizations and analysis to understand the data distribution, relationships between features, and initial insights into weather patterns. This includes:
    *   Pairplots of selected features.
    *   Correlation heatmap of numerical variables.
    *   Temperature distribution histogram.
    *   Sunrise and sunset hour distribution.
    *   Moon phases distribution pie chart.
    *   Scatter plot of weather conditions over time.
    
3.  **Data Preprocessing:** Steps to prepare the data for machine learning models, including handling missing values, encoding categorical features, and splitting data into training and testing sets.

4.  **Model Application and Evaluation:** Implementation and evaluation of various machine learning models for weather condition prediction. The models include:
    *   Logistic Regression
    *   Decision Tree Classifier
    *   Random Forest Classifier
    *   Support Vector Machine (SVM)
    *   K-Nearest Neighbors (KNN)
    *   Gaussian Naive Bayes
    *   Neural Network (Multi-layer Perceptron)
    *   Linear Regression (for continuous target)
    *   K-Means Clustering (for identifying weather clusters)
    
5.  **Results and Visualization:** Presentation of model performance metrics (accuracy, confusion matrix, classification report, R² score, Silhouette score) and visualizations to interpret results, including:
    *   Model comparison bar chart.
    *   Confusion matrix heatmap.
    *   Feature importance in Random Forest.
    *   K-Means cluster visualization using PCA.
    *   Cluster center values bar plot.
    
6.  **Comparison with Other Models:** A table comparing the performance and limitations of different models, including those from external sources.

## Libraries Used
*   pandas
*   numpy
*   sklearn
*   matplotlib
*   seaborn

## How to Run the Notebook
1.  Open the notebook in Google Colab.
2.  Ensure you have access to the dataset (you may need to upload it to your Colab environment or link your Kaggle account).
3.  Run each code cell sequentially.

## Key Findings
*   [Random Forest performed best among the classification models with an accuracy of approximately 90%.
The temperature distribution shows a peak around 20-30°C.
Sunrise times are mostly concentrated between 5-7 AM, and sunset times between 6-8 PM.
The most frequent moon phases are Waxing Crescent and Waning Crescent.
Features like cloud cover and humidity show high importance in the Random Forest model.
K-Means clustering with k=3 reveals distinct weather patterns based on the selected features.]

## Model Performance Summary
| Model               | Metric            | Score   |
|---------------------|-------------------|---------|
| Logistic Regression | Accuracy          | 0.73    |
| Decision Tree       | Accuracy          | 0.86    |
| Random Forest       | Accuracy          | 0.90    |
| SVM                 | Accuracy          | 0.79    |
| KNN                 | Accuracy          | 0.77    |
| Naive Bayes         | Accuracy          | 0.29    |
| Neural Network      | Accuracy          | 0.88    |
| Linear Regression   | R² Score          | 0.25    |
| K-Means Clustering  | Silhouette Score  | 0.13    |

*(Note: The metrics and scores for Linear Regression and K-Means Clustering are different from classification models and are reported accordingly.)*

## Limitations
*   [The dataset provides a snapshot of weather data and may not capture long-term climate trends or seasonal variations comprehensively.
Some weather conditions have very few data points, which can affect the performance of classification models for those specific classes.
The multi-class classification problem with a large number of classes (41 unique conditions) makes it challenging for some models to achieve high accuracy across all classes, as seen in the confusion matrices and classification reports.
The low Silhouette Score for K-Means clustering suggests that the clusters are not well-separated based on the features used.]

## Future Work
*   [Perform time-series analysis to investigate how weather patterns change over longer periods.
Explore more advanced deep learning models (e.g., LSTMs) for weather forecasting, especially for sequential data.
Incorporate additional data sources, such as historical weather data, to build more robust models.
Address the class imbalance issue for less frequent weather conditions to improve model performance on those classes.
Further investigate the optimal number of clusters for K-Means or explore other clustering algorithms.]

## Contact
fardaustaohyd31@gmail.com
