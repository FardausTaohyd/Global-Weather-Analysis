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
*   [Summarize some of your key findings from the EDA and model evaluation sections. For example: "Random Forest performed best among the classification models with an accuracy of X%," "Temperature distribution shows a peak around Y°C," "Certain features like cloud cover and humidity were identified as highly important for predicting weather conditions."]

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
*   [Mention any limitations of your analysis or models. For example: "The dataset represents a snapshot in time and may not capture long-term climate trends," "Some weather conditions have limited data points, affecting model performance for those classes," "The interpretability of some models (e.g., Neural Network) is limited."]

## Future Work
*   [Suggest potential areas for future exploration or improvement. For example: "Include time-series analysis to study weather changes over time," "Experiment with more advanced deep learning models for weather forecasting," "Incorporate additional weather-related datasets."]

## License
[Specify the license for your project, e.g., MIT, Apache 2.0]

## Contact
[Fardaus Taohyd/github.com/FardausTaohyd]
[Your Email Address (Optional)]
