# Air Quality and Health Impact Analysis Project

## Project Purpose

The purpose of this project is to analyze the impact of air quality on public health by leveraging machine learning algorithms. It aims to classify health impact levels based on various air quality indicators and environmental factors. By building predictive models, the project helps in identifying areas with higher health risks due to poor air quality. This can assist policymakers in making informed decisions for public health interventions.

## Code Explanation

### 1. Data Loading and Exploration

In the first section of the code, the project starts by importing the `pandas` library, which is essential for handling and manipulating data in Python. The dataset containing air quality and health impact data is loaded using the `pd.read_csv` function. After successfully loading the data into a DataFrame, the first two rows are displayed to provide a quick overview of the dataset, ensuring that it has been correctly imported and structured.

### 2. Gradient Boosting Classifier

This section explores the use of the **Gradient Boosting** algorithm, a powerful machine learning technique for classification tasks.

- Necessary libraries such as `pandas`, `numpy`, and `scikit-learn` are imported.
- The dataset is reloaded, and a custom function `classify_health_impact` is defined to categorize the `HealthImpactScore` into specific classes. This new information is added as a column to the dataset.
- Features (X) and the target variable (y) are prepared, followed by splitting the data into training and testing sets.
- Features are standardized using `StandardScaler` to ensure that all variables contribute equally to model performance.
- A `GradientBoostingClassifier` is initialized and trained on the data.
- A custom function `get_user_input` is created to allow users to input specific data points, which are then scaled for prediction purposes. The model outputs the predicted health impact class along with its confidence level.
- The model is evaluated using metrics like the F1 score, classification report, and confusion matrix to assess its overall performance.

### 3. Decision Tree Classifier

In this section, the project explores the **Decision Tree** algorithm, another widely used machine learning technique for classification tasks.

- The necessary libraries are imported, and the dataset is loaded once again.
- Missing values are checked using the `isnull().sum()` function.
- The data is split into features (X) and the target variable (y), and then divided into training and testing sets.
- Features are standardized to improve model performance.
- A `DecisionTreeClassifier` is initialized and trained on the scaled data.
- The model is evaluated using the F1 score and accuracy metrics.
- Similar to the previous section, the model allows for specific data points to be inputted by users, which are then scaled and used for prediction, demonstrating the model’s real-world applicability.

## Conclusion

This project highlights the use of machine learning algorithms to assess the impact of air quality on public health. By leveraging models such as **Gradient Boosting** and **Decision Trees**, the project offers a comprehensive approach to predicting health outcomes based on environmental data. These insights can be invaluable for shaping public health policy and intervention strategies.

