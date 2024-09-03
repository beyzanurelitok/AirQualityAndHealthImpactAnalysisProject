# AirQualityAndHealthImpactAnalysisProject

Project Purpose:

The purpose of this project is to analyze the impact of air quality on public health by leveraging machine learning algorithms. It aims to classify health impact levels based on various air quality indicators and environmental factors. By building predictive models, the project can help in identifying areas with higher health risks due to poor air quality and thus assist policymakers in making informed decisions for public health interventions.

Code Explanation:

In the first cell, the project begins by importing the necessary pandas library, which is essential for handling and manipulating data in Python. The code then specifies the file path to the dataset, which contains air quality and health impact data. The pd.read_csv function is used to read this dataset into a DataFrame, enabling efficient data analysis. Finally, the code displays the first two rows of the dataset to provide a quick glimpse of the data structure, ensuring that it has been loaded correctly.

In the second cell, this cell delves into the Gradient Boosting algorithm, a powerful machine learning technique used for classification tasks. After importing the necessary libraries (pandas, numpy, scikit-learn), the code reads the dataset once again. A custom function classify_health_impact is defined to classify the HealthImpactScore into specific categories, which are then added as a new column in the dataset.
The features (X) and the target variable (y) are prepared, and the dataset is split into training and testing sets. The features are standardized using StandardScaler to ensure that all variables contribute equally to the model's performance.
A GradientBoostingClassifier is initialized and trained on the dataset. Additionally, a function get_user_input is created to allow users to input specific data points, which are then scaled and used for prediction. The predicted health impact class and its confidence level are displayed. The model is also evaluated using the F1 score, classification report, and confusion matrix to assess its performance.

The third cell explores the use of the Decision Tree algorithm, another popular machine learning technique for classification tasks. After importing necessary libraries, the code reads the dataset and checks for any missing values using the isnull().sum() function. The data is then split into features (X) and the target variable (y), followed by a train-test split to prepare for model training.
Standard scaling is applied to the features to improve the model's performance. A DecisionTreeClassifier is then initialized and trained on the scaled data. The model's performance is evaluated using the F1 score and accuracy metrics. Finally, the code allows for inputting specific data points, which are scaled and used for prediction, demonstrating the model's application in real-world scenarios.

Finally, This project highlights the use of machine learning algorithms to assess the impact of air quality on public health. By using models such as Gradient Boosting and Decision Trees, the project offers a comprehensive approach to predicting health outcomes based on environmental data, providing valuable insights for public health policy and intervention strategies.
