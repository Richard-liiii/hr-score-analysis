# hr-score-analysis
First Year Project – HR Data Analysis and Prediction Tool
📊 HR Training Score Analysis and Prediction

🔍 Overview
This project explores an HR dataset to identify patterns in employee training scores. It allows users to filter the data by fields like department or education, visualize key trends, and apply machine learning techniques to predict and classify training performance.

🧠 Features
Interactive filtering of HR data based on user input (e.g., department, education level).
Automated data cleaning and encoding of categorical variables.

  Data visualizations:
    Correlation heatmap
    Training score distribution
    Age vs. training score with regression line

  Machine Learning Models:
    Regression: Predict average training score using MLPRegressor
    Classification: Label scores as Low, Medium, or High using MLPClassifier
    Evaluation with metrics like MSE, R², confusion matrix, and classification report

🛠 Technologies Used
Python
pandas
scikit-learn
matplotlib
seaborn

📂 Dataset
The dataset (emp.csv) contains anonymized HR data including age, department, education, gender, and training scores.

📈 Output
Regression results: Mean Squared Error and R² score printed.
Classification results: Confusion matrix and classification report printed.
Visualizations: Correlation heatmap, training score histogram, scatter plot, regression prediction plots, and confusion matrix saved as PNG files and displayed.

📬 Author
Richard Li
rli09975@usc.edu

📝 License
This project is for academic purposes and may be adapted or extended with credit to the author.
