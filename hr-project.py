# Richard Li
# ITP-216, 31883
# final project
# This project looks at an HR dataset to find patterns in employee training scores The user can choose to filter the data by things like department or education level. The program shows helpful charts and uses machine learning to make predictions and group the scores into Low, Medium, or High.


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report


def clean_and_encode_data(df):
    '''
        Cleans the HR dataset by dropping the 'employee_id' column and any rows with missing values
        Encodes all categorical  columns using Label Encoding
    Parameters: df (pd.DataFrame)

    Returns: pd.DataFrame, dict_encoders

    Side Effects: Prints shape of DataFrame before and after cleaning
    '''
    print(f"Original shape: {df.shape}")
    df = df.drop(columns=['employee_id'])
    df = df.dropna()
    print(f"After dropping missing values: {df.shape}")

    encoders = {}
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    return df, encoders


def visualize_data(df):
    """
    Description: Generates and saves multiple plots:Correlation heatmap/Histogram of training scores/Scatter plot of age vs training score with regression line

    Parameters:df (pd.DataFrame)

    no return

    Side Effects:Saves and displays 3 plots 
    """
    #correlation heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("correlation_heatmap_hr.png")
    plt.show()

    # Score histogram
    plt.figure(figsize=(8, 5))
    sns.histplot(df['avg_training_score'], bins=30, kde=True)
    plt.title("Distribution of Training Scores")
    plt.xlabel("Score")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("training_score_distribution.png")
    plt.show()

    # Scatter plot with trend
    plt.figure(figsize=(10, 6))
    sns.regplot(x="age", y="avg_training_score", data=df, scatter_kws={'alpha':0.4})
    plt.title("Age vs. Average Training Score")
    plt.tight_layout()
    plt.savefig("age_vs_score_trend.png")
    plt.show()


def train_regression_model(df):
    """
    Description:TrainMLP regression model to predict avg_training_score with mse and r^2

    Parameters: df (pd.DataFrame)

    no returns

    Side Effects:Trains model, prints evaluation metrics, saves and shows plots comparing predicted and actual scores
    """
    X = df.drop(columns=['avg_training_score'])
    y = df['avg_training_score']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=100)

    model = MLPRegressor(hidden_layer_sizes=(32, 16, 8), max_iter=1000, random_state=100)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\nRegression Model Evaluation:")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R-squared Score: {r2:.2f}")

    # Plot predictions vs actuals
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.to_numpy()[:100], label='Actual', marker='o', color='blue')
    plt.plot(y_pred[:100], label='Predicted', marker='x', color='orange')
    plt.title("Line Plot: Actual vs Predicted Training Scores (first 100 samples)")
    plt.xlabel("Sample Index")
    plt.ylabel("Training Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig("lineplot_actual_vs_predicted.png")
    plt.show()
    
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], '--', color='red')
    plt.xlabel("Actual Score")
    plt.ylabel("Predicted Score")
    plt.title("Actual vs Predicted Training Scores")
    plt.tight_layout()
    plt.savefig("regression_prediction_vs_actual.png")
    plt.show()

def classify_training_score(df):
    """
    Description:
        Classifies avg_training_score into Low, Medium, and High categories then train the mlp classifier on the labeled data and visualize them in a confusion matrix

    Parameters:df (pd.DataFrame)

    no return

    Side Effects:print classification report and show confusion matrix

    """
    def label_score(score):
        if score < 50:
            return "Low"
        elif score < 70:
            return "Medium"
        else:
            return "High"

    df['score_label'] = df['avg_training_score'].apply(label_score)

    # Split and encode labels
    X = df.drop(columns=['avg_training_score', 'score_label'])
    y = df['score_label']
    label_enc = LabelEncoder()
    y_encoded = label_enc.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.25, random_state=100)
    clf = MLPClassifier(hidden_layer_sizes=(32,16,8), max_iter=1000, random_state=30)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    labels = label_enc.classes_
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix: Score Classification (1-50 low; 50-70 medium; 70-100 high)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig("conf_matrix_score_label.png")
    plt.show()

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=labels))



def main():
    """
    Description:
        Main function to run the full analysis pipeline:load dataset, clean the data, filter and analyse

    no parameter

    no return

    Side Effects:Reads file, prompts user input, modifies and filters data,displays and saves plots, prints outputs.
    """
    df = pd.read_csv("emp.csv")
    df, encoders = clean_and_encode_data(df)

    # Ask if the user want to filter the data
    print("\nWould you like to filter the dataset before analysis?")
    print("1. Yes")
    print("2. No")
    filter_choice = input("Enter choice (1/2): ")

    if filter_choice == "1":
        print("\nAvailable columns to filter on: department, education")
        filter_col = input("Enter column to filter by: ").strip().lower()

        if filter_col in encoders:
            le = encoders[filter_col]
            original_labels = le.classes_
            label_dict = {i: label for i, label in enumerate(original_labels)}
            print(f"\nAvailable values for {filter_col}:")
            for i, label in label_dict.items():
                print(f"{i}: {label}")

            try:
                user_input = input(f"Enter the label number to filter {filter_col} by: ").strip()
                filter_val = int(user_input)
                df = df[df[filter_col] == filter_val]
                print(f"Filtered data shape: {df.shape}")
            except:
                print("Invalid input. Proceeding without filtering.")

        elif filter_col in df.columns:
            unique_vals = df[filter_col].unique()
            print(f"\nAvailable values for {filter_col}: {list(unique_vals)}")
            filter_val = input(f"Enter the value to filter {filter_col} by: ")
            try:
                df = df[df[filter_col] == filter_val]
                print(f"Filtered data shape: {df.shape}")
            except:
                print("Invalid filter value. Proceeding without filtering.")
        else:
            print("Invalid column. Proceeding without filtering.")
    visualize_data(df)
    train_regression_model(df)
    classify_training_score(df)

if __name__ == '__main__':
    main()

