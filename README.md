# Lab3_NumPy
 Introduction : This lab focuses on building a machine learning model to predict Titanic survival. It includes data preprocessing, handling missing values, feature encoding, and scaling. Multiple models (KNN, Logistic Regression, Decision Tree) are trained, evaluated, and compared using accuracy and confusion matrix
 

Overview :
This lab explores machine learning models to predict passenger survival on the Titanic dataset.
We begin with a baseline KNN model, then extend the analysis with Logistic Regression and Decision Tree Classifier.
The goal is to compare performance across models, interpret results, and highlight trade-offs between accuracy, interpretability, and generalization.

Step 1: Import Libraries
 { pandas, numpy, matplotlib, seaborn }
[ scikit-learn (preprocessing, models, metrics) ]

Step 2: Load Data
Training and test datasets loaded from CSV files.

Step 3: Data Cleaning & Preprocessing: 
Dropped irrelevant columns: PassengerId, Name, Ticket, Cabin.

Handled missing values: median for Age, mode for Embarked, median for Fare.

Encoded categorical variables with one-hot encoding.

Aligned train and test columns.

Separated features (X_train, X_test) and target (y_train, y_test).

Train Models:
KNN: Baseline model, sensitive to scaling and parameter choice.

Logistic Regression: Achieved highest accuracy, interpretable coefficients.

Decision Tree: Captured non-linear rules, provided feature importance, mild overfitting risk.

Evaluation:
Each model was evaluated using accuracy, classification report, and confusion matrix.

Visualizations:
Logistic Regression coefficients (feature impact).

Decision Tree feature importance.

Confusion matrix heatmaps for all models.

Conclusion:
KNN provided a baseline but was sensitive to scaling and parameter choice.

Logistic Regression achieved the highest accuracy (~81%) and offered interpretable coefficients, making it the most robust model.

Decision Tree captured non-linear rules and provided human-readable decision paths, though it risked mild overfitting.

Overall, Logistic Regression emerged as the best-performing model, while Decision Tree added interpretability and KNN highlighted preprocessing importance.

