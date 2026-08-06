📌 Project Overview
This project aims to predict a startup's profit based on its key operational expenses using multiple machine learning and deep learning models. The dataset includes variables such as R&D Spend, Administration Cost, and Marketing Spend as input features, and Profit as the target variable.

📁 Project Structure
Profit_Prediction_Of_Startup.ipynb – Jupyter Notebook containing data preprocessing, model training, evaluation, and prediction logic.

50_Startups.csv – Dataset with historical startup expenditure and profit data.

🧰 Technologies Used
Programming Language: Python

Libraries:

Data Manipulation: Pandas, NumPy

Visualization: Matplotlib, Seaborn

Machine Learning: Scikit-learn, XGBoost , Regression

Deep Learning: TensorFlow

🤖 Models Implemented
The following regression models were trained and evaluated:

✅ Linear Regression

✅ Lasso Regression

✅ Ridge Regression

✅ ElasticNet Regression

✅ Decision Tree Regressor

✅ Random Forest Regressor

✅ Support Vector Regressor (SVR)

✅ XGBoost Regressor

✅ Feedforward Neural Network (FNN)

📊 Model Evaluation Metrics
All models were evaluated based on:

🔹 R² Score

🔹 Mean Absolute Error (MAE)

🔹 Mean Squared Error (MSE)

A ranking was derived from the above metrics to determine the best-performing model.

🏆 Best Performing Model
After comprehensive evaluation, the Random Forest Regressor achieved the highest accuracy and was selected as the final model for profit prediction.

🚀 How to Use
🖥️ Steps to Run:
Clone the repository:
git clone <https://github.com/rushi330/ML-projects.git>

Navigate into the project directory and install dependencies:
pip install pandas numpy matplotlib seaborn scikit-learn xgboost tensorflow
Launch Jupyter Notebook:

jupyter notebook
Open Profit_Prediction_Of_Startup.ipynb and run all cells sequentially.

🧮 Example Input:
Enter R & D Spend         : 8000  
Enter Administration Cost : 5000  
Enter Marketing Spend     : 110000
📈 Output:
The predicted profit of the Startup is: ₹53,324.09
