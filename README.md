houses predction
Developed a Machine Learning project for California house value prediction using real housing data. Tested Linear Regression, Ridge, Decision Tree, Random Forest, Gradient Boosting, and KNN. Applied preprocessing, feature scaling, model comparison, and evaluation using R², MAE, and RMSE, achieving strong predictive performance.
California House Value Prediction

This project predicts California house values using machine learning.

I started by loading the housing dataset and exploring its columns and data types.

Before building a model, I checked the dataset for missing values.

I also looked at the basic statistics of the numerical features.

After understanding the data, I prepared the features for machine learning.

I separated the target house value from the input features.

Then I split the dataset into training and testing data.

I started with Linear Regression as my baseline model.

The purpose was to have a simple result that I could compare other models against.

After the baseline, I tried several different regression algorithms.

I tested Ridge Regression, Decision Tree, Random Forest, Gradient Boosting, and KNN.

Each model works differently, so I wanted to see which one suited the data best.

I trained each model using the training data.

Then I used the test data to make predictions.

For evaluation, I used R², MAE, and RMSE.

R² helped me compare how much of the variation in house values the models could explain.

MAE showed the average absolute prediction error.

RMSE gave more weight to larger prediction errors.

I compared the models instead of assuming that the most complex model would be the best.

This made the project more useful as a learning experiment.

I also learned how different algorithms can produce very different results on the same dataset.

The project helped me understand the importance of having a baseline.

It also gave me more practice with regression and model evaluation.

The final model was chosen based on the actual test results.

what i used

Python, Pandas, NumPy, Matplotlib, and Scikit-learn.

Future Improvements

I would like to add cross-validation, hyperparameter tuning, better feature engineering, and model explainability.

I would also like to deploy the final model so users can enter house information and receive a prediction.
