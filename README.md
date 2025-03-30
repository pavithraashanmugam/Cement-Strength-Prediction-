# Cement Strength Prediction Using Machine Learning

## Project Overview

This project aims to predict the **compressive strength** of concrete based on its ingredients, such as Cement, Blast Furnace Slag, Fly Ash, Water, Superplasticizer, Coarse Aggregate, Fine Aggregate, and Age, using machine learning techniques. The dataset contains concrete mix data, and we use this data to train a machine learning model to predict the **Strength** of concrete.

## Dataset

The dataset contains the following columns:
- **Cement**: The amount of cement in the concrete mix (kg)
- **Blast Furnace Slag**: The amount of blast furnace slag in the concrete mix (kg)
- **Fly Ash**: The amount of fly ash in the concrete mix (kg)
- **Water**: The amount of water in the concrete mix (kg)
- **Superplasticizer**: The amount of superplasticizer used in the concrete mix (kg)
- **Coarse Aggregate**: The amount of coarse aggregate in the concrete mix (kg)
- **Fine Aggregate**: The amount of fine aggregate in the concrete mix (kg)
- **Age**: The age of the concrete (days)
- **Strength**: The compressive strength of the concrete (MPa) - This is the target variable we are trying to predict.

### Sample Data
| Cement | Blast Furnace Slag | Fly Ash | Water | Superplasticizer | Coarse Aggregate | Fine Aggregate | Age | Strength |
|--------|--------------------|---------|-------|------------------|------------------|-----------------|-----|----------|
| 540.0  | 0.0                | 0.0     | 162.0 | 2.5              | 1040.0           | 676.0           | 28  | 79.99    |
| 540.0  | 0.0                | 0.0     | 162.0 | 2.5              | 1055.0           | 676.0           | 28  | 61.89    |
| 332.5  | 142.5              | 0.0     | 228.0 | 0.0              | 932.0            | 594.0           | 270 | 40.27    |
| 332.5  | 142.5              | 0.0     | 228.0 | 0.0              | 932.0            | 594.0           | 365 | 41.05    |

## Approach

1. **Data Loading**: The concrete dataset is loaded into a pandas DataFrame from a CSV file (`concrete_data.csv`).

2. **Data Preprocessing**: 
    - **Null values check**: We ensure that there are no missing values in the dataset.
    - **Descriptive Statistics**: We calculate summary statistics (mean, standard deviation, min, max) for each feature.
    - **Feature and Target Split**: We separate the features (input variables) from the target (Strength) variable.

3. **Data Standardization**: 
    - We standardize the features using **StandardScaler**. Standardization ensures that each feature has a mean of 0 and a standard deviation of 1. This helps improve the performance of machine learning algorithms, especially when the features have different scales.

4. **Data Splitting**:
    - The data is split into **training** and **testing** sets using an 80-20 split. 80% of the data is used for training, and 20% is used for testing the model.

5. **Model 1: Linear Regression**:
    - We initially try using **Linear Regression** for predicting concrete strength. Linear regression fits a linear relationship between the input features and the target variable.
    - After training the model, we evaluate its performance using metrics such as **Mean Squared Error (MSE)** and **R² score**.
    - The linear regression model did not perform well, as the R² score was only 0.62 for training data and 0.57 for testing data, indicating that linear regression could not fully capture the complexities of the relationship between the features and strength.

6. **Model 2: XGBoost**:
    - We switch to **XGBoost** (Extreme Gradient Boosting), which is a more advanced model that often performs better for regression tasks, especially with non-linear relationships.
    - XGBoost is a gradient boosting framework that builds decision trees iteratively to minimize the error.
    - After training the XGBoost model, we evaluate it using **Mean Squared Error (MSE)** and **R² score**.
    - The XGBoost model outperforms linear regression, with an R² score of 0.99 for training data and 0.91 for testing data.

7. **Model Evaluation**:
    - **MSE (Mean Squared Error)**: A lower MSE indicates better performance. We compare the training and testing MSE values for both models.
    - **R² Score**: The R² score indicates how well the model explains the variance in the data. An R² value closer to 1 indicates a better fit.

8. **Prediction**:
    - Finally, we use the trained XGBoost model to predict the **Strength** of concrete for a given input of features. The input for prediction consists of values such as cement content, water, age, and other ingredients in the concrete mix.

## Results

- **Linear Regression**:
  - **Training MSE**: 107.89
  - **Testing MSE**: 105.76
  - **Training R²**: 0.624
  - **Testing R²**: 0.570
  
- **XGBoost**:
  - **Training MSE**: 1.36
  - **Testing MSE**: 21.16
  - **Training R²**: 0.995
  - **Testing R²**: 0.914

The **XGBoost** model demonstrates much better performance, with a higher R² score and lower MSE compared to the linear regression model.

## Model Prediction

Using the trained **XGBoost** model, we predict the strength of concrete for a given input:

```python
input = [(266.0 ,114.0 ,0.0 ,228.0 ,0.0 ,932.0 ,670.0 ,90 )]
input_as_array = np.asarray(input)
input_reshaped = input_as_array.reshape(1,-1)
std_input = scaler.transform(input_reshaped)
predict = xgb_model.predict(std_input)
print(' The predicted value is', predict)
```

### Output:
```
The predicted value is [47.15569]
```

This predicted value (47.15 MPa) represents the expected compressive strength of the concrete given the specified mix of ingredients.

## Conclusion

- We successfully trained a machine learning model to predict the compressive strength of concrete.
- **Linear Regression** was initially tested but didn't give good results.
- **XGBoost** significantly improved the predictions, showing a strong fit with an R² score of 0.91 on the test data.
- This model can be used to predict the strength of concrete based on its ingredients, aiding in the design and optimization of concrete mixes.

## How to Run the Code

1. Clone this repository to your local machine.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Make sure the `concrete_data.csv` file is in the correct directory (or update the file path in the code).
4. Run the script or Jupyter notebook to train the model and make predictions.

## Dependencies

- `numpy`
- `pandas`
- `scikit-learn`
- `xgboost`
