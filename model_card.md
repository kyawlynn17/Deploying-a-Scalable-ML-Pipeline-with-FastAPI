# Model Card

## Model Details

This project uses a Random Forest classifier to predict whether an individual's annual income is greater than $50,000 or less than or equal to $50,000 based on Census Income data.

The model was trained using scikit-learn. Categorical features were one-hot encoded and the target salary variable was converted to a binary label.

## Intended Use

The model is intended for educational purposes as part of a machine learning deployment project.

It predicts one of two income categories:

- `<=50K`
- `>50K`

The model is not intended to be used for high-stakes decisions such as employment, lending, insurance, housing, or other decisions that could significantly affect an individual.

## Training Data

The model was trained using the Census Income dataset provided with the project in `data/census.csv`.

The dataset contains demographic and employment-related features such as age, workclass, education, occupation, marital status, race, sex, hours worked per week, and native country.

The data was split into training and test sets using an 80/20 train-test split with a fixed random seed for reproducibility.

Categorical variables were processed using one-hot encoding.

## Evaluation Data

The evaluation data consists of the 20% test split held out from the Census Income dataset.

The test data was processed using the same encoder learned from the training data.

Model performance was also evaluated on categorical data slices. For each unique value of each categorical feature, precision, recall, and F1 score were computed and stored in `slice_output.txt`.

## Metrics

The model was evaluated using precision, recall, and F1 score.

Overall performance on the held-out test set was:

- Precision: **0.7353**
- Recall: **0.6378**
- F1 score: **0.6831**

Slice-level performance varies between demographic and employment-related groups. The full slice metrics are available in `slice_output.txt`.

## Ethical Considerations

The Census Income dataset contains sensitive demographic attributes including race and sex.

As a result, model predictions may reflect historical biases or imbalances present in the source data. Performance may also differ across demographic groups, as shown by the slice-level evaluation.

This model should therefore not be used to make high-stakes decisions about individuals.

The model was created for educational and demonstration purposes only.

## Caveats and Recommendations

The model is trained on historical Census Income data and may not generalize well to current populations or populations that differ from the training data.

Some categorical values have very few observations, which can produce unstable or misleading slice-level metrics.

Future improvements could include:

- additional hyperparameter tuning
- cross-validation
- fairness analysis across sensitive groups
- additional feature engineering
- evaluation on newer or external datasets
- calibration and threshold analysis
