# Credit Card Default Prediction Using Logistic Regression

A machine learning project focused on predicting credit card payment defaults through an interpretable logistic regression model. The project explores customer financial and demographic characteristics, evaluates predictive performance under class imbalance, and analyzes the factors associated with payment default.

## Project Overview

Credit risk assessment is an important task for financial institutions. Identifying customers who are more likely to default can support risk management and decision-making.

This project develops a logistic regression classifier to estimate the probability of credit card payment default using demographic information, credit limits, billing statements, and payment behavior.

The analysis uses the **Default of Credit Card Clients** dataset, which contains 30,000 customer records from a Taiwanese bank.

## Dataset

- **Source:** [UCI Machine Learning Repository – Default of Credit Card Clients](https://doi.org/10.24432/C55S3H)
- **Observations:** 30,000 customers
- **Explanatory variables:** 23
- **Target variable:** `default`
- **Prediction target:** Whether a customer will default on their credit card payment in the following month.

The dataset includes information such as:

- Credit limit.
- Demographic characteristics, including age, education, and marital status.
- Repayment status.
- Monthly bill statements.
- Previous payment amounts.

The target variable is binary:
- `1`: Payment default.
- `0`: No payment default.

## Methodology

The project follows a five-stage workflow:

### 1. Data Cleaning and Preprocessing

- Removed the customer ID column and standardized the target variable name.
- Reviewed undocumented categories in education and marital status and grouped them into an "Others" category.
- Checked missing values and duplicate records.
- Interpreted undocumented repayment-status codes (`-2` and `0`) using contextual assumptions.
- Created derived variables and grouped repayment-status categories.
- Aggregated monthly billing and payment amounts using their averages to reduce multicollinearity.
- Scaled monetary variables to units of 10,000 New Taiwan Dollars (NTD).

### 2. Exploratory Data Analysis (EDA)

Explored the distribution of customer characteristics and payment behavior, including:

- Default rates across demographic groups.
- Repayment status and its relationship with default.
- Credit limits, billing statements, and payment amounts.
- Correlations among numerical variables.
- Class imbalance in the target variable.

### 3. Train-Test Split

The dataset was divided using stratified sampling to preserve the original target distribution:

| Dataset | Records | Percentage |
|---|---:|---:|
| Training | 24,000 | 80% |
| Testing | 6,000 | 20% |
| **Total** | **30,000** | **100%** |

A random seed of `42` was used to support reproducibility.

### 4. Logistic Regression Modeling

An interpretable logistic regression model was fitted using `statsmodels`.

The model incorporated continuous and categorical predictors related to customer demographics, credit limits, and payment behavior.

Statistical analysis included:

- Coefficient estimation using maximum likelihood.
- Odds ratios (OR).
- 95% confidence intervals.
- Wald tests for individual predictor significance.
- Likelihood ratio test.
- McFadden's pseudo-\(R^2\).
- Variance Inflation Factor (VIF) to assess multicollinearity.

### 5. Model Evaluation and Threshold Optimization

Since the dataset is imbalanced (approximately 78% non-defaulting customers and 22% defaulting customers), accuracy alone was not considered sufficient.

The classification threshold was evaluated using the training set, exploring values from `0.10` to `0.60` in increments of `0.01`. The threshold that maximized the F1-score was `0.25`.

The final model was evaluated on the independent test set using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix
- ROC curve

## Results

<details>
<summary>Model performance on the test set</summary>

| Metric | Threshold = 0.50 | Threshold = 0.25 |
|---|---:|---:|
| Accuracy | 82.0% | 78.0% |
| Precision | 69.9% | 50.3% |
| Recall | 32.7% | 50.1% |
| F1-score | — | 0.502 |

**ROC-AUC: 0.742**

The threshold of `0.25` increased recall from 32.7% to 50.1%, improving the model's ability to identify customers who actually defaulted. This came at the cost of lower precision and accuracy.

</details>

## Key Findings

- **Recent repayment behavior was the strongest predictor of default.** Customers with a payment delay of three or more months had 14.08 times the odds of default compared with the minimum-payment reference category (OR = 14.08; 95% CI: 11.08–17.89).
- A one-month payment delay was also strongly associated with default (OR = 3.72; 95% CI: 3.38–4.11).
- Higher credit limits and average payment amounts were associated with lower odds of default.
- The model achieved moderate discrimination, with a ROC-AUC of 0.742.
- Threshold selection substantially affected the balance between precision and recall, highlighting the importance of evaluating classification decisions in imbalanced datasets.

*Odds ratios describe statistical associations and should not be interpreted as causal effects.*

## Technologies and Libraries

| Technology | Purpose |
|---|---|
| Python 3.13 | Programming language |
| Google Colab | Development environment |
| Pandas | Data manipulation and preprocessing |
| SciPy | Statistical analysis |
| Statsmodels | Logistic regression and statistical inference |
| Scikit-learn | Data splitting and model evaluation |
| Plotly | Interactive data visualizations |

## How to Run

The analysis was developed in Google Colab using Python 3.13.

1. Clone or download this repository.
2. Open the project notebook in Google Colab or a compatible Jupyter environment.
3. Install the required dependencies if they are not already available:

```bash
pip install pandas scipy statsmodels scikit-learn plotly
```

4. Download the dataset from the [UCI Machine Learning Repository](https://doi.org/10.24432/C55S3H).
5. Update the dataset path in the notebook if necessary.
6. Run the notebook cells in sequence to reproduce the preprocessing, exploratory analysis, model training, and evaluation.

## Limitations and Future Improvements

Some limitations should be considered when interpreting the results:

- The dataset comes from a single Taiwanese bank and covers a period in 2005, limiting its direct applicability to current financial environments or other countries.
- Evaluation relies on a single stratified train-test split rather than cross-validation.
- The classification threshold was optimized for F1-score, not for actual financial costs associated with false positives and false negatives.
- The model captures linear relationships in the log-odds and does not include interaction terms or nonlinear effects.
- Some repayment-status values required assumptions because they are not officially documented in the dataset.

Potential improvements include:

- Applying stratified cross-validation.
- Comparing logistic regression with regularized models, decision trees, and ensemble methods.
- Incorporating additional repayment-history features.
- Selecting the classification threshold based on a business cost matrix.
- Evaluating the model using more recent datasets and data from other markets.
- Assessing model fairness across demographic groups.

## References

- Yeh, I.-C. (2016). *Default of credit card clients* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C55S3H
- Yeh, I.-C., & Lien, C.-H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. *Expert Systems with Applications, 36*(2), 2473–2480. https://doi.org/10.1016/j.eswa.2007.12.020
- Seabold, S., & Perktold, J. (2010). Statsmodels: Econometric and statistical modeling with Python. *Proceedings of the 9th Python in Science Conference*, 92–96. https://doi.org/10.25080/Majora-92bf1922-011

---

*This project was developed for educational and analytical purposes. It is not intended to be used as a production-ready credit scoring system.*
