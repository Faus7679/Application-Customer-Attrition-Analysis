from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / 'data'
REPORTS_DIR = REPO_ROOT / 'reports'
FIGURES_DIR = REPORTS_DIR / 'figures'
DEFAULT_REMOTE_URL = 'https://raw.githubusercontent.com/treselle-systems/customer_churn_analysis/master/WA_Fn-UseC_-Telco-Customer-Churn.csv'
RAW_DATA_PATH = DATA_DIR / 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
PREPARED_DATA_PATH = DATA_DIR / 'prepared_telco_customer_churn.csv'
REPORT_PATH = REPORTS_DIR / 'logistic_regression_analysis_report.md'
SUMMARY_PATH = REPORTS_DIR / 'logistic_regression_model_summary.txt'
SEED = 42

sns.set_theme(style='whitegrid', palette='deep')


def ensure_directories() -> None:
    for directory in (DATA_DIR, REPORTS_DIR, FIGURES_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def resolve_input_path(cli_path: str | None) -> tuple[str, str]:
    if cli_path:
        return cli_path, 'user supplied file'

    local_excel = DATA_DIR / 'CST-570-RS-WAFn-UseC-Telco-Customer-Churn.xlsx'
    if local_excel.exists():
        return str(local_excel), 'repository Excel copy'

    if RAW_DATA_PATH.exists():
        return str(RAW_DATA_PATH), 'cached repository CSV'

    return DEFAULT_REMOTE_URL, 'public IBM Telco churn CSV mirror'


def load_dataset(cli_path: str | None) -> tuple[pd.DataFrame, str, str]:
    source, source_label = resolve_input_path(cli_path)
    if source.lower().endswith('.xlsx'):
        df = pd.read_excel(source)
    else:
        df = pd.read_csv(source)

    df.to_csv(RAW_DATA_PATH, index=False)
    return df, source, source_label


def prepare_dataset(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    df = raw_df.copy()
    df.columns = [column.strip() for column in df.columns]

    for column in df.select_dtypes(include='object').columns:
        df[column] = df[column].map(lambda value: value.strip() if isinstance(value, str) else value)

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    missing_totalcharges = int(df['TotalCharges'].isna().sum())

    df = df.dropna(subset=['TotalCharges']).copy()
    df['MultipleLines'] = df['MultipleLines'].replace({'No phone service': 'No'})
    no_internet_columns = [
        'OnlineSecurity',
        'OnlineBackup',
        'DeviceProtection',
        'TechSupport',
        'StreamingTV',
        'StreamingMovies',
    ]
    for column in no_internet_columns:
        df[column] = df[column].replace({'No internet service': 'No'})

    df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
    df['ChurnFlag'] = df['Churn'].map({'No': 0, 'Yes': 1})

    df.to_csv(PREPARED_DATA_PATH, index=False)
    return df, {'missing_totalcharges_dropped': missing_totalcharges}


def markdown_table(headers: Iterable[str], rows: Iterable[Iterable[object]]) -> str:
    headers = list(headers)
    header_row = '| ' + ' | '.join(map(str, headers)) + ' |'
    separator = '| ' + ' | '.join(['---'] * len(headers)) + ' |'
    body_rows = ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows]
    return '\n'.join([header_row, separator, *body_rows])


def format_p_value(value: float) -> str:
    return '<0.0001' if value < 0.0001 else f'{value:.4f}'


def prettify_feature_name(feature: str) -> str:
    feature = feature.replace('num__', '').replace('cat__', '')
    return feature.replace('_', ' = ', 1).replace('_', ' ')


def summarise_numeric(df: pd.DataFrame, columns: list[str]) -> list[list[object]]:
    rows: list[list[object]] = []
    for column in columns:
        series = df[column]
        rows.append([
            column,
            round(float(series.mean()), 2),
            round(float(series.median()), 2),
            round(float(series.std()), 2),
            round(float(series.min()), 2),
            round(float(series.max()), 2),
        ])
    return rows


def summarise_categorical(df: pd.DataFrame, columns: list[str]) -> list[list[object]]:
    rows: list[list[object]] = []
    for column in columns:
        top_level = df[column].mode().iat[0]
        top_share = (df[column].value_counts(normalize=True).iat[0] * 100)
        rows.append([column, df[column].nunique(), top_level, f'{top_share:.1f}%'])
    return rows


def plot_target_distribution(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    order = ['No', 'Yes']
    sns.countplot(data=df, x='Churn', order=order, ax=ax)
    ax.set_title('Target Variable Distribution: Churn')
    ax.set_xlabel('Churn')
    ax.set_ylabel('Customer count')
    for patch in ax.patches:
        ax.annotate(
            f'{int(patch.get_height())}',
            (patch.get_x() + patch.get_width() / 2, patch.get_height()),
            ha='center',
            va='bottom',
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'target_distribution.png', dpi=200)
    plt.close(fig)


def plot_numeric_distributions(df: pd.DataFrame, numeric_columns: list[str]) -> None:
    fig, axes = plt.subplots(1, len(numeric_columns), figsize=(15, 4))
    for ax, column in zip(axes, numeric_columns):
        sns.histplot(df[column], kde=True, ax=ax, color='#4C72B0')
        ax.set_title(column)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'numeric_predictor_distributions.png', dpi=200)
    plt.close(fig)


def plot_categorical_counts(df: pd.DataFrame, columns: list[str]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, column in zip(axes.flatten(), columns):
        order = df[column].value_counts().index
        sns.countplot(data=df, x=column, order=order, ax=ax)
        ax.set_title(f'{column} distribution')
        ax.tick_params(axis='x', rotation=20)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'categorical_predictor_counts.png', dpi=200)
    plt.close(fig)


def plot_numeric_by_churn(df: pd.DataFrame, numeric_columns: list[str]) -> None:
    fig, axes = plt.subplots(1, len(numeric_columns), figsize=(15, 4))
    for ax, column in zip(axes, numeric_columns):
        sns.boxplot(data=df, x='Churn', y=column, ax=ax)
        ax.set_title(f'{column} by churn')
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'numeric_predictors_by_churn.png', dpi=200)
    plt.close(fig)


def plot_churn_rate_by_category(df: pd.DataFrame, columns: list[str]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, column in zip(axes.flatten(), columns):
        rates = (
            df.groupby(column, observed=False)['ChurnFlag']
            .mean()
            .sort_values(ascending=False)
            .mul(100)
        )
        sns.barplot(x=rates.index, y=rates.values, ax=ax)
        ax.set_title(f'Churn rate by {column}')
        ax.set_ylabel('Churn rate (%)')
        ax.tick_params(axis='x', rotation=20)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'churn_rate_by_category.png', dpi=200)
    plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame, numeric_columns: list[str]) -> None:
    corr = df[numeric_columns + ['ChurnFlag']].corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', square=True, ax=ax)
    ax.set_title('Correlation heatmap for numeric predictors and churn')
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'correlation_heatmap.png', dpi=200)
    plt.close(fig)


def fit_models(df: pd.DataFrame) -> dict[str, object]:
    numeric_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']
    feature_columns = [column for column in df.columns if column not in {'customerID', 'Churn', 'ChurnFlag'}]
    categorical_columns = [column for column in feature_columns if column not in numeric_columns]

    X = df[feature_columns]
    y = df['ChurnFlag']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=SEED,
        stratify=y,
    )

    pipeline = Pipeline(
        steps=[
            (
                'preprocess',
                ColumnTransformer(
                    transformers=[
                        ('num', StandardScaler(), numeric_columns),
                        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), categorical_columns),
                    ]
                ),
            ),
            ('model', LogisticRegression(max_iter=4000, solver='liblinear')),
        ]
    )
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_prob),
    }

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    feature_names = pipeline.named_steps['preprocess'].get_feature_names_out()
    coefficients = pd.Series(pipeline.named_steps['model'].coef_[0], index=feature_names)

    statsmodels_X_train = pd.DataFrame(
        pipeline.named_steps['preprocess'].transform(X_train),
        columns=feature_names,
        index=X_train.index,
    )
    statsmodels_X_train = sm.add_constant(statsmodels_X_train)
    statsmodels_model = sm.Logit(y_train, statsmodels_X_train).fit(disp=False, maxiter=200)
    summary_frame = statsmodels_model.summary2().tables[1].copy()
    summary_frame['odds_ratio'] = np.exp(summary_frame['Coef.'])
    summary_frame['ci_lower_or'] = np.exp(summary_frame['Coef.'] - 1.96 * summary_frame['Std.Err.'])
    summary_frame['ci_upper_or'] = np.exp(summary_frame['Coef.'] + 1.96 * summary_frame['Std.Err.'])

    return {
        'numeric_columns': numeric_columns,
        'categorical_columns': categorical_columns,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'metrics': metrics,
        'fpr': fpr,
        'tpr': tpr,
        'coefficients': coefficients,
        'statsmodels_model': statsmodels_model,
        'summary_frame': summary_frame,
        'confusion_matrix': confusion_matrix(y_test, y_pred),
    }


def plot_model_outputs(results: dict[str, object]) -> None:
    y_test = results['y_test']
    y_prob = results['y_prob']
    confusion = results['confusion_matrix']
    fpr = results['fpr']
    tpr = results['tpr']
    roc_auc = results['metrics']['roc_auc']
    coefficients = results['coefficients']
    summary_frame = results['summary_frame']

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion, display_labels=['No churn', 'Churn']).plot(ax=ax, colorbar=False)
    ax.set_title('Confusion matrix')
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'confusion_matrix.png', dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.3f}')
    ax.plot([0, 1], [0, 1], linestyle='--', color='grey')
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.set_title('Receiver operating characteristic')
    ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'roc_curve.png', dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    score_df = pd.DataFrame({
        'Actual churn': y_test.map({0: 'No', 1: 'Yes'}).to_numpy(),
        'Predicted probability': y_prob,
    })
    sns.kdeplot(data=score_df, x='Predicted probability', hue='Actual churn', common_norm=False, fill=True, ax=ax)
    ax.set_title('Predicted churn probabilities by actual class')
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'predicted_probability_by_class.png', dpi=200)
    plt.close(fig)

    top_coefficients = coefficients.reindex(coefficients.abs().sort_values(ascending=False).head(10).index).sort_values()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=top_coefficients.values, y=top_coefficients.index, orient='h', ax=ax)
    ax.set_title('Top logistic regression coefficients in the model feature space')
    ax.set_xlabel('Coefficient value')
    ax.set_ylabel('Encoded feature')
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'top_model_coefficients.png', dpi=200)
    plt.close(fig)

    significant = summary_frame.drop(index='const').query('`P>|z|` < 0.05').copy()
    significant['abs_log_or'] = np.abs(np.log(significant['odds_ratio']))
    top_or = significant.sort_values('abs_log_or', ascending=False).head(10).sort_values('odds_ratio')
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=top_or['odds_ratio'], y=top_or.index, orient='h', ax=ax)
    ax.axvline(1.0, linestyle='--', color='grey')
    ax.set_title('Significant odds ratios from the statistical model')
    ax.set_xlabel('Odds ratio')
    ax.set_ylabel('Feature')
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'significant_odds_ratios.png', dpi=200)
    plt.close(fig)


def create_visualizations(df: pd.DataFrame, results: dict[str, object]) -> None:
    plot_target_distribution(df)
    plot_numeric_distributions(df, results['numeric_columns'])
    plot_categorical_counts(df, ['Contract', 'InternetService', 'PaymentMethod', 'TechSupport'])
    plot_numeric_by_churn(df, results['numeric_columns'])
    plot_churn_rate_by_category(df, ['Contract', 'InternetService', 'PaymentMethod', 'TechSupport'])
    plot_correlation_heatmap(df, results['numeric_columns'])
    plot_model_outputs(results)


def build_report(df: pd.DataFrame, results: dict[str, object], source: str, source_label: str, prep_stats: dict[str, int]) -> str:
    metrics = results['metrics']
    summary_frame = results['summary_frame']
    significant = summary_frame.drop(index='const').query('`P>|z|` < 0.05').copy()
    significant = significant.sort_values('P>|z|').head(10)
    predictor_count = len(results['numeric_columns']) + len(results['categorical_columns'])

    numeric_summary = summarise_numeric(df, results['numeric_columns'])
    categorical_summary = summarise_categorical(
        df,
        ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'InternetService', 'Contract', 'PaymentMethod'],
    )

    churn_rate = df['ChurnFlag'].mean() * 100
    contract_rates = (
        df.groupby('Contract', observed=False)['ChurnFlag'].mean().mul(100).sort_values(ascending=False)
    )
    internet_rates = (
        df.groupby('InternetService', observed=False)['ChurnFlag'].mean().mul(100).sort_values(ascending=False)
    )
    payment_rates = (
        df.groupby('PaymentMethod', observed=False)['ChurnFlag'].mean().mul(100).sort_values(ascending=False)
    )

    significant_rows = []
    for feature, row in significant.iterrows():
        direction = 'higher' if row['odds_ratio'] > 1 else 'lower'
        significant_rows.append([
            prettify_feature_name(feature),
            f"{row['Coef.']:.3f}",
            f"{row['odds_ratio']:.4f}",
            format_p_value(row['P>|z|']),
            direction,
        ])

    metrics_rows = [[
        'Accuracy', f"{metrics['accuracy']:.3f}"
    ], [
        'Precision', f"{metrics['precision']:.3f}"
    ], [
        'Recall', f"{metrics['recall']:.3f}"
    ], [
        'F1 score', f"{metrics['f1_score']:.3f}"
    ], [
        'ROC AUC', f"{metrics['roc_auc']:.3f}"
    ]]

    significant_bullets = '\n'.join(
        f"- `{prettify_feature_name(feature)}`"
        for feature in significant.index
    )

    report = f"""# Logistic Regression Analysis Report: Telecommunications Customer Attrition

## Introduction
This report analyzes the IBM Telco Customer Churn data set to explain why customers leave a telecommunications provider for competitors. The workflow is reproducible from this repository and supports the assignment with data preparation notes, exploratory findings, logistic regression modeling, and visual evidence.

**Data source used for this run:** `{source}` ({source_label}). When the assignment Excel workbook is unavailable, the repository falls back to the canonical public CSV mirror of the same Telco churn data.

## Part I. Data Preparation and Exploration

### 1. Data extraction and preparation
- The raw file was loaded into pandas and cached at `data/WA_Fn-UseC_-Telco-Customer-Churn.csv` for reproducibility.
- Column names and string values were stripped of leading/trailing whitespace.
- `TotalCharges` was converted from text to numeric.
- `{prep_stats['missing_totalcharges_dropped']}` blank `TotalCharges` values were coerced to missing and removed. These rows all belong to zero-tenure customers, so dropping them preserves a complete-case analysis without inventing bill totals.
- Redundant service labels were standardized by converting `No phone service` to `No` in `MultipleLines` and `No internet service` to `No` in internet add-on columns. This keeps categories interpretable and prevents perfect multicollinearity in the logistic model.
- `SeniorCitizen` was recoded to `Yes`/`No`, and the target `Churn` was encoded as `ChurnFlag` (1 = churn, 0 = stay).
- The prepared analysis table was saved as `data/prepared_telco_customer_churn.csv`.

### 2. How the data was prepared for the analysis
The preparation choices match the needs of logistic regression. Logistic regression requires a binary target, numeric predictors in machine-readable form, and well-defined categorical levels. Dummy encoding was applied automatically during modeling, while numeric predictors were standardized for the predictive model so coefficient magnitudes are comparable and solver convergence is stable.

### 3. Preliminary exploration findings
- The cleaned data contains **{len(df):,} customers** and **{predictor_count} predictors** plus the binary target.
- The churn rate is **{churn_rate:.1f}%**, indicating a moderately imbalanced but still usable classification problem.
- Contract type is strongly related to churn: **{contract_rates.index[0]}** customers churn at **{contract_rates.iloc[0]:.1f}%**, while **{contract_rates.index[-1]}** customers churn at **{contract_rates.iloc[-1]:.1f}%**.
- Internet service matters as well: **{internet_rates.index[0]}** customers have the highest churn rate at **{internet_rates.iloc[0]:.1f}%**.
- Payment behavior also differs by churn risk: **{payment_rates.index[0]}** has the highest churn rate at **{payment_rates.iloc[0]:.1f}%**.

#### Numeric predictor summary
{markdown_table(['Variable', 'Mean', 'Median', 'Std. Dev.', 'Min', 'Max'], numeric_summary)}

#### Selected categorical predictor summary
{markdown_table(['Variable', 'Levels', 'Most common level', 'Share'], categorical_summary)}

### 4. Justification of tools and visualization formats
- **pandas** was used for ingestion, cleaning, reshaping, and summary statistics because the data is tabular and requires careful type conversion.
- **seaborn** and **matplotlib** were used because histograms, count plots, boxplots, bar charts, heatmaps, and ROC curves communicate distributional and classification behavior clearly.
- **scikit-learn** was selected for the predictive workflow because its preprocessing pipeline cleanly combines scaling, one-hot encoding, train/test splitting, and evaluation metrics.
- **statsmodels** was used alongside scikit-learn because it provides coefficient-level inferential statistics such as z-tests, p-values, and confidence intervals, which are necessary for a report-oriented interpretation of the regression model.

### Supporting graphics
![Target distribution](figures/target_distribution.png)

![Numeric predictor distributions](figures/numeric_predictor_distributions.png)

![Selected categorical predictor counts](figures/categorical_predictor_counts.png)

![Correlation heatmap](figures/correlation_heatmap.png)

## Part II. Logistic Regression Analysis

### 1. Objective of the analysis
The goal of the analysis is to estimate the probability that a customer will churn and to identify which demographic, contract, billing, and service features are most strongly associated with attrition.

### 2. Target variable
The target variable is **`Churn`**, a binary field with values `Yes` and `No`. For modeling, it was encoded as **`ChurnFlag`** where `Yes = 1` and `No = 0`.

### 3. Univariate analysis of the predictors
The univariate review shows a mix of numeric and categorical predictors:
- `tenure` is right-skewed, with many newer customers and a long tail of retained long-tenure customers.
- `MonthlyCharges` spans a broad range, suggesting meaningful variation in service bundles and billing plans.
- `TotalCharges` is strongly right-skewed because it accumulates across tenure.
- Several categorical predictors are dominated by a few levels, especially `Month-to-month` contracts and electronic billing/payment combinations, which is common in churn studies.

### 4. Bivariate distribution analysis in the context of churn
- Customers who churn tend to have **lower tenure** and **higher monthly charges** than those who stay.
- Churn is highest among **month-to-month** customers and markedly lower for one-year and two-year contracts.
- **Fiber optic** customers show higher churn than DSL or customers without internet service.
- **Electronic check** users show higher churn than other payment groups.
- Support-oriented features such as **TechSupport = No** align with higher churn.

![Numeric predictors by churn](figures/numeric_predictors_by_churn.png)

![Churn rate by selected categories](figures/churn_rate_by_category.png)

### 5. Logistic regression procedure
1. Split the cleaned data into training and test sets using an 80/20 stratified split (`random_state = {SEED}`).
2. Standardized numeric predictors (`tenure`, `MonthlyCharges`, `TotalCharges`) for the predictive model.
3. One-hot encoded all categorical predictors with the first level dropped as the reference category.
4. Fit a scikit-learn logistic regression model for predictive evaluation.
5. Reused the same transformed training design matrix in statsmodels to estimate coefficient significance, odds ratios, and confidence intervals without using holdout labels in the inferential summary.
6. Evaluated out-of-sample classification performance with accuracy, precision, recall, and F1 score at the default **0.50 probability threshold**, and summarized ranking performance with ROC AUC.

### 6. Findings and model accuracy
#### Predictive performance on the holdout test set
{markdown_table(['Metric', 'Value'], metrics_rows)}

The model achieves **{metrics['accuracy']:.1%} accuracy** and an **ROC AUC of {metrics['roc_auc']:.3f}**, which indicates good discrimination for a business churn screen. Precision and recall are both moderate at the default **0.50 classification threshold**, which is expected because churn is the minority class and some false positives are acceptable in retention campaigns.

#### Most statistically significant predictors
{markdown_table(['Feature', 'Coefficient', 'Odds ratio', 'p-value', 'Effect on churn odds'], significant_rows)}

Interpretation highlights:
- Longer **tenure** reduces churn odds substantially.
- **One-year** and **two-year contracts** are associated with much lower churn odds than month-to-month service.
- **Electronic check**, **paperless billing**, and **multiple lines** are associated with higher churn odds.
- **Fiber optic** service aligns with higher churn odds, while customers without internet service are materially less likely to churn than the DSL reference group.
- Streaming-oriented service bundles can remain churn-prone even after controlling for contract and billing features.

![Confusion matrix](figures/confusion_matrix.png)

![ROC curve](figures/roc_curve.png)

![Predicted probability by actual class](figures/predicted_probability_by_class.png)

![Top model coefficients](figures/top_model_coefficients.png)

![Significant odds ratios](figures/significant_odds_ratios.png)

## Part III. Summary of Findings

### 1. Whether the data is discriminating
Yes. The data is meaningfully discriminating because the model separates churners from non-churners with an ROC AUC of **{metrics['roc_auc']:.3f}**. The probability-density plot and ROC curve show that churners receive systematically higher predicted probabilities than non-churners, even though the classes still overlap.

### 2. Variables that significantly interact with the target variable
Using the logistic regression significance tests, the strongest variables associated with churn are:
{significant_bullets}

In practical business terms, customer commitment structure (contract length), billing/payment behavior, and service configuration explain the largest share of churn risk.

### 3. Limitations of the logistic regression model
- Logistic regression assumes a linear relationship between predictors and the log-odds of churn, which may oversimplify customer behavior.
- The analysis is observational, so statistically significant effects should not be interpreted as direct causal drivers.
- Churn is the minority class, so threshold choice affects the precision/recall trade-off.
- The report does not include engineered interaction terms or non-linear transformations; adding them could improve fit but would reduce the simplicity and transparency of the baseline model.
- The data reflects one telecommunications portfolio and may not generalize perfectly to other markets or time periods.

## References
- Hosmer, D. W., Lemeshow, S., & Sturdivant, R. X. (2013). *Applied Logistic Regression* (3rd ed.). Wiley.
- James, G., Witten, D., Hastie, T., & Tibshirani, R. (2021). *An Introduction to Statistical Learning* (2nd ed.). Springer.
- IBM Sample Data. *Telco Customer Churn* data set, distributed through the public mirror referenced above.
"""
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description='Run the customer churn logistic regression analysis workflow.')
    parser.add_argument('--input-path', help='Optional path to the Excel or CSV source file.')
    args = parser.parse_args()

    ensure_directories()
    raw_df, source, source_label = load_dataset(args.input_path)
    df, prep_stats = prepare_dataset(raw_df)
    results = fit_models(df)
    create_visualizations(df, results)

    report = build_report(df, results, source, source_label, prep_stats)
    REPORT_PATH.write_text(report, encoding='utf-8')
    SUMMARY_PATH.write_text(results['statsmodels_model'].summary2().as_text(), encoding='utf-8')

    print(f'Prepared data written to: {PREPARED_DATA_PATH}')
    print(f'Report written to: {REPORT_PATH}')
    print(f'Model summary written to: {SUMMARY_PATH}')
    for key, value in results['metrics'].items():
        print(f'{key}: {value:.4f}')


if __name__ == '__main__':
    main()
