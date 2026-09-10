# Application-Customer-Attrition-Analysis

This repository contains a reproducible logistic regression analysis of the IBM Telco Customer Churn data set for a telecommunications attrition assignment.

## Deliverables

- `analysis_run.py` – end-to-end workflow for data loading, preparation, exploratory analysis, logistic regression modeling, and figure/report generation
- `reports/logistic_regression_analysis_report.md` – completed report addressing the assignment prompts
- `reports/logistic_regression_model_summary.txt` – full statistical summary from the logistic regression model
- `reports/figures/` – graphical outputs used in the report
- `data/prepared_telco_customer_churn.csv` – cleaned analysis-ready data

## Data source

The script first looks for a repository-local copy of the assignment Excel workbook in `data/CST-570-RS-WAFn-UseC-Telco-Customer-Churn.xlsx`. If no local workbook is present, it uses a public mirror of the IBM Telco Customer Churn CSV and caches it in `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`. You can also point the workflow at any local Excel or CSV file with `--input-path`.

## Run the workflow

```bash
python -m pip install -r requirements.txt
python analysis_run.py
```

To point at a specific local source file:

```bash
python analysis_run.py --input-path /absolute/path/to/CST-570-RS-WAFn-UseC-Telco-Customer-Churn.xlsx
```
