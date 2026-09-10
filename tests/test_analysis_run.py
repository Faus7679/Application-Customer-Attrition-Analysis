import tempfile
import unittest
from pathlib import Path

import pandas as pd

import analysis_run


class AnalysisWorkflowSmokeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.original_paths = {
            'DATA_DIR': analysis_run.DATA_DIR,
            'REPORTS_DIR': analysis_run.REPORTS_DIR,
            'FIGURES_DIR': analysis_run.FIGURES_DIR,
            'RAW_DATA_PATH': analysis_run.RAW_DATA_PATH,
            'PREPARED_DATA_PATH': analysis_run.PREPARED_DATA_PATH,
            'REPORT_PATH': analysis_run.REPORT_PATH,
            'SUMMARY_PATH': analysis_run.SUMMARY_PATH,
        }
        self.tempdir = tempfile.TemporaryDirectory()
        self.base = Path(self.tempdir.name)
        analysis_run.DATA_DIR = self.base / 'data'
        analysis_run.REPORTS_DIR = self.base / 'reports'
        analysis_run.FIGURES_DIR = analysis_run.REPORTS_DIR / 'figures'
        analysis_run.RAW_DATA_PATH = analysis_run.DATA_DIR / 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
        analysis_run.PREPARED_DATA_PATH = analysis_run.DATA_DIR / 'prepared_telco_customer_churn.csv'
        analysis_run.REPORT_PATH = analysis_run.REPORTS_DIR / 'logistic_regression_analysis_report.md'
        analysis_run.SUMMARY_PATH = analysis_run.REPORTS_DIR / 'logistic_regression_model_summary.txt'

    def tearDown(self) -> None:
        for name, value in self.original_paths.items():
            setattr(analysis_run, name, value)
        self.tempdir.cleanup()

    def test_workflow_generates_expected_artifacts(self) -> None:
        source_df = pd.read_csv(self.original_paths['RAW_DATA_PATH'])
        sample_df = pd.concat(
            [
                source_df[source_df['Churn'] == 'No'].head(150),
                source_df[source_df['Churn'] == 'Yes'].head(150),
            ],
            ignore_index=True,
        )
        sample_path = self.base / 'sample.csv'
        sample_df.to_csv(sample_path, index=False)

        analysis_run.ensure_directories()
        raw_df, source, source_label = analysis_run.load_dataset(str(sample_path))
        prepared_df, prep_stats = analysis_run.prepare_dataset(raw_df)
        results = analysis_run.fit_models(prepared_df)
        analysis_run.create_visualizations(prepared_df, results)
        report = analysis_run.build_report(prepared_df, results, source, source_label, prep_stats)
        analysis_run.REPORT_PATH.write_text(report, encoding='utf-8')
        analysis_run.SUMMARY_PATH.write_text(results['statsmodels_model'].summary2().as_text(), encoding='utf-8')

        self.assertTrue(analysis_run.PREPARED_DATA_PATH.exists())
        self.assertTrue(analysis_run.REPORT_PATH.exists())
        self.assertTrue(analysis_run.SUMMARY_PATH.exists())
        self.assertTrue((analysis_run.FIGURES_DIR / 'roc_curve.png').exists())
        self.assertIn('Logistic Regression Analysis Report', analysis_run.REPORT_PATH.read_text(encoding='utf-8'))
        self.assertGreaterEqual(results['metrics']['roc_auc'], 0.0)
        self.assertLessEqual(results['metrics']['roc_auc'], 1.0)


if __name__ == '__main__':
    unittest.main()
