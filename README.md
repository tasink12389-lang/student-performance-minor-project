# Student Performance Analysis and Prediction System

A minor project that explores student academic patterns and predicts final examination marks with leakage-safe linear regression pipelines.

## What is included

- `notebooks/` and `src/` — notebook and Python implementation.
- `requirements.txt` — Python dependencies.
- `src/student_performance_analysis.py` — complete preprocessing, EDA, model evaluation, prediction, and output-generation workflow.
- `src/model.py` — guideline-named command-line entry point for the complete workflow.
- `notebooks/student_performance_analysis.ipynb` — notebook entry point for running the project.

The dataset, trained model, generated outputs, screenshots, and PDF report are not included. The dataset's redistribution permission is unknown, and generated files can contain record-level student information. These paths are ignored by Git.

## Machine-learning model

This is the **Student Performance Analysis and Prediction System**, not a fake-news detector. Its required machine-learning task is to predict final exam scores using Linear Regression; it also compares a cross-validated Ridge Regression model. Run `python src/model.py` to train and save the selected fitted pipeline to `models/student_score_model.joblib`. Its metrics and unseen-test predictions are written to `outputs/evaluation_results/`.

## Dataset and data-quality note

The source spreadsheet used for this project was supplied with the assignment, but its original publisher and a public download URL have not been verified. The data is intentionally not published because redistribution permission is unknown.

The suggested [rashida048/Datasets `StudentsPerformance.csv`](https://github.com/rashida048/Datasets/blob/master/StudentsPerformance.csv) is **not** the dataset used by this project. Its columns describe race/ethnicity, lunch, test preparation, and math/reading/writing scores. The project CSV instead has fields such as `student_id`, `study_time_hours`, `attendance_percent`, `previous_grade`, and `final_exam_score`; this workflow predicts `final_exam_score`. The suggested file therefore cannot be cited as the source for the current model or its results.

Obtain the assignment dataset from the instructor or its original provider, and only use/share it in accordance with its terms. Put an authorized compatible CSV at `data/student_performance_dataset.csv`, or pass its path with `--data`. The required columns are checked by the script; invalid numeric targets are excluded rather than imputed. The generated report summarizes the data actually used.

The `final_grade` field is excluded because it is derived from the target score and would leak the answer. `student_id` is an identifier, not a predictor. `gender` is also excluded so the model focuses on actionable study and access factors rather than a demographic attribute. The model describes associations in this dataset; it does not establish that any factor causes a score change or diagnose an individual student's abilities.

## Setup

Use Python 3.10 or later:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run

From the project root:

```bash
python src/model.py
```

Or open `notebooks/student_performance_analysis.ipynb` in Jupyter and run all cells. The script writes reproducible results to `outputs/` and builds `Project_Report.pdf`. Use `--data` to point to a different CSV and `--output` to select another output directory.

## Method

The workflow inspects and cleans the data, creates descriptive summaries and EDA charts, and holds out 20% of valid-target records for final testing (random seed 42). Numeric missing values are median-imputed and categorical missing values are mode-imputed inside scikit-learn pipelines, so imputation and one-hot encoding are fit only on training folds. Linear Regression is compared with Ridge Regression; Ridge's regularization strength is selected by 5-fold cross-validation on the training set. The test set is used once for final MAE, RMSE, and R-squared reporting.

## Generated outputs

After running the workflow locally, it creates:
- `outputs/visualizations/score_distribution.png`
- `outputs/visualizations/average_score_by_activity.png` — required bar chart.
- `outputs/visualizations/feature_relationships.png`
- `outputs/visualizations/score_by_parental_education_boxplot.png` — required box plot.
- `outputs/visualizations/correlation_heatmap.png` — required correlation heatmap.
- `outputs/visualizations/actual_vs_predicted.png`
- `outputs/visualizations/residuals.png`
- `outputs/visualizations/feature_coefficients.png`
- `outputs/evaluation_results/model_metrics.csv`
- `outputs/evaluation_results/test_predictions.csv`
- `outputs/evaluation_results/support_review_candidates.csv` — lowest 10 estimated marks in the held-out test sample, for supportive human follow-up only.
- `outputs/evaluation_results/data_quality.json`
- `screenshots/evaluation_and_prediction_output.png`
- `Project_Report.pdf`

It also writes the selected model to `models/student_score_model.joblib`. The five required EDA chart types are included: bar chart, histogram, scatter plots, box plot, and correlation heatmap. The generated PDF report follows the Minor Project's 13 required sections in order and uses placeholders for personal details. Generated files may contain information derived from student records and are ignored by Git; do not publish them without permission. The folder layout follows the guideline recommendation, including `src/model.py`; the longer analysis implementation is kept in `src/student_performance_analysis.py`.

## Limitations

The supplied spreadsheet has no source citation or context about how scores were collected. The 41 invalid target cells reduce the usable sample. Results may not generalize to other schools, cohorts, or grading systems. Some input factors may be self-reported or confounded. This project is intended for educational analysis and supportive discussion, not high-stakes student decisions.

## Student details

Add your name and other required student details to your own report before submission. Generated reports now use placeholders instead of personal identifiers. Do not publish a dataset, report, screenshots, model, or generated results unless you have permission to share them.
