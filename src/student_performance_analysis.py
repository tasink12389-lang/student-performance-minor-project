"""Student Performance Analysis and Prediction System.

Run from the project root with:
    python src/student_performance_analysis.py
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET = "final_exam_score"
NUMERIC_FEATURES = [
    "study_time_hours",
    "attendance_percent",
    "sleep_hours",
    "previous_grade",
]
CATEGORICAL_FEATURES = [
    "parental_education",
    "internet_access",
    "extracurricular_activities",
    "part_time_job",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
RANDOM_STATE = 42


def load_and_clean_data(csv_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load CSV, normalize column names, reject invalid targets, and deduplicate."""
    data = pd.read_csv(csv_path)
    data.columns = data.columns.astype(str).str.strip()
    required = set(FEATURES + [TARGET, "student_id"])
    missing_columns = sorted(required.difference(data.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    original_rows = len(data)
    duplicates = int(data.duplicated().sum())
    data = data.drop_duplicates().copy()
    scores = pd.to_numeric(data[TARGET], errors="coerce")
    invalid_target_rows = int(scores.isna().sum())
    data[TARGET] = scores
    data = data.loc[data[TARGET].notna()].copy()

    for column in NUMERIC_FEATURES:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    for column in CATEGORICAL_FEATURES:
        data[column] = data[column].astype("string").str.strip()
        data[column] = data[column].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
        data[column] = data[column].astype(object).where(data[column].notna(), np.nan)

    quality = {
        "source_rows": original_rows,
        "source_columns": int(len(data.columns)),
        "duplicate_rows_removed": duplicates,
        "invalid_target_rows_excluded": invalid_target_rows,
        "usable_rows": int(len(data)),
        "missing_values_after_target_cleaning": {
            str(k): int(v) for k, v in data[FEATURES + [TARGET]].isna().sum().items()
        },
        "target_min": float(data[TARGET].min()),
        "target_max": float(data[TARGET].max()),
        "target_mean": float(data[TARGET].mean()),
    }
    if data.empty:
        raise ValueError("No rows with valid numeric final examination scores remain.")
    return data, quality


def make_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def metric_row(name: str, actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    return {
        "model": name,
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
    }


def feature_names_and_coefficients(model: Pipeline) -> pd.DataFrame:
    names = model.named_steps["preprocessor"].get_feature_names_out()
    coefficients = model.named_steps["model"].coef_
    result = pd.DataFrame({"feature": names, "coefficient": coefficients})
    result["absolute_coefficient"] = result["coefficient"].abs()
    return result.sort_values("absolute_coefficient", ascending=False)


def create_visualizations(
    data: pd.DataFrame,
    actual: np.ndarray,
    predicted: np.ndarray,
    coefficients: pd.DataFrame,
    visual_dir: Path,
) -> None:
    visual_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data[TARGET], bins=25, kde=True, ax=ax, color="#4472C4")
    ax.set(title="Distribution of Final Examination Scores", xlabel="Final exam score", ylabel="Students")
    fig.tight_layout()
    fig.savefig(visual_dir / "score_distribution.png", dpi=160)
    plt.close(fig)

    category_scores = (
        data.groupby("extracurricular_activities", dropna=False)[TARGET]
        .mean()
        .sort_values(ascending=False)
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(category_scores.index.astype(str), category_scores.values, color="#55A868")
    ax.set(
        title="Average Exam Score by Extracurricular Participation",
        xlabel="Extracurricular activities",
        ylabel="Average final exam score",
    )
    fig.tight_layout()
    fig.savefig(visual_dir / "average_score_by_activity.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    education_order = ["None", "High School", "Bachelors", "Masters", "PhD"]
    sns.boxplot(
        data=data,
        x="parental_education",
        y=TARGET,
        order=[level for level in education_order if level in set(data["parental_education"])],
        ax=ax,
        color="#9ecae1",
    )
    ax.set(
        title="Final Exam Score by Parental Education",
        xlabel="Parental education",
        ylabel="Final exam score",
    )
    fig.tight_layout()
    fig.savefig(visual_dir / "score_by_parental_education_boxplot.png", dpi=160)
    plt.close(fig)

    correlation_columns = NUMERIC_FEATURES + [TARGET]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        data[correlation_columns].corr(),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        ax=ax,
    )
    ax.set_title("Correlation Heatmap of Numeric Academic Features")
    fig.tight_layout()
    fig.savefig(visual_dir / "correlation_heatmap.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, feature, label in zip(
        axes,
        ["study_time_hours", "attendance_percent", "previous_grade"],
        ["Study time (hours)", "Attendance (%)", "Previous grade"],
    ):
        sns.regplot(
            data=data,
            x=feature,
            y=TARGET,
            ax=ax,
            scatter_kws={"alpha": 0.35, "s": 18},
            line_kws={"color": "#C44E52"},
        )
        ax.set(xlabel=label, ylabel="Final exam score")
    fig.suptitle("Academic Factors and Final Examination Score")
    fig.tight_layout()
    fig.savefig(visual_dir / "feature_relationships.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(actual, predicted, alpha=0.65, color="#4472C4")
    low = float(min(actual.min(), predicted.min()))
    high = float(max(actual.max(), predicted.max()))
    ax.plot([low, high], [low, high], linestyle="--", color="#C44E52", label="Ideal prediction")
    ax.set(
        title="Actual vs. Predicted Final Scores",
        xlabel="Actual score",
        ylabel="Predicted score",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(visual_dir / "actual_vs_predicted.png", dpi=160)
    plt.close(fig)

    residuals = actual - predicted
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(predicted, residuals, alpha=0.65, color="#55A868")
    ax.axhline(0, linestyle="--", color="#C44E52")
    ax.set(title="Residual Analysis", xlabel="Predicted score", ylabel="Actual - predicted")
    fig.tight_layout()
    fig.savefig(visual_dir / "residuals.png", dpi=160)
    plt.close(fig)

    top = coefficients.head(12).sort_values("coefficient")
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#C44E52" if value < 0 else "#55A868" for value in top["coefficient"]]
    ax.barh(top["feature"], top["coefficient"], color=colors)
    ax.set(title="Largest Standardized Model Coefficients", xlabel="Coefficient")
    fig.tight_layout()
    fig.savefig(visual_dir / "feature_coefficients.png", dpi=160)
    plt.close(fig)

    evidence_dir = visual_dir.parent.parent / "screenshots"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    for source_name, evidence_name in [
        ("score_distribution.png", "score_distribution.png"),
        ("average_score_by_activity.png", "average_score_by_activity.png"),
        ("feature_relationships.png", "feature_relationships.png"),
        ("score_by_parental_education_boxplot.png", "score_by_parental_education_boxplot.png"),
        ("correlation_heatmap.png", "correlation_heatmap.png"),
        ("actual_vs_predicted.png", "actual_vs_predicted.png"),
        ("feature_coefficients.png", "feature_coefficients.png"),
    ]:
        shutil.copy2(visual_dir / source_name, evidence_dir / evidence_name)


def create_prediction_evidence(
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    evidence_dir: Path,
) -> None:
    evidence_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].axis("off")
    metric_table = axes[0].table(
        cellText=[
            [row.model, f"{row.mae:.3f}", f"{row.rmse:.3f}", f"{row.r2:.3f}"]
            for row in metrics.itertuples(index=False)
        ],
        colLabels=["Model", "MAE", "RMSE", "R²"],
        loc="center",
        cellLoc="center",
    )
    metric_table.auto_set_font_size(False)
    metric_table.set_fontsize(9)
    metric_table.scale(1, 1.8)
    axes[0].set_title("Held-out Test Evaluation")

    sample = predictions.head(8)[
        ["student_id", "actual_score", "predicted_score", "absolute_error"]
    ].copy()
    sample[["actual_score", "predicted_score", "absolute_error"]] = sample[
        ["actual_score", "predicted_score", "absolute_error"]
    ].round(2)
    axes[1].axis("off")
    prediction_table = axes[1].table(
        cellText=sample.values,
        colLabels=["Student ID", "Actual", "Predicted", "Abs. error"],
        loc="center",
        cellLoc="center",
    )
    prediction_table.auto_set_font_size(False)
    prediction_table.set_fontsize(8)
    prediction_table.scale(1, 1.6)
    axes[1].set_title("Sample Unseen-Test Predictions")
    fig.tight_layout()
    fig.savefig(evidence_dir / "evaluation_and_prediction_output.png", dpi=160)
    plt.close(fig)


def build_report(
    output_dir: Path,
    quality: dict[str, Any],
    descriptive: pd.DataFrame,
    metrics: pd.DataFrame,
    cv_rmse: float,
    best_alpha: float,
    selected_name: str,
    coefficient_table: pd.DataFrame,
    train_rows: int,
    test_rows: int,
) -> None:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        Image,
        KeepTogether,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    pdf_path = output_dir.parent / "Project_Report.pdf"
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ProjectTitle", parent=styles["Title"], alignment=TA_CENTER, leading=24))
    styles.add(ParagraphStyle(name="SmallNote", parent=styles["BodyText"], fontSize=8, leading=10))
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
    )
    story = [
        Paragraph("Student Performance Analysis and Prediction System", styles["ProjectTitle"]),
        Spacer(1, 12),
        Paragraph("<b>Minor Project | AI/ML with Python Training Program</b>", styles["Heading2"]),
        Spacer(1, 6),
        Paragraph("1. Project title and student details", styles["Heading1"]),
        Paragraph(
            "Student name: [Add name]. Registered email: [Add email]. "
            "Roll number: [Add roll number]. Program: AI/ML with Python. "
            "Batch: September 2026.",
            styles["BodyText"],
        ),
        Paragraph("2. Introduction and problem statement", styles["Heading1"]),
        Paragraph(
            "Students and educators can use historical patterns to discuss where additional "
            "academic support may be useful. The objective is to build a reproducible Python "
            "workflow that describes the data and estimates a student's final examination mark "
            "without using information derived from that mark.",
            styles["BodyText"],
        ),
        Paragraph("3. Project objectives", styles["Heading1"]),
        Paragraph(
            "Load and validate the dataset; handle invalid target values, duplicates, and missing "
            "features; summarize and visualize academic patterns; train and compare regression "
            "models; report MAE, RMSE, and R-squared on unseen data; and document limitations.",
            styles["BodyText"],
        ),
        Paragraph("4. Dataset description and source", styles["Heading1"]),
        Paragraph(
            f"The dataset has {quality['source_rows']:,} rows and {quality['source_columns']} columns. "
            "Its original publisher and public URL were not provided. The target is final_exam_score. "
            "Predictors are study_time_hours, attendance_percent, sleep_hours, previous_grade, "
            "parental_education, internet_access, extracurricular_activities, and part_time_job. "
            "student_id is retained only for traceability. gender and final_grade are excluded "
            "from model inputs; final_grade is derived from the score and would leak target information.",
            styles["BodyText"],
        ),
        Spacer(1, 6),
    ]

    quality_data = [["Data-quality check", "Result"]]
    quality_data.extend(
        [
            ["Source rows", str(quality["source_rows"])],
            ["Exact duplicates removed", str(quality["duplicate_rows_removed"])],
            ["Rows with nonnumeric target excluded", str(quality["invalid_target_rows_excluded"])],
            ["Rows used", str(quality["usable_rows"])],
            ["Target range", f'{quality["target_min"]:.1f} to {quality["target_max"]:.1f}'],
            ["Target mean", f'{quality["target_mean"]:.2f}'],
        ]
    )
    story.append(Table(quality_data, colWidths=[3.5 * inch, 2.5 * inch], style=TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#24476B")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F5F8")]),
        ("PAD", (0, 0), (-1, -1), 5),
    ])))
    story.extend(
        [
            Paragraph(
                f"Data-quality note: {quality['invalid_target_rows_excluded']} records with invalid "
                f"targets were excluded rather than imputed; {quality['usable_rows']} records remain. "
                f"Exact duplicate rows removed: {quality['duplicate_rows_removed']}. "
                f"Missing values in model features after target cleaning: "
                f"{sum(quality['missing_values_after_target_cleaning'][name] for name in FEATURES)}. "
                "Feature imputation is performed inside the training pipeline.",
                styles["SmallNote"],
            ),
            Paragraph("5. Data preprocessing and feature engineering", styles["Heading1"]),
            Paragraph(
                "Column names are normalized, numeric values are parsed, and blank categorical "
                "values are treated as missing. Numeric features are median-imputed and standardized; "
                "categorical features are mode-imputed and one-hot encoded. These transformations "
                "are inside a scikit-learn pipeline, so they are fit on training data only. No target "
                "or derived grade is used as an input. Gender is omitted to keep the model focused "
                "on actionable factors and reduce unnecessary demographic influence.",
                styles["BodyText"],
            ),
            Paragraph("6. Exploratory data analysis and visualizations", styles["Heading1"]),
        ]
    )
    for fname, caption in [
        ("score_distribution.png", "Figure 1. Distribution of valid final examination scores."),
        ("average_score_by_activity.png", "Figure 2. Bar chart of average scores by extracurricular participation."),
        ("feature_relationships.png", "Figure 3. Scatter plots of score relationships with study time, attendance, and previous grade."),
        ("score_by_parental_education_boxplot.png", "Figure 4. Score spread grouped by parental education."),
        ("correlation_heatmap.png", "Figure 5. Correlations among numeric academic variables; correlation is not causation."),
    ]:
        story.append(KeepTogether([Image(str(output_dir / "visualizations" / fname), width=6.9 * inch, height=3.0 * inch), Paragraph(caption, styles["SmallNote"])]))

    desc_rows = [["Measure", "Mean", "Std. dev.", "Minimum", "Maximum"]]
    for col in ["study_time_hours", "attendance_percent", "sleep_hours", "previous_grade", TARGET]:
        r = descriptive.loc[col]
        desc_rows.append([col, f'{r["mean"]:.2f}', f'{r["std"]:.2f}', f'{r["min"]:.2f}', f'{r["max"]:.2f}'])
    story.append(Table(desc_rows, colWidths=[1.9 * inch, 1.0 * inch, 1.0 * inch, 1.0 * inch, 1.0 * inch], style=TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#24476B")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("PAD", (0, 0), (-1, -1), 4),
    ])))
    story.extend(
        [
            Paragraph(
                "Interpretation: the scatter plots show observational associations in this sample. "
                "They should not be read as evidence that changing a single factor will cause a "
                "particular score change.",
                styles["BodyText"],
            ),
            Paragraph("7. Machine-learning methodology", styles["Heading1"]),
            Paragraph(
                "Valid-target records are split into 80% training and 20% test subsets using "
                "random_state=42. Linear Regression is evaluated as the required baseline. Ridge "
                "Regression is tuned over alpha values [0.01, 0.1, 1, 10, 100] with 5-fold "
                "cross-validation on training data only, using RMSE. The better cross-validated "
                "linear model is selected for final held-out evaluation.",
                styles["BodyText"],
            ),
            Paragraph("8. Model training and testing approach", styles["Heading1"]),
            Paragraph(
                f"The {quality['usable_rows']} records with numeric targets are split once into a "
                f"{train_rows}-row training subset and a {test_rows}-row test subset "
                "(80/20; random_state=42). "
                "Imputation, scaling, and one-hot encoding are fit inside each training "
                "fold and then applied to validation/test data. The test subset is held "
                "out for final evaluation and actual-versus-predicted examples.",
                styles["BodyText"],
            ),
        ]
    )
    metric_data = [["Model", "MAE", "RMSE", "R-squared"]]
    for _, row in metrics.iterrows():
        metric_data.append([row["model"], f'{row["mae"]:.3f}', f'{row["rmse"]:.3f}', f'{row["r2"]:.3f}'])
    story.extend(
        [
            Paragraph("9. Evaluation metrics and results", styles["Heading1"]),
            Paragraph(
                f"Selected model: <b>{selected_name}</b>. Ridge best alpha: {best_alpha:g}. "
                f"Training-set 5-fold CV RMSE for the tuned Ridge model: {cv_rmse:.3f}. "
                "Final test metrics use held-out records not used for fitting or tuning.",
                styles["BodyText"],
            ),
            Table(metric_data, colWidths=[2.6 * inch, 1.1 * inch, 1.1 * inch, 1.3 * inch], style=TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#24476B")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("PAD", (0, 0), (-1, -1), 5),
            ])),
            Paragraph("10. Actual versus predicted examination marks", styles["Heading1"]),
            Image(str(output_dir / "visualizations" / "actual_vs_predicted.png"), width=5.0 * inch, height=5.0 * inch),
            Paragraph("Figure 6. Held-out actual scores compared with model predictions.", styles["SmallNote"]),
            Image(str(output_dir / "visualizations" / "residuals.png"), width=6.2 * inch, height=4.2 * inch),
            Paragraph("Figure 7. Residuals (actual minus predicted) against predicted scores.", styles["SmallNote"]),
            Image(str(output_dir.parent / "screenshots" / "evaluation_and_prediction_output.png"), width=6.9 * inch, height=2.7 * inch),
            Paragraph("Table 2. Sample held-out predictions and model metrics.", styles["SmallNote"]),
            Paragraph("11. Interpretation of findings and academic performance patterns", styles["Heading1"]),
            Paragraph(
                "Use the residual plot and model coefficients to understand where predictions "
                "systematically over- or under-estimate. The largest model coefficients are shown "
                "in the final chart and coefficient CSV. The ten lowest predictions in the held-out "
                "test subset are saved separately as candidates for supportive human follow-up; "
                "they are not diagnoses or automatic labels. No pass/fail classifier is included "
                "because the required task is score regression and no validated pass threshold was supplied.",
                styles["BodyText"],
            ),
            Image(str(output_dir / "visualizations" / "feature_coefficients.png"), width=6.4 * inch, height=4.2 * inch),
            Paragraph("Figure 8. Largest standardized linear-model coefficients; category terms are one-hot encoded.", styles["SmallNote"]),
            Paragraph("12. Limitations and future improvements", styles["Heading1"]),
            Paragraph(
                "The dataset's provenance, collection process, grading scale, and cohort context "
                "are unknown. Forty-one target cells are malformed. A random split within one "
                "dataset does not test performance across schools or future cohorts. Associations "
                "may be confounded and model predictions have uncertainty. Future work should "
                "validate data provenance, collect a larger representative sample, evaluate fairness "
                "and subgroup errors with appropriate safeguards, test temporal or school-level "
                "generalization, and involve educators before any student-support use.",
                styles["BodyText"],
            ),
            Paragraph("13. Conclusion and references", styles["Heading1"]),
            Paragraph(
                f"The project provides a reproducible EDA and linear regression workflow. On "
                f"held-out data, the selected {selected_name} achieved MAE {metrics.loc[metrics['model'] == selected_name, 'mae'].iloc[0]:.3f}, "
                f"RMSE {metrics.loc[metrics['model'] == selected_name, 'rmse'].iloc[0]:.3f}, and "
                f"R-squared {metrics.loc[metrics['model'] == selected_name, 'r2'].iloc[0]:.3f}. "
                "These results are descriptive of this sample only and should be used to support "
                "learning discussions rather than high-stakes decisions.",
                styles["BodyText"],
            ),
            Paragraph(
                "References: dataset supplied by the student for this assignment; no original "
                "publisher or public URL was supplied. Python, pandas, NumPy, Matplotlib, "
                "seaborn, and scikit-learn documentation.",
                styles["SmallNote"],
            ),
            Spacer(1, 8),
            Paragraph("Appendix: implementation and reproduction", styles["Heading1"]),
            Paragraph(
                "Install dependencies with `pip install -r requirements.txt`; run "
                "`python src/student_performance_analysis.py`; or run all cells in "
                "`notebooks/student_performance_analysis.ipynb`. Outputs include the evaluation "
                "CSV, test predictions, data-quality JSON, coefficient table, and visualization PNG files. "
                "The selected fitted preprocessing-and-regression pipeline is saved to "
                "`models/student_score_model.joblib`.",
                styles["BodyText"],
            ),
            Paragraph("Student details are intentionally placeholders; complete them before submission.", styles["SmallNote"]),
        ]
    )
    doc.build(story)


def run_project(csv_path: Path, output_dir: Path) -> dict[str, Any]:
    data, quality = load_and_clean_data(csv_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = output_dir / "evaluation_results"
    eval_dir.mkdir(parents=True, exist_ok=True)
    descriptive = data[NUMERIC_FEATURES + [TARGET]].describe().T
    descriptive.to_csv(eval_dir / "descriptive_statistics.csv", index_label="feature")
    data[FEATURES + [TARGET]].corr(numeric_only=True).to_csv(eval_dir / "numeric_correlation_matrix.csv")
    (eval_dir / "data_quality.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")

    x_train, x_test, y_train, y_test = train_test_split(
        data[FEATURES],
        data[TARGET],
        test_size=0.2,
        random_state=RANDOM_STATE,
    )
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    linear_pipeline = Pipeline(
        steps=[("preprocessor", make_preprocessor()), ("model", LinearRegression())]
    )
    linear_pipeline.fit(x_train, y_train)
    linear_cv_rmse = -cross_val_score(
        Pipeline([("preprocessor", make_preprocessor()), ("model", LinearRegression())]),
        x_train,
        y_train,
        cv=cv,
        scoring="neg_root_mean_squared_error",
    ).mean()

    ridge_pipeline = Pipeline(
        steps=[("preprocessor", make_preprocessor()), ("model", Ridge())]
    )
    search = GridSearchCV(
        ridge_pipeline,
        param_grid={"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=1,
        refit=True,
    )
    search.fit(x_train, y_train)
    best_ridge = search.best_estimator_
    ridge_cv_rmse = float(-search.best_score_)
    selected_name = "Ridge Regression" if ridge_cv_rmse < linear_cv_rmse else "Linear Regression"
    selected_model = best_ridge if selected_name == "Ridge Regression" else linear_pipeline
    models_dir = output_dir.parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    dump(selected_model, models_dir / "student_score_model.joblib")

    linear_pred = linear_pipeline.predict(x_test)
    ridge_pred = best_ridge.predict(x_test)
    metrics = pd.DataFrame(
        [
            metric_row("Linear Regression", y_test.to_numpy(), linear_pred),
            metric_row("Ridge Regression", y_test.to_numpy(), ridge_pred),
        ]
    )
    metrics["selected_for_interpretation"] = metrics["model"].eq(selected_name)
    metrics.to_csv(eval_dir / "model_metrics.csv", index=False)
    chosen_pred = selected_model.predict(x_test)
    predictions = pd.DataFrame(
        {
            "student_id": data.loc[x_test.index, "student_id"].to_numpy(),
            "actual_score": y_test.to_numpy(),
            "predicted_score": chosen_pred,
            "absolute_error": np.abs(y_test.to_numpy() - chosen_pred),
        }
    )
    predictions.to_csv(eval_dir / "test_predictions.csv", index=False)
    evidence_dir = output_dir.parent / "screenshots"
    create_prediction_evidence(metrics, predictions, evidence_dir)
    review_candidates = (
        predictions.sort_values("predicted_score")
        .head(10)[["student_id", "predicted_score"]]
        .reset_index(drop=True)
    )
    review_candidates.insert(0, "review_rank", np.arange(1, len(review_candidates) + 1))
    review_candidates.to_csv(eval_dir / "support_review_candidates.csv", index=False)
    coefficients = feature_names_and_coefficients(selected_model)
    coefficients.to_csv(eval_dir / "model_coefficients.csv", index=False)

    create_visualizations(
        data,
        y_test.to_numpy(),
        chosen_pred,
        coefficients,
        output_dir / "visualizations",
    )
    build_report(
        output_dir=output_dir,
        quality=quality,
        descriptive=descriptive,
        metrics=metrics,
        cv_rmse=ridge_cv_rmse,
        best_alpha=float(search.best_params_["model__alpha"]),
        selected_name=selected_name,
        coefficient_table=coefficients,
        train_rows=len(x_train),
        test_rows=len(x_test),
    )
    result = {
        "usable_rows": len(data),
        "selected_model": selected_name,
        "ridge_best_alpha": float(search.best_params_["model__alpha"]),
        "linear_cv_rmse": float(linear_cv_rmse),
        "ridge_cv_rmse": ridge_cv_rmse,
        "saved_model": str(models_dir / "student_score_model.joblib"),
        "metrics": metrics.to_dict(orient="records"),
    }
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--data", type=Path, default=default_root / "data" / "student_performance_dataset.csv")
    parser.add_argument("--output", type=Path, default=default_root / "outputs")
    args = parser.parse_args()
    if not args.data.is_file():
        parser.error(
            f"dataset not found: {args.data}. The dataset is not included; "
            "provide an authorized CSV with --data."
        )
    run_project(args.data, args.output)


if __name__ == "__main__":
    main()
