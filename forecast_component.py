import pandas as pd
import numpy as np
import joblib
from pathlib import Path


STUDENT_PREDICTION_COLUMNS = [
    "student_id",
    "course_code",
    "current_is_at_risk",
    "current_risk_probability",
    "six_month_is_at_risk",
    "six_month_risk_probability",
    "final_exam_is_at_risk",
    "final_exam_risk_probability",
]


class MentoringDemandForecaster:

    def __init__(
        self,
        model_path,
        feature_path,
        threshold=0.65
    ):
        self.model = joblib.load(model_path)
        self.feature_names = joblib.load(feature_path)
        self.threshold = threshold

    def forecast(
        self,
        student_df,
        student_ids=None,
        course_codes=None,
        output_path=None,
    ):

        df = student_df.copy()
        if student_ids is not None and len(student_ids) != len(df):
            raise ValueError(
                "The number of student IDs must match the number of feature rows."
            )
        if course_codes is not None and len(course_codes) != len(df):
            raise ValueError(
                "The number of course codes must match the number of feature rows."
            )
        required_features = set(self.feature_names) | {
            "engagement_change",
            "assessment_score_trend",
            "inactive_days",
            "missed_assessment_count",
            "upcoming_assessment_count",
            "assessment_completion_rate",
            "historical_course_failure_rate",
        }
        missing_features = sorted(required_features - set(df.columns))
        if missing_features:
            raise ValueError(
                "Student dataset is missing required features: "
                + ", ".join(missing_features)
            )

        durations = pd.to_numeric(
            df["course_duration_days"],
            errors="coerce"
        ).fillna(0).clip(lower=0)

        # Features are engineered at 50% of the course duration.
        remaining_days = durations * 0.5
        six_month_days = np.minimum(180, remaining_days)
        end_of_course_days = remaining_days

        baseline_probs = self._predict_probabilities(df)
        six_month_frame = self._project_features(df, six_month_days)
        end_of_course_frame = self._project_features(df, end_of_course_days)

        six_month_probs = self._predict_probabilities(six_month_frame)
        end_of_course_probs = self._predict_probabilities(end_of_course_frame)

        baseline = self._summarize(baseline_probs)
        six_month = self._summarize(six_month_probs)
        end_of_course = self._summarize(end_of_course_probs)
        if student_ids is None:
            student_ids = [str(index) for index in df.index]
        else:
            student_ids = [
                str(student_id) if pd.notna(student_id) else str(index)
                for index, student_id in enumerate(student_ids)
            ]

        student_forecasts = []
        if course_codes is None:
            course_codes = [None] * len(df)
        course_columns = [
            column for column in df.columns
            if column.startswith("code_module_")
        ]
        for row_index, student_id in enumerate(student_ids):
            course_code = course_codes[row_index]
            if pd.isna(course_code):
                course_code = None
            if course_code is None and course_columns:
                active_courses = [
                    column.removeprefix("code_module_")
                    for column in course_columns
                    if pd.to_numeric(
                        df.iloc[row_index][column],
                        errors="coerce"
                    ) > 0
                ]
                if len(active_courses) == 1:
                    course_code = active_courses[0]

            current_risk = int(
                baseline_probs[row_index] >= self.threshold
            )
            six_month_risk = int(
                six_month_probs[row_index] >= self.threshold
            )
            final_exam_risk = int(
                end_of_course_probs[row_index] >= self.threshold
            )
            student_forecasts.append({
                "student_id": str(student_id),
                "course_code": course_code,
                "baseline_risk_probability": round(
                    float(baseline_probs[row_index]), 4
                ),
                "baseline_is_at_risk": bool(current_risk),
                "current_is_at_risk": current_risk,
                "current_risk_probability": round(
                    float(baseline_probs[row_index]), 4
                ),
                "six_month_risk_probability": round(
                    float(six_month_probs[row_index]), 4
                ),
                "six_month_is_at_risk": six_month_risk,
                "course_end_risk_probability": round(
                    float(end_of_course_probs[row_index]), 4
                ),
                "course_end_is_at_risk": bool(final_exam_risk),
                "final_exam_is_at_risk": final_exam_risk,
                "final_exam_risk_probability": round(
                    float(end_of_course_probs[row_index]), 4
                ),
            })

        six_month_growth = round(
            (
                (six_month["at_risk_students"] - baseline["at_risk_students"])
                / max(baseline["at_risk_students"], 1)
            ) * 100,
            2
        )
        final_growth = round(
            (
                (end_of_course["at_risk_students"] - baseline["at_risk_students"])
                / max(baseline["at_risk_students"], 1)
            ) * 100,
            2
        )

        result = {
            "total_students": len(df),
            "forecast_type": "scenario_projection_using_existing_classifier",
            "risk_threshold": self.threshold,
            "assumptions": [
                "The input is the engineered mid-course snapshot, not a course-start snapshot.",
                "Observed assessment-score and engagement trends continue at their current rates.",
                "Assessment completion behavior remains consistent with the observed completion rate.",
                "Future features are estimates; this classifier was trained on final outcomes, not separate 6- or 12-month labels.",
            ],
            "student_forecasts": student_forecasts,
            "baseline_snapshot": baseline,
            "after_6_months": {
                **six_month,
                "average_days_projected": round(float(np.mean(six_month_days)), 1),
                "capped_at_course_end": bool((remaining_days < 180).any()),
            },
            "after_12_months_or_course_end": {
                **end_of_course,
                "average_days_projected": round(float(np.mean(end_of_course_days)), 1),
                "course_end_is_used_when_sooner": True,
            },
            # Keep these keys for the existing mentor recommendation endpoint.
            "current_at_risk_students": baseline["at_risk_students"],
            "forecast_after_6_months": six_month["at_risk_students"],
            "forecast_before_final_exam": end_of_course["at_risk_students"],
            "forecast_after_12_months": end_of_course["at_risk_students"],
            "six_month_growth_percent": six_month_growth,
            "final_exam_growth_percent": final_growth,
        }
        if output_path is not None:
            self.export_student_predictions(result, output_path)
        return result

    @staticmethod
    def export_student_predictions(forecast_result, output_path):
        """Write the standardized student prediction CSV without changing API output."""
        rows = forecast_result.get("student_forecasts")
        if not isinstance(rows, list):
            raise ValueError(
                "Forecast result must contain a student_forecasts list."
            )
        output_rows = []
        for index, student in enumerate(rows):
            student_id = student.get("student_id")
            if student_id is None or pd.isna(student_id) or not str(student_id).strip():
                student_id = str(index)
            course_code = student.get("course_code")
            if course_code is not None and pd.isna(course_code):
                course_code = None
            output_rows.append({
                "student_id": str(student_id),
                "course_code": course_code,
                "current_is_at_risk": int(
                    student.get(
                        "current_is_at_risk",
                        student.get("baseline_is_at_risk", False),
                    )
                ),
                "current_risk_probability": float(
                    student.get(
                        "current_risk_probability",
                        student["baseline_risk_probability"],
                    )
                ),
                "six_month_is_at_risk": int(student["six_month_is_at_risk"]),
                "six_month_risk_probability": float(
                    student["six_month_risk_probability"]
                ),
                "final_exam_is_at_risk": int(
                    student.get(
                        "final_exam_is_at_risk",
                        student["course_end_is_at_risk"],
                    )
                ),
                "final_exam_risk_probability": float(
                    student.get(
                        "final_exam_risk_probability",
                        student["course_end_risk_probability"],
                    )
                ),
            })
        unique_rows = {}
        for row in output_rows:
            existing = unique_rows.get(row["student_id"])
            if existing is None:
                unique_rows[row["student_id"]] = row
                continue
            has_higher_final_exam_risk = (
                row["final_exam_risk_probability"]
                > existing["final_exam_risk_probability"]
            )
            for probability_column, classification_column in (
                ("current_risk_probability", "current_is_at_risk"),
                ("six_month_risk_probability", "six_month_is_at_risk"),
                ("final_exam_risk_probability", "final_exam_is_at_risk"),
            ):
                row_risk = row[probability_column]
                if row_risk > existing[probability_column]:
                    existing[probability_column] = row_risk
                existing[classification_column] = max(
                    existing[classification_column],
                    row[classification_column],
                )
            if has_higher_final_exam_risk:
                existing["course_code"] = row["course_code"]
        output = pd.DataFrame(
            list(unique_rows.values()),
            columns=STUDENT_PREDICTION_COLUMNS,
        )
        for probability_column in (
            "current_risk_probability",
            "six_month_risk_probability",
            "final_exam_risk_probability",
        ):
            probabilities = pd.to_numeric(
                output[probability_column],
                errors="coerce",
            )
            if (
                probabilities.isna().any()
                or not np.isfinite(probabilities.to_numpy()).all()
                or ((probabilities < 0) | (probabilities > 1)).any()
            ):
                raise ValueError(
                    f"Invalid probability values in '{probability_column}'."
                )
            output[probability_column] = probabilities
        for classification_column in (
            "current_is_at_risk",
            "six_month_is_at_risk",
            "final_exam_is_at_risk",
        ):
            if not output[classification_column].isin([0, 1]).all():
                raise ValueError(
                    f"Invalid binary values in '{classification_column}'."
                )

        path = Path(output_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(path, index=False, columns=STUDENT_PREDICTION_COLUMNS)
        return path

    def _predict_probabilities(self, features):
        return self.model.predict_proba(
            features[self.feature_names].to_numpy()
        )[:, 1]

    def _project_features(self, student_df, days_ahead):
        projected = student_df.copy()
        days_ahead = np.asarray(days_ahead, dtype=float)
        course_days = pd.to_numeric(
            student_df["course_duration_days"],
            errors="coerce"
        ).fillna(0).to_numpy(dtype=float)
        course_days = np.maximum(course_days, 0)

        score_change = (
            pd.to_numeric(
                student_df["assessment_score_trend"],
                errors="coerce"
            ).fillna(0).to_numpy(dtype=float)
            * days_ahead
        )
        for column in (
            "latest_assessment_score",
            "average_assessment_score",
            "assessment_weighted_score",
        ):
            if column in projected.columns:
                projected[column] = (
                    pd.to_numeric(projected[column], errors="coerce")
                    .fillna(0)
                    .to_numpy(dtype=float)
                    + score_change
                ).clip(0, 100)

        window_days = np.maximum(course_days / 8, 1)
        engagement_delta = pd.to_numeric(
            student_df["engagement_change"],
            errors="coerce"
        ).fillna(0).to_numpy(dtype=float)
        average_clicks = pd.to_numeric(
            student_df["average_daily_clicks"],
            errors="coerce"
        ).fillna(0).to_numpy(dtype=float)
        projected_daily_clicks = np.maximum(
            0,
            average_clicks + engagement_delta * days_ahead / window_days
        )

        if "average_daily_clicks" in projected.columns:
            projected["average_daily_clicks"] = projected_daily_clicks
        if "recent_clicks" in projected.columns:
            projected["recent_clicks"] = projected_daily_clicks * window_days
        if "total_clicks" in projected.columns:
            projected["total_clicks"] = (
                pd.to_numeric(projected["total_clicks"], errors="coerce")
                .fillna(0)
                .to_numpy(dtype=float)
                + projected_daily_clicks * days_ahead
            )
        if "active_days" in projected.columns:
            baseline_active_days = pd.to_numeric(
                student_df["active_days"],
                errors="coerce"
            ).fillna(0).to_numpy(dtype=float)
            observed_days = np.maximum(course_days * 0.5, 1)
            projected["active_days"] = np.minimum(
                course_days,
                baseline_active_days + baseline_active_days / observed_days * days_ahead
            )

        due_before_snapshot = pd.to_numeric(
            student_df["assessments_due_by_cutoff"],
            errors="coerce"
        ).fillna(0).to_numpy(dtype=float)
        missed_before_snapshot = pd.to_numeric(
            student_df["missed_assessment_count"],
            errors="coerce"
        ).fillna(0).to_numpy(dtype=float)
        schedule_density = pd.to_numeric(
            student_df["assessment_schedule_density"],
            errors="coerce"
        ).fillna(0).to_numpy(dtype=float)
        future_assessments = schedule_density * days_ahead
        historical_miss_rate = np.divide(
            missed_before_snapshot,
            due_before_snapshot,
            out=np.zeros_like(missed_before_snapshot),
            where=due_before_snapshot > 0
        ).clip(0, 1)
        projected_missed = (
            missed_before_snapshot + future_assessments * historical_miss_rate
        )
        projected_due = due_before_snapshot + future_assessments

        projected["missed_assessment_count"] = projected_missed
        projected["assessments_due_by_cutoff"] = projected_due
        projected["assessment_completion_rate"] = np.divide(
            (projected_due - projected_missed) * 100,
            projected_due,
            out=np.zeros_like(projected_due),
            where=projected_due > 0
        ).clip(0, 100)

        remaining_after_projection = np.maximum(
            0,
            course_days * 0.5 - days_ahead
        )
        projected["upcoming_assessment_count"] = (
            schedule_density * np.minimum(60, remaining_after_projection)
        )

        return projected

    def _summarize(self, probabilities):
        at_risk = int((probabilities >= self.threshold).sum())
        total = len(probabilities)
        return {
            "at_risk_students": at_risk,
            "at_risk_percentage": round(
                (at_risk / total * 100) if total else 0,
                2
            ),
            "average_risk_probability": round(
                float(np.mean(probabilities)) if total else 0,
                4
            ),
        }