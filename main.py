from pathlib import Path

import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from forecast_component import MentoringDemandForecaster
from mentor_management import MentorManagement

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

# =====================================================
# APP
# =====================================================

app = FastAPI(
    title="Predictive Mentor Capacity Balancer",
    version="1.0.0"
)

# =====================================================
# LOAD MODELS
# =====================================================

model_path = MODEL_DIR / "risk_model.pkl"
feature_path = MODEL_DIR / "risk_features.pkl"
if not model_path.is_file() or not feature_path.is_file():
    raise FileNotFoundError(
        f"Model artifacts are missing from {MODEL_DIR}. "
        "Run `uv run python model_train.py` from the project directory first."
    )

forecaster = MentoringDemandForecaster(
    model_path=model_path,
    feature_path=feature_path,
)
risk_model = forecaster.model
risk_features = forecaster.feature_names

# =====================================================
# REQUEST SCHEMA
# =====================================================

class StudentPredictionRequest(BaseModel):
    """Shared request body accepted by both student prediction endpoints."""

    model_config = ConfigDict(extra="allow")

    student_id: str | int | None = None
    id_student: str | int | None = None
    code_module: str | None = None
    course_code: str | None = None
    code_presentation: str | None = None
    features: dict[str, object] | None = None
    students: list[dict[str, object]] | None = Field(default=None, min_length=1)
    student_dataset_path: str | None = None


# Backward-compatible request names; both routes use the same schema.
RiskPredictionRequest = StudentPredictionRequest
ForecastRequest = StudentPredictionRequest


class MentorRequest(BaseModel):

    student_dataset_path: str
    mentor_dataset_path: str


def read_dataset(dataset_path: str) -> pd.DataFrame:
    path = Path(dataset_path).expanduser()
    if not path.is_absolute():
        path = BASE_DIR / path
    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"Dataset file not found: {path}",
        )
    return pd.read_csv(path)


MODEL_CATEGORICAL_COLUMNS = (
    "code_module",
    "gender",
    "region",
    "highest_education",
    "imd_band",
    "age_band",
    "disability",
)
MODEL_NUMERIC_FEATURES = [
    feature for feature in risk_features
    if not any(
        feature.startswith(f"{column}_")
        for column in MODEL_CATEGORICAL_COLUMNS
    )
]
STUDENT_DATASET_COLUMNS = (
    set(risk_features)
    | set(MODEL_CATEGORICAL_COLUMNS)
    | {
        "id_student",
        "student_id",
        "code_module",
        "course_code",
        "code_presentation",
    }
)


def read_student_dataset(dataset_path: str) -> pd.DataFrame:
    path = Path(dataset_path).expanduser()
    if not path.is_absolute():
        path = BASE_DIR / path
    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"Dataset file not found: {path}",
        )
    try:
        return pd.read_csv(
            path,
            usecols=lambda column: column in STUDENT_DATASET_COLUMNS,
        )
    except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError) as error:
        raise HTTPException(
            status_code=422,
            detail=f"Could not read student dataset '{path}': {error}",
        ) from error


def prepare_model_features(
    student_df: pd.DataFrame,
    *,
    strict_values: bool = False,
) -> pd.DataFrame:
    categorical_columns = [
        column
        for column in MODEL_CATEGORICAL_COLUMNS
        if column in student_df.columns
    ]
    encoded_feature_columns = {
        feature
        for feature in risk_features
        if any(
            feature.startswith(f"{column}_")
            for column in categorical_columns
        )
    }
    source_columns = [
        feature
        for feature in risk_features
        if feature in student_df.columns
        and feature not in encoded_feature_columns
    ]
    source = student_df.loc[:, source_columns]
    if categorical_columns:
        categorical_data = student_df.loc[:, categorical_columns].copy()
        for column in categorical_columns:
            categorical_data[column] = (
                categorical_data[column].fillna("Unknown").astype(str)
            )
        encoded_features = pd.get_dummies(
            categorical_data,
            columns=categorical_columns,
            dtype=int,
        )
        source = pd.concat([source, encoded_features], axis=1)

    missing_features = sorted(
        set(MODEL_NUMERIC_FEATURES) - set(source.columns)
    )
    if missing_features:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Student dataset is missing required numeric model features.",
                "missing_features": missing_features,
            },
        )

    model_input = source.reindex(
        columns=risk_features,
        fill_value=0,
    )
    model_features = model_input.apply(
        pd.to_numeric,
        errors="coerce",
    )
    if strict_values:
        invalid_values = []
        for feature in risk_features:
            original = model_input[feature]
            converted = model_features[feature]
            invalid_mask = original.notna() & converted.isna()
            invalid_mask |= converted.notna() & ~np.isfinite(converted)
            if invalid_mask.any():
                invalid_values.append(feature)
        if invalid_values:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "Student input contains invalid numeric model features.",
                    "invalid_features": invalid_values,
                },
            )
    model_features = model_features.replace(
        [np.inf, -np.inf],
        np.nan,
    ).fillna(0)
    return model_features


def request_student_rows(request: StudentPredictionRequest):
    """Extract flat or legacy nested student records from the common body."""
    payload = request.model_dump(exclude_unset=True)
    extra_fields = getattr(request, "__pydantic_extra__", None) or {}
    payload.update(extra_fields)
    rows = payload.get("students")
    if rows is not None:
        if not rows:
            raise HTTPException(
                status_code=422,
                detail="The students list must contain at least one student.",
            )
        return rows
    if payload.get("student_dataset_path") is not None:
        return None

    features = payload.get("features")
    row = dict(features) if isinstance(features, dict) else {}
    structural_fields = {
        "student_id",
        "id_student",
        "code_module",
        "course_code",
        "code_presentation",
        "features",
        "students",
        "student_dataset_path",
    }
    row.update(
        {
            key: value
            for key, value in payload.items()
            if key not in structural_fields
        }
    )
    for field in ("student_id", "id_student", "code_module", "course_code",
                  "code_presentation"):
        if field in payload:
            row[field] = payload[field]
    return [row]


def has_inline_student_data(request: StudentPredictionRequest) -> bool:
    payload = request.model_dump(exclude_unset=True)
    payload.update(getattr(request, "__pydantic_extra__", None) or {})
    return any(
        key != "student_dataset_path" and value is not None
        for key, value in payload.items()
    )


def normalize_forecast_rows(rows: list[dict[str, object]]):
    student_records = []
    for index, row in enumerate(rows):
        embedded_features = row.get("features")
        record = (
            dict(embedded_features)
            if isinstance(embedded_features, dict)
            else dict(row)
        )
        student_id = row.get("student_id")
        if student_id is None:
            student_id = row.get("id_student")
        if student_id is None:
            student_id = record.get("id_student")
        if student_id is None:
            student_id = index
        course_code = row.get("course_code")
        if course_code is None:
            course_code = record.get("code_module")
        record["student_id"] = str(student_id)
        record["course_code"] = course_code
        student_records.append(record)

    student_df = pd.DataFrame.from_records(student_records)
    student_ids = []
    course_codes = []
    for index, row in enumerate(student_records):
        student_id = row.get("student_id")
        if student_id is None or pd.isna(student_id) or not str(student_id).strip():
            student_id = row.get("id_student")
        if student_id is None or pd.isna(student_id) or not str(student_id).strip():
            student_id = index
        course_code = row.get("course_code")
        if course_code is None or pd.isna(course_code):
            course_code = row.get("code_module")
        student_ids.append(str(student_id))
        course_codes.append(
            course_code if course_code is not None and pd.notna(course_code) else None
        )
    student_df = student_df.drop(
        columns=["student_id", "course_code"],
        errors="ignore",
    )
    return student_df, student_ids, course_codes


def dataset_identifiers(student_df: pd.DataFrame):
    id_column = (
        "id_student"
        if "id_student" in student_df.columns
        else "student_id"
        if "student_id" in student_df.columns
        else None
    )
    base_ids = (
        student_df[id_column].astype(str).tolist()
        if id_column
        else [str(index) for index in range(len(student_df))]
    )
    course_codes = (
        student_df["code_module"].tolist()
        if "code_module" in student_df.columns
        else (
            student_df["course_code"].tolist()
            if "course_code" in student_df.columns
            else [None] * len(student_df)
        )
    )
    course_codes = [
        course if pd.notna(course) else None
        for course in course_codes
    ]
    return base_ids, course_codes


def summarize_chunk_forecasts(chunk_results, total_students):
    student_forecasts = []
    for chunk in chunk_results:
        student_forecasts.extend(chunk.pop("student_forecasts"))

    def summarize(probability_key, risk_key):
        probabilities = np.fromiter(
            (student[probability_key] for student in student_forecasts),
            dtype=float,
            count=len(student_forecasts),
        )
        at_risk = sum(student[risk_key] for student in student_forecasts)
        return {
            "at_risk_students": int(at_risk),
            "at_risk_percentage": round(
                at_risk / total_students * 100 if total_students else 0,
                2,
            ),
            "average_risk_probability": round(
                float(probabilities.mean()) if probabilities.size else 0,
                4,
            ),
        }

    baseline = summarize(
        "baseline_risk_probability",
        "baseline_is_at_risk",
    )
    six_month = summarize(
        "six_month_risk_probability",
        "six_month_is_at_risk",
    )
    course_end = summarize(
        "course_end_risk_probability",
        "course_end_is_at_risk",
    )
    six_month_days = sum(
        chunk["after_6_months"]["average_days_projected"]
        * chunk["total_students"]
        for chunk in chunk_results
    ) / total_students
    course_end_days = sum(
        chunk["after_12_months_or_course_end"]["average_days_projected"]
        * chunk["total_students"]
        for chunk in chunk_results
    ) / total_students
    summary = {
        "total_students": total_students,
        "forecast_type": chunk_results[0]["forecast_type"],
        "risk_threshold": chunk_results[0]["risk_threshold"],
        "student_forecasts": student_forecasts,
        "baseline_snapshot": baseline,
        "after_6_months": {
            **six_month,
            "average_days_projected": round(six_month_days, 1),
            "capped_at_course_end": any(
                chunk["after_6_months"]["capped_at_course_end"]
                for chunk in chunk_results
            ),
        },
        "after_12_months_or_course_end": {
            **course_end,
            "average_days_projected": round(course_end_days, 1),
            "course_end_is_used_when_sooner": True,
        },
        "current_at_risk_students": baseline["at_risk_students"],
        "forecast_after_6_months": six_month["at_risk_students"],
        "forecast_before_final_exam": course_end["at_risk_students"],
        "forecast_after_12_months": course_end["at_risk_students"],
        "six_month_growth_percent": round(
            (six_month["at_risk_students"] - baseline["at_risk_students"])
            / max(baseline["at_risk_students"], 1)
            * 100,
            2,
        ),
        "final_exam_growth_percent": round(
            (course_end["at_risk_students"] - baseline["at_risk_students"])
            / max(baseline["at_risk_students"], 1)
            * 100,
            2,
        ),
    }
    if "assumptions" in chunk_results[0]:
        summary["assumptions"] = chunk_results[0]["assumptions"]
    return summary


def forecast_in_chunks(student_df, student_ids, course_codes, chunk_size=10_000):
    if student_df.empty:
        raise HTTPException(
            status_code=422,
            detail="Student dataset must contain at least one row.",
        )
    chunks = []
    for start in range(0, len(student_df), chunk_size):
        end = start + chunk_size
        chunks.append(
            forecaster.forecast(
                student_df.iloc[start:end],
                student_ids=student_ids[start:end],
                course_codes=course_codes[start:end],
            )
        )
    return summarize_chunk_forecasts(chunks, len(student_df))


def add_capacity_plans(results, mentor_df):
    manager = MentorManagement(mentor_df)
    capacity = manager.calculate_capacity()
    forecast_periods = {
        "current": (
            "baseline_is_at_risk",
            "baseline_risk_probability",
        ),
        "after_6_months": (
            "six_month_is_at_risk",
            "six_month_risk_probability",
        ),
        "after_12_months_or_course_end": (
            "course_end_is_at_risk",
            "course_end_risk_probability",
        ),
    }
    try:
        plans = {
            period: manager.build_capacity_plan(
                results["student_forecasts"],
                period=period,
                risk_key=risk_key,
                probability_key=probability_key,
            )
            for period, (risk_key, probability_key) in forecast_periods.items()
        }
    except ValueError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error

    plans["before_final_exam"] = plans["after_12_months_or_course_end"]
    results["mentor_capacity_plans"] = plans
    results["mentor_capacity"] = capacity
    results["mentor_allocation"] = plans[
        "after_12_months_or_course_end"
    ]["allocation"]
    return manager, plans

# =====================================================
# INTRO
# =====================================================
@app.get("/")
def root():
    return {
        "message": "Welcome to the Predictive Mentor Capacity Balancer API. "
                   "Use the /docs endpoint for API documentation."
    }

# =====================================================
# HEALTH
# =====================================================

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# =====================================================
# RISK PREDICTION
# =====================================================

@app.post("/predict-risk")
def predict_risk(request: RiskPredictionRequest):
    if request.student_dataset_path:
        student_df = read_student_dataset(request.student_dataset_path)
        student_ids, course_codes = dataset_identifiers(student_df)
        strict_values = False
    else:
        rows = request_student_rows(request)
        student_df, student_ids, course_codes = normalize_forecast_rows(rows)
        strict_values = True

    try:
        model_features = prepare_model_features(
            student_df,
            strict_values=strict_values,
        )
    except HTTPException:
        raise

    probabilities = risk_model.predict_proba(model_features.to_numpy())[:, 1]
    student_predictions = [
        {
            "student_id": student_id,
            "course_code": course_code,
            "risk_probability": round(float(probability), 4),
            "is_at_risk": int(probability >= forecaster.threshold),
        }
        for student_id, course_code, probability in zip(
            student_ids,
            course_codes,
            probabilities,
        )
    ]
    if len(student_predictions) == 1:
        return {
            "risk_probability": student_predictions[0]["risk_probability"],
            "is_at_risk": student_predictions[0]["is_at_risk"],
        }
    return {
        "total_students": len(student_predictions),
        "student_predictions": student_predictions,
        "at_risk_students": sum(
            prediction["is_at_risk"]
            for prediction in student_predictions
        ),
    }


# =====================================================
# FORECAST DEMAND
# =====================================================

@app.post("/forecast-demand")
def forecast_demand(request: ForecastRequest):

    if request.student_dataset_path:
        if has_inline_student_data(request):
            raise HTTPException(
                status_code=422,
                detail="Provide either students or student_dataset_path, not both.",
            )
        students_df = read_student_dataset(request.student_dataset_path)
        student_ids, course_codes = dataset_identifiers(students_df)
        strict_values = False
    else:
        rows = request_student_rows(request)
        students_df, student_ids, course_codes = normalize_forecast_rows(
            rows
        )
        strict_values = True
    if request.students is not None and len(student_ids) != len(set(student_ids)):
        raise HTTPException(
            status_code=422,
            detail="Each student_id must be unique within the request.",
        )

    try:
        model_features = prepare_model_features(
            students_df,
            strict_values=strict_values,
        )
    except HTTPException:
        raise
    try:
        results = forecast_in_chunks(
            model_features,
            student_ids,
            course_codes,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    mentor_paths = (
        BASE_DIR / "mentor_data.csv",
        BASE_DIR / "mentor_profiles.csv",
    )
    mentor_path = next(
        (path for path in mentor_paths if path.is_file()),
        None,
    )
    if mentor_path is None:
        raise HTTPException(
            status_code=500,
            detail=(
                "Mentor data was not found. "
                "Run `uv run python mentor_generation.py` first."
            ),
        )
    try:
        mentor_df = pd.read_csv(mentor_path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError) as error:
        raise HTTPException(
            status_code=500,
            detail=f"Could not read mentor dataset '{mentor_path}': {error}",
        ) from error
    add_capacity_plans(results, mentor_df)
    return results


# # =====================================================
# # MENTOR RECOMMENDATIONS
# # =====================================================

# @app.post("/mentor-recommendations")
# def mentor_recommendations(
#         request: MentorRequest
# ):

#     student_df = read_student_dataset(request.student_dataset_path)
#     mentor_df = read_dataset(request.mentor_dataset_path)
#     student_ids, course_codes = dataset_identifiers(student_df)
#     try:
#         prepared_students = prepare_model_features(student_df)
#         forecast_results = forecast_in_chunks(
#             prepared_students,
#             student_ids=student_ids,
#             course_codes=course_codes,
#         )
#     except ValueError as error:
#         raise HTTPException(status_code=422, detail=str(error)) from error

#     manager, plans = add_capacity_plans(forecast_results, mentor_df)
#     capacity = manager.calculate_capacity()

#     demand = manager.demand_analysis(
#         current_demand=
#         forecast_results[
#             "current_at_risk_students"
#         ],

#         six_month_demand=
#         forecast_results[
#             "forecast_after_6_months"
#         ],

#         final_exam_demand=
#         forecast_results[
#             "forecast_before_final_exam"
#         ]
#     )

#     demand["additional_mentors_needed_after_6_months"] = plans[
#         "after_6_months"
#     ]["additional_mentors_required"]
#     demand["additional_mentors_needed_before_final_exam"] = plans[
#         "after_12_months_or_course_end"
#     ]["additional_mentors_required"]
#     demand["additional_mentors_needed_current"] = plans[
#         "current"
#     ]["additional_mentors_required"]
#     demand["period_capacity_plans"] = plans

#     return {

#         "forecast": forecast_results,

#         "capacity":
#         capacity,

#         "demand_analysis":
#         demand,

#         "recommendations":
#         plans["after_12_months_or_course_end"]["allocation"]
#     }
