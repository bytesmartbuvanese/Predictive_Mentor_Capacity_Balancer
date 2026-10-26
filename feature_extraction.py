from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_DIR = BASE_DIR / "cleaned_data"
OUTPUT_DIR = BASE_DIR / "engineered_data"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "students": INPUT_DIR / "studentInfo_cleaned.csv",
    "registration": INPUT_DIR / "studentRegistration_cleaned.csv",
    "student_assessment": INPUT_DIR / "studentAssessment_cleaned.csv",
    "assessments": INPUT_DIR / "assessments_cleaned.csv",
    "student_vle": INPUT_DIR / "studentVle_cleaned.csv",
    "vle": INPUT_DIR / "vle_cleaned.csv",
    "courses": INPUT_DIR / "courses_cleaned.csv"
}

# Prediction timeline:
# Prediction Cutoff = module_presentation_length * 0.50 (course duration x 0.50)
# Forecast Horizon = 60 days
# Forecast End = min(cutoff + horizon, course duration)
FORECAST_HORIZON_DAYS = 60

# Binary classification target:
# 1 if final_result in ('Fail', 'Withdrawn') else 0 (Pass, Distinction)
INCLUDE_WITHDRAWN_AS_AT_RISK = True

RANDOM_SEED = 42


# ============================================================
# 2. LOAD CLEANED DATASETS
# ============================================================

def load_datasets():

    datasets = {}

    for name, path in FILES.items():

        if not path.exists():
            raise FileNotFoundError(
                f"Missing input dataset: {path}"
            )

        datasets[name] = pd.read_csv(path)

        print(
            f"Loaded {name}: "
            f"{len(datasets[name]):,} records"
        )

    return datasets


# ============================================================
# 3. GENERAL DATA PREPARATION
# ============================================================

def prepare_data(datasets):

    students = datasets["students"].copy()
    courses = datasets["courses"].copy()

    for col in ["id_student"]:
        students[col] = pd.to_numeric(
            students[col], errors="coerce"
        )

    students = students.dropna(
        subset=[
            "id_student",
            "code_module",
            "code_presentation"
        ]
    )

    students["id_student"] = (
        students["id_student"].astype(int)
    )

    # One row per student-course-presentation.
    students = students.drop_duplicates(
        subset=[
            "id_student",
            "code_module",
            "code_presentation"
        ]
    )

    # Convert identifiers to consistent string types.
    for col in ["code_module", "code_presentation"]:
        students[col] = students[col].astype(str)
        courses[col] = courses[col].astype(str)

    # Attach course duration and calculate timeline parameters:
    # Cutoff = module_presentation_length * 0.50
    # Forecast End = MIN(Cutoff + 60, module_presentation_length)
    courses["module_presentation_length"] = pd.to_numeric(
        courses["module_presentation_length"], errors="coerce"
    )

    course_timeline = courses[
        ["code_module", "code_presentation", "module_presentation_length"]
    ].drop_duplicates(subset=["code_module", "code_presentation"]).copy()

    course_timeline["cutoff"] = course_timeline["module_presentation_length"] * 0.50
    course_timeline["forecast_end"] = np.minimum(
        course_timeline["cutoff"] + FORECAST_HORIZON_DAYS,
        course_timeline["module_presentation_length"]
    )

    students = students.merge(
        course_timeline,
        on=["code_module", "code_presentation"],
        how="left"
    )

    return students


# ============================================================
# 4. REGISTRATION FEATURES
# ============================================================

def create_registration_features(
    students,
    registration
):

    registration = registration.copy()

    registration["date_registration"] = pd.to_numeric(
        registration["date_registration"],
        errors="coerce"
    )

    # registration_lead_days = MAX(0, -date_registration)
    registration["registration_lead_days"] = (
        -registration["date_registration"]
    ).clip(lower=0)

    registration_features = registration[
        [
            "id_student",
            "code_module",
            "code_presentation",
            "registration_lead_days"
        ]
    ]

    result = students.merge(
        registration_features,
        on=[
            "id_student",
            "code_module",
            "code_presentation"
        ],
        how="left"
    )

    result["registration_lead_days"] = (
        result["registration_lead_days"].fillna(0)
    )

    return result


# ============================================================
# 5. ASSESSMENT FEATURES
# ============================================================

def create_assessment_features(
    students,
    student_assessment,
    assessments
):
    """
    Derives audited assessment features:
    - expected_assessments: Count distinct id_assessment for the course (including undated).
    - submitted_assessments: Count distinct valid submissions with date_submitted <= cutoff.
    - upcoming_assessment_count: Assessments with cutoff < date <= forecast_end and not already submitted by cutoff.
    - missed_assessment_count: Assessments with date <= cutoff and no submission by cutoff.
    - assessment_completion_rate: (Submitted assessments due by cutoff / Total assessments due by cutoff) * 100.
    - average_assessment_score: Mean of valid submitted scores by cutoff (excluding banked).
    - latest_assessment_score: Score of most recent valid submission by cutoff.
    - assessment_score_trend: Linear regression slope across submitted assessments (date_submitted as x_i).
    - assessment_score_std: Population standard deviation (ddof=0) of valid scores submitted by cutoff.
    - assessment_weighted_score: Weighted average of eligible submitted scores.
    """
    sa = student_assessment.copy()
    a = assessments.copy()

    numeric_columns = ["date_submitted", "score", "is_banked"]
    for col in numeric_columns:
        if col in sa.columns:
            sa[col] = pd.to_numeric(sa[col], errors="coerce")

    a["date"] = pd.to_numeric(a["date"], errors="coerce")
    a["weight"] = pd.to_numeric(a["weight"], errors="coerce")

    # Exclude banked assessment results from current-presentation performance
    if "is_banked" in sa.columns:
        sa = sa[sa["is_banked"] == 0].copy()

    # Total expected assessments for each course presentation (including missing dates)
    expected_all = (
        a.groupby(["code_module", "code_presentation"])
        .agg(expected_assessments=("id_assessment", "nunique"))
        .reset_index()
    )

    # Attach course timeline info (cutoff & forecast_end) to assessments
    course_meta = students[
        ["code_module", "code_presentation", "cutoff", "forecast_end"]
    ].drop_duplicates()

    a_with_meta = a.merge(course_meta, on=["code_module", "code_presentation"], how="left")

    # Assessments due by cutoff (must have valid date <= cutoff)
    due_assessments = a_with_meta[
        a_with_meta["date"].notna()
        & (a_with_meta["date"] <= a_with_meta["cutoff"])
    ][["code_module", "code_presentation", "id_assessment", "date", "weight"]].copy()

    due_by_cutoff_counts = (
        due_assessments.groupby(["code_module", "code_presentation"])
        .agg(assessments_due_by_cutoff=("id_assessment", "nunique"))
        .reset_index()
    )

    # Assessments scheduled in the upcoming forecast window (cutoff < date <= forecast_end)
    upcoming_assessments = a_with_meta[
        a_with_meta["date"].notna()
        & (a_with_meta["date"] > a_with_meta["cutoff"])
        & (a_with_meta["date"] <= a_with_meta["forecast_end"])
    ][["code_module", "code_presentation", "id_assessment", "date"]].copy()

    # Merge student assessment submissions with assessment info and course timeline
    submissions = sa.merge(
        a[["id_assessment", "code_module", "code_presentation", "date", "weight"]],
        on="id_assessment",
        how="inner"
    )

    submissions = submissions.merge(
        students[["id_student", "code_module", "code_presentation", "cutoff", "forecast_end"]],
        on=["id_student", "code_module", "code_presentation"],
        how="inner"
    )

    # Valid submissions submitted on or before cutoff with valid score [0, 100]
    valid_submissions = submissions[
        submissions["date_submitted"].notna()
        & (submissions["date_submitted"] <= submissions["cutoff"])
        & submissions["score"].notna()
        & submissions["score"].between(0, 100)
    ].copy()

    # Calculate student-level submission performance metrics
    def calculate_student_assessment_metrics(group):
        group = group.sort_values("date_submitted")
        scores = group["score"].values
        dates = group["date_submitted"].values
        weights = group["weight"].values
        n = len(scores)

        if n == 0:
            return pd.Series({
                "average_assessment_score": np.nan,
                "latest_assessment_score": np.nan,
                "assessment_score_trend": np.nan,
                "assessment_score_std": np.nan,
                "assessment_weighted_score": np.nan,
                "submitted_assessments": 0
            })

        avg_score = float(np.mean(scores))
        latest_score = float(scores[-1])

        # Population standard deviation (ddof=0)
        score_std = float(np.std(scores, ddof=0))

        # Weighted score: denominator includes only weights of assessments with valid submitted scores
        valid_weight_mask = ~np.isnan(weights) & (weights > 0)
        sum_weights = np.sum(weights[valid_weight_mask])
        if sum_weights > 0:
            weighted_score = float(np.sum(scores[valid_weight_mask] * weights[valid_weight_mask]) / sum_weights)
        else:
            weighted_score = avg_score

        # Linear regression slope: date_submitted as x_i, score as y_i
        # Formula: m = (n * sum(x*y) - sum(x)*sum(y)) / (n * sum(x^2) - (sum(x))^2)
        # If fewer than 2 valid scores, set trend to np.nan
        if n >= 2:
            sum_x = np.sum(dates)
            sum_y = np.sum(scores)
            sum_xy = np.sum(dates * scores)
            sum_x2 = np.sum(dates ** 2)
            denom = n * sum_x2 - (sum_x ** 2)
            if denom != 0:
                trend = float((n * sum_xy - sum_x * sum_y) / denom)
            else:
                trend = np.nan
        else:
            trend = np.nan

        submitted_cnt = int(group["id_assessment"].nunique())

        return pd.Series({
            "average_assessment_score": avg_score,
            "latest_assessment_score": latest_score,
            "assessment_score_trend": trend,
            "assessment_score_std": score_std,
            "assessment_weighted_score": weighted_score,
            "submitted_assessments": submitted_cnt
        })

    # Group valid submissions by student and course
    assessment_stats = (
        valid_submissions.groupby(
            ["id_student", "code_module", "code_presentation"],
            as_index=False
        )
        .apply(calculate_student_assessment_metrics, include_groups=False)
        .reset_index()
    )

    # Calculate student-specific missed assessments & completion rate:
    # Identify distinct assessments due by cutoff that each student actually submitted by cutoff
    submitted_ids = set(
        zip(valid_submissions["id_student"], valid_submissions["id_assessment"])
    )

    # Cross join enrolled students with their presentation's due assessments to check each deadline
    student_course_pairs = students[
        ["id_student", "code_module", "code_presentation"]
    ].drop_duplicates()

    due_per_student = student_course_pairs.merge(
        due_assessments[["code_module", "code_presentation", "id_assessment"]],
        on=["code_module", "code_presentation"],
        how="inner"
    )

    due_per_student["was_submitted"] = [
        (s, a_id) in submitted_ids
        for s, a_id in zip(due_per_student["id_student"], due_per_student["id_assessment"])
    ]

    due_summary = (
        due_per_student.groupby(["id_student", "code_module", "code_presentation"])
        .agg(
            submitted_due_by_cutoff=("was_submitted", "sum"),
            assessments_due_by_cutoff=("id_assessment", "nunique")
        )
        .reset_index()
    )

    due_summary["missed_assessment_count"] = (
        due_summary["assessments_due_by_cutoff"] - due_summary["submitted_due_by_cutoff"]
    ).clip(lower=0)

    # Upcoming assessments: assessments due between cutoff and forecast_end not already submitted by cutoff
    upcoming_per_student = student_course_pairs.merge(
        upcoming_assessments[["code_module", "code_presentation", "id_assessment"]],
        on=["code_module", "code_presentation"],
        how="inner"
    )

    upcoming_per_student["was_submitted"] = [
        (s, a_id) in submitted_ids
        for s, a_id in zip(upcoming_per_student["id_student"], upcoming_per_student["id_assessment"])
    ]

    upcoming_summary = (
        upcoming_per_student[~upcoming_per_student["was_submitted"]]
        .groupby(["id_student", "code_module", "code_presentation"])
        .agg(upcoming_assessment_count=("id_assessment", "nunique"))
        .reset_index()
    )

    # Merge features together
    result = students.merge(expected_all, on=["code_module", "code_presentation"], how="left")
    result = result.merge(due_by_cutoff_counts, on=["code_module", "code_presentation"], how="left")
    result = result.merge(
        assessment_stats,
        on=["id_student", "code_module", "code_presentation"],
        how="left"
    )
    result = result.merge(
        due_summary[[
            "id_student", "code_module", "code_presentation",
            "submitted_due_by_cutoff", "missed_assessment_count"
        ]],
        on=["id_student", "code_module", "code_presentation"],
        how="left"
    )
    result = result.merge(
        upcoming_summary,
        on=["id_student", "code_module", "code_presentation"],
        how="left"
    )

    result["expected_assessments"] = result["expected_assessments"].fillna(0)
    result["assessments_due_by_cutoff"] = result["assessments_due_by_cutoff"].fillna(0)
    result["submitted_assessments"] = result["submitted_assessments"].fillna(0)
    result["submitted_due_by_cutoff"] = result["submitted_due_by_cutoff"].fillna(0)
    result["missed_assessment_count"] = result["missed_assessment_count"].fillna(
        result["assessments_due_by_cutoff"]
    )
    result["upcoming_assessment_count"] = result["upcoming_assessment_count"].fillna(0)

    # assessment_completion_rate: (Submitted assessments due by cutoff / Total assessments due by cutoff) * 100
    # If zero assessments due by cutoff, set to np.nan
    result["assessment_completion_rate"] = np.where(
        result["assessments_due_by_cutoff"] > 0,
        (result["submitted_due_by_cutoff"] / result["assessments_due_by_cutoff"]) * 100.0,
        np.nan
    )

    # Drop intermediate helper columns
    result = result.drop(columns=["submitted_due_by_cutoff"], errors="ignore")

    return result


# ============================================================
# 6. VLE ENGAGEMENT FEATURES
# ============================================================

def create_engagement_features(
    students,
    student_vle,
    vle
):
    """
    Derives audited VLE engagement features:
    - total_clicks: SUM(sum_click) on or before cutoff.
    - active_days: Count distinct dates with sum_click > 0 and date <= cutoff.
    - last_activity_day: MAX(date) on or before cutoff.
    - early_clicks: SUM(sum_click) during early window [0, W - 1].
    - recent_clicks: SUM(sum_click) during recent window [cutoff - W + 1, cutoff].
    - average_daily_clicks: total_clicks / active_days (NULL if active_days == 0).
    - engagement_change: recent_daily_click_rate - early_daily_click_rate over equal windows W.
    - inactive_days: cutoff - last_activity_day (NULL if no activity).
    - resource_diversity: Count distinct id_site on or before cutoff.
    - activity_type_diversity: Count distinct activity_type accessed on or before cutoff.
    """
    sv = student_vle.copy()
    vle = vle.copy()

    sv["date"] = pd.to_numeric(sv["date"], errors="coerce")
    sv["sum_click"] = pd.to_numeric(sv["sum_click"], errors="coerce")

    # Attach cutoff to student VLE records
    student_cutoffs = students[
        ["id_student", "code_module", "code_presentation", "cutoff"]
    ].drop_duplicates()

    sv = sv.merge(
        student_cutoffs,
        on=["id_student", "code_module", "code_presentation"],
        how="inner"
    )

    # Filter interactions observed on or before student's presentation cutoff
    sv = sv[
        sv["date"].notna()
        & (sv["date"] <= sv["cutoff"])
        & sv["sum_click"].notna()
        & (sv["sum_click"] >= 0)
    ].copy()

    # Window definition: W = floor((cutoff + 1) / 4)
    # Early window: [0, W - 1]
    # Recent window: [cutoff - W + 1, cutoff]
    sv["W"] = np.floor((sv["cutoff"] + 1) / 4.0).astype(int)
    sv["early_clicks"] = np.where(
        (sv["date"] >= 0) & (sv["date"] <= sv["W"] - 1),
        sv["sum_click"],
        0.0
    )
    sv["recent_clicks"] = np.where(
        (sv["date"] >= sv["cutoff"] - sv["W"] + 1) & (sv["date"] <= sv["cutoff"]),
        sv["sum_click"],
        0.0
    )

    # Attach resource activity type from vle
    sv = sv.merge(
        vle[
            ["id_site", "code_module", "code_presentation", "activity_type"]
        ].drop_duplicates(subset=["id_site", "code_module", "code_presentation"]),
        on=["id_site", "code_module", "code_presentation"],
        how="left"
    )

    # Filter strictly positive clicks for active days count
    sv_active = sv[sv["sum_click"] > 0]
    active_days_df = (
        sv_active.groupby(["id_student", "code_module", "code_presentation"])
        .agg(active_days=("date", "nunique"))
        .reset_index()
    )

    engagement = (
        sv.groupby(["id_student", "code_module", "code_presentation"])
        .agg(
            total_clicks=("sum_click", "sum"),
            last_activity_day=("date", "max"),
            early_clicks=("early_clicks", "sum"),
            recent_clicks=("recent_clicks", "sum"),
            resource_diversity=("id_site", "nunique"),
            activity_type_diversity=("activity_type", "nunique"),
            W=("W", "first")
        )
        .reset_index()
    )

    engagement = engagement.merge(
        active_days_df,
        on=["id_student", "code_module", "code_presentation"],
        how="left"
    )
    engagement["active_days"] = engagement["active_days"].fillna(0).astype(int)

    # average_daily_clicks: total_clicks / active_days. If active_days == 0, NULL.
    engagement["average_daily_clicks"] = np.where(
        engagement["active_days"] > 0,
        engagement["total_clicks"] / engagement["active_days"],
        np.nan
    )

    # engagement_change: recent_daily_click_rate - early_daily_click_rate
    # where each rate is clicks divided by window length W
    engagement["engagement_change"] = np.where(
        engagement["W"] > 0,
        (engagement["recent_clicks"] / engagement["W"]) - (engagement["early_clicks"] / engagement["W"]),
        np.nan
    )

    result = students.merge(
        engagement[[
            "id_student", "code_module", "code_presentation",
            "total_clicks", "active_days", "last_activity_day",
            "early_clicks", "recent_clicks", "resource_diversity",
            "activity_type_diversity", "average_daily_clicks", "engagement_change"
        ]],
        on=["id_student", "code_module", "code_presentation"],
        how="left"
    )

    # Students with no VLE activity: keep total_clicks, active_days, click sums, diversity at 0
    zero_fill_cols = [
        "total_clicks", "active_days", "early_clicks",
        "recent_clicks", "resource_diversity", "activity_type_diversity"
    ]
    for col in zero_fill_cols:
        result[col] = result[col].fillna(0)

    # If student has total_clicks == 0, engagement_change is 0.0
    result["engagement_change"] = np.where(
        result["total_clicks"] == 0,
        0.0,
        result["engagement_change"]
    )

    # inactive_days: cutoff - last_activity_day. If no activity, keep NULL (handled during clean/impute)
    result["inactive_days"] = np.where(
        result["last_activity_day"].notna(),
        (result["cutoff"] - result["last_activity_day"]).clip(lower=0),
        np.nan
    )

    return result


# ============================================================
# 7. COURSE AND ENROLLMENT FEATURES
# ============================================================

def create_course_features(
    students,
    courses,
    registration
):
    """
    Derives audited course and enrollment features:
    - course_enrollment_count: Students registered and still enrolled as of cutoff.
    - enrollment_growth_rate: Percentage change in cutoff enrollment compared to previous presentation.
    - historical_course_failure_rate: (Previous Fail students / Total previous enrolled students) * 100.
    - course_duration_days: module_presentation_length.
    """
    result = students.copy()
    reg = registration.copy()

    reg["date_registration"] = pd.to_numeric(reg["date_registration"], errors="coerce")
    reg["date_unregistration"] = pd.to_numeric(reg["date_unregistration"], errors="coerce")

    # Attach cutoff to registration
    course_cutoffs = students[
        ["code_module", "code_presentation", "cutoff"]
    ].drop_duplicates()

    reg_with_cutoff = reg.merge(course_cutoffs, on=["code_module", "code_presentation"], how="inner")

    # Registered on or before cutoff, and not unregistered by cutoff (unregistration is null or > cutoff)
    active_at_cutoff = reg_with_cutoff[
        (reg_with_cutoff["date_registration"].isna() | (reg_with_cutoff["date_registration"] <= reg_with_cutoff["cutoff"]))
        & (reg_with_cutoff["date_unregistration"].isna() | (reg_with_cutoff["date_unregistration"] > reg_with_cutoff["cutoff"]))
    ]

    enrollment = (
        active_at_cutoff.groupby(["code_module", "code_presentation"])
        .agg(course_enrollment_count=("id_student", "nunique"))
        .reset_index()
    )

    result = result.merge(
        enrollment,
        on=["code_module", "code_presentation"],
        how="left"
    )
    result["course_enrollment_count"] = result["course_enrollment_count"].fillna(0).astype(int)

    # Chronological ordering of presentations per module
    presentations = enrollment.copy()
    presentations["year"] = (
        presentations["code_presentation"].str.extract(r"(\d{4})")[0].astype(float)
    )
    presentations["term_order"] = (
        presentations["code_presentation"].str.extract(r"(\D+)$")[0].map({"B": 0, "J": 1}).fillna(2)
    )
    presentations = presentations.sort_values(["code_module", "year", "term_order"])

    # Previous presentation equivalent cutoff enrollment
    presentations["previous_enrollment"] = (
        presentations.groupby("code_module")["course_enrollment_count"].shift(1)
    )

    # Formula: ((Current enrollment - Previous enrollment) / Previous enrollment) * 100
    # If no previous presentation (NaN or 0), set to np.nan (or 0)
    presentations["enrollment_growth_rate"] = np.where(
        presentations["previous_enrollment"] > 0,
        (
            (presentations["course_enrollment_count"] - presentations["previous_enrollment"])
            / presentations["previous_enrollment"]
        ) * 100.0,
        np.nan
    )

    result = result.merge(
        presentations[["code_module", "code_presentation", "enrollment_growth_rate"]],
        on=["code_module", "code_presentation"],
        how="left"
    )

    # Historical course failure rate:
    # Use ONLY earlier presentations of the same module. Current presentation outcomes are strictly excluded.
    # Formula: (Previous Fail students / Total previous enrolled students) * 100
    if "final_result" in students.columns:
        outcomes = students[
            ["id_student", "code_module", "code_presentation", "final_result"]
        ].drop_duplicates(subset=["id_student", "code_module", "code_presentation"]).copy()

        outcomes["is_fail"] = (outcomes["final_result"] == "Fail").astype(int)

        history = (
            outcomes.groupby(["code_module", "code_presentation"])
            .agg(
                presentation_students=("id_student", "nunique"),
                presentation_failures=("is_fail", "sum")
            )
            .reset_index()
        )

        history["year"] = history["code_presentation"].str.extract(r"(\d{4})")[0].astype(float)
        history["term_order"] = history["code_presentation"].str.extract(r"(\D+)$")[0].map({"B": 0, "J": 1}).fillna(2)
        history = history.sort_values(["code_module", "year", "term_order"])

        # Cumulative previous students and previous failures
        history["previous_students"] = history.groupby("code_module")["presentation_students"].transform(
            lambda x: x.cumsum().shift(1)
        )
        history["previous_failures"] = history.groupby("code_module")["presentation_failures"].transform(
            lambda x: x.cumsum().shift(1)
        )

        history["historical_course_failure_rate"] = np.where(
            history["previous_students"] > 0,
            (history["previous_failures"] / history["previous_students"]) * 100.0,
            np.nan
        )

        result = result.merge(
            history[["code_module", "code_presentation", "historical_course_failure_rate"]],
            on=["code_module", "code_presentation"],
            how="left"
        )
    else:
        result["historical_course_failure_rate"] = np.nan

    # Course duration days
    result["course_duration_days"] = result["module_presentation_length"]

    return result


# ============================================================
# 8. ASSESSMENT SCHEDULE DENSITY
# ============================================================

def add_schedule_density(features):
    """
    Derives assessment_schedule_density:
    Formula: Assessments due by cutoff / Cutoff day
    (density of scheduled assessments per elapsed course day)
    """
    result = features.copy()

    result["assessment_schedule_density"] = np.where(
        result["cutoff"] > 0,
        result["assessments_due_by_cutoff"] / result["cutoff"],
        0.0
    )

    return result


# ============================================================
# 9. CREATE TARGET VARIABLE
# ============================================================

def create_target(features):
    """
    Target variable definition:
    - final_result: Retained as ground truth academic outcome reference.
    - is_at_risk: Binary classification target.
      1 if final_result in ('Fail', 'Withdrawn')
      0 if final_result in ('Pass', 'Distinction')
    """
    result = features.copy()

    result["final_result"] = result["final_result"].astype(str).str.strip()

    valid_results = ["Fail", "Pass", "Distinction", "Withdrawn"]
    result = result[result["final_result"].isin(valid_results)].copy()

    result["is_at_risk"] = result["final_result"].isin(["Fail", "Withdrawn"]).astype(int)

    return result


# ============================================================
# 10. FINAL DATASET COMPOSITION (40 COLUMNS)
# ============================================================

DATASET_COLUMNS = [
    # A. Identifiers & Demographics (12)
    "code_module",
    "code_presentation",
    "id_student",
    "gender",
    "region",
    "highest_education",
    "imd_band",
    "age_band",
    "num_of_prev_attempts",
    "studied_credits",
    "disability",
    "registration_lead_days",

    # B. Academic Assessment Features (10)
    "expected_assessments",
    "average_assessment_score",
    "latest_assessment_score",
    "assessment_score_trend",
    "assessment_score_std",
    "assessment_weighted_score",
    "submitted_assessments",
    "upcoming_assessment_count",
    "missed_assessment_count",
    "assessment_completion_rate",

    # C. Student Engagement & VLE Activity Features (10)
    "total_clicks",
    "active_days",
    "last_activity_day",
    "early_clicks",
    "recent_clicks",
    "resource_diversity",
    "activity_type_diversity",
    "average_daily_clicks",
    "engagement_change",
    "inactive_days",

    # D. Enrollment & Course-Level Features (4)
    "course_enrollment_count",
    "enrollment_growth_rate",
    "historical_course_failure_rate",
    "course_duration_days",

    # E. Assessment Schedule & Workload Features (2)
    "assessments_due_by_cutoff",
    "assessment_schedule_density",

    # F. Prediction Target & Outcome Reference (2)
    "final_result",
    "is_at_risk"
]


def finalize_feature_dataset(features):
    """
    Selects the exact 40 audited columns in consistent order.
    Preserves legitimate NaNs in student_course_features.csv for auditing.
    """
    df = features.copy()

    # Drop internal helper columns if present
    drop_cols = ["cutoff", "forecast_end", "module_presentation_length", "W"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # Reorder columns
    df = df[DATASET_COLUMNS].copy()

    return df


# ============================================================
# 11. CREATE MODEL-READY CSV
# ============================================================

def create_model_training_dataset(features):
    """
    Prepares model-ready numeric dataset:
    - 38 potential model input features (encoded / imputed).
    - Excludes final_result, id_student, code_presentation.
    - One-hot encodes code_module and categorical demographics.
    - Target: is_at_risk.
    """
    df = features.copy()

    # Target
    target = df["is_at_risk"].astype(int).values

    # Exclude identifiers and outcome reference from predictors
    drop_from_predictors = [
        "id_student",
        "code_presentation",
        "final_result",
        "is_at_risk"
    ]

    predictors = df.drop(columns=drop_from_predictors, errors="ignore")

    # Categorical columns to encode
    cat_cols = [
        "code_module",
        "gender",
        "region",
        "highest_education",
        "imd_band",
        "age_band",
        "disability"
    ]

    for col in cat_cols:
        if col in predictors.columns:
            predictors[col] = predictors[col].astype(str).fillna("Unknown")

    predictors = pd.get_dummies(predictors, columns=cat_cols, dtype=int)

    # Impute numeric NaNs for model training baseline
    predictors = predictors.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Ensure all columns are numeric
    for col in predictors.columns:
        predictors[col] = pd.to_numeric(predictors[col], errors="coerce").fillna(0)

    predictors["is_at_risk"] = target

    return predictors


# ============================================================
# 12. FEATURE DOCUMENTATION
# ============================================================

def create_feature_documentation():

    documentation = [
        ["code_module", "studentInfo.csv", "Original", "Course module identifier."],
        ["code_presentation", "studentInfo.csv", "Original", "Academic presentation identifier."],
        ["id_student", "studentInfo.csv", "Original", "Unique student identifier."],
        ["gender", "studentInfo.csv", "Original", "Student demographic gender."],
        ["region", "studentInfo.csv", "Original", "Student geographical region."],
        ["highest_education", "studentInfo.csv", "Original", "Highest educational level on entry."],
        ["imd_band", "studentInfo.csv", "Original", "Index of Multiple Deprivation socioeconomic band."],
        ["age_band", "studentInfo.csv", "Original", "Student age band."],
        ["num_of_prev_attempts", "studentInfo.csv", "Original", "Number of previous attempts for this course."],
        ["studied_credits", "studentInfo.csv", "Original", "Total credits the student is currently studying."],
        ["disability", "studentInfo.csv", "Original", "Declared disability status."],
        ["registration_lead_days", "studentRegistration.csv", "Derived", "MAX(0, -date_registration). Days registered prior to course start."],

        ["expected_assessments", "assessments.csv", "Derived", "Count distinct id_assessment in module and presentation (including undated)."],
        ["average_assessment_score", "studentAssessment.csv", "Derived", "Mean of valid scores submitted by cutoff (excluding banked results)."],
        ["latest_assessment_score", "studentAssessment.csv", "Derived", "Score from the most recent valid submission by cutoff."],
        ["assessment_score_trend", "studentAssessment.csv", "Derived", "Linear regression slope across submitted assessments using date_submitted as x_i."],
        ["assessment_score_std", "studentAssessment.csv", "Derived", "Population standard deviation (ddof=0) of valid scores submitted by cutoff."],
        ["assessment_weighted_score", "studentAssessment.csv + assessments.csv", "Derived", "Weighted average of submitted scores: Sum(score * weight) / Sum(weight)."],
        ["submitted_assessments", "studentAssessment.csv", "Derived", "Count distinct assessments submitted with date_submitted <= cutoff."],
        ["upcoming_assessment_count", "assessments.csv + studentAssessment.csv", "Derived", "Count distinct assessments with cutoff < date <= forecast_end not submitted by cutoff."],
        ["missed_assessment_count", "assessments.csv + studentAssessment.csv", "Derived", "Count distinct assessments with date <= cutoff that have no submission by cutoff."],
        ["assessment_completion_rate", "assessments.csv + studentAssessment.csv", "Derived", "Percentage of assessments due by cutoff that were submitted by cutoff."],

        ["total_clicks", "studentVle.csv", "Derived", "SUM(sum_click) for interactions with date <= cutoff."],
        ["active_days", "studentVle.csv", "Derived", "Count distinct dates with sum_click > 0 and date <= cutoff."],
        ["last_activity_day", "studentVle.csv", "Derived", "MAX(date) for student VLE activity on or before cutoff."],
        ["early_clicks", "studentVle.csv", "Derived", "SUM(sum_click) in early window [0, W - 1], where W = floor((cutoff + 1) / 4)."],
        ["recent_clicks", "studentVle.csv", "Derived", "SUM(sum_click) in recent window [cutoff - W + 1, cutoff]."],
        ["resource_diversity", "studentVle.csv", "Derived", "Count distinct id_site accessed on or before cutoff."],
        ["activity_type_diversity", "studentVle.csv + vle.csv", "Derived", "Count distinct activity_type values accessed on or before cutoff."],
        ["average_daily_clicks", "studentVle.csv", "Derived", "total_clicks / active_days (NULL if active_days == 0)."],
        ["engagement_change", "studentVle.csv", "Derived", "recent_daily_click_rate - early_daily_click_rate over equal windows W."],
        ["inactive_days", "studentVle.csv", "Derived", "cutoff - last_activity_day (NULL if no activity)."],

        ["course_enrollment_count", "studentRegistration.csv", "Derived", "Count distinct students registered and still enrolled as of cutoff."],
        ["enrollment_growth_rate", "studentRegistration.csv", "Derived", "Percentage change in cutoff enrollment vs equivalent point in previous presentation."],
        ["historical_course_failure_rate", "studentInfo.csv", "Derived", "Fail percentage from earlier presentations only of the same module."],
        ["course_duration_days", "courses.csv", "Original", "module_presentation_length."],

        ["assessments_due_by_cutoff", "assessments.csv", "Derived", "Count distinct assessments scheduled with date <= cutoff."],
        ["assessment_schedule_density", "assessments.csv + courses.csv", "Derived", "Assessments due by cutoff divided by cutoff day."],

        ["final_result", "studentInfo.csv", "Original", "Overall academic outcome (Pass, Distinction, Fail, Withdrawn)."],
        ["is_at_risk", "Derived from studentInfo.csv", "Target", "1 if final_result in ('Fail', 'Withdrawn'), otherwise 0 (Pass, Distinction)."]
    ]

    return pd.DataFrame(
        documentation,
        columns=[
            "feature",
            "source_dataset",
            "feature_type",
            "description_and_formula"
        ]
    )


# ============================================================
# 13. SAVE OUTPUTS
# ============================================================

def save_outputs(features, model_data):

    feature_path = OUTPUT_DIR / "student_course_features.csv"
    model_path = OUTPUT_DIR / "student_model_training.csv"
    documentation_path = OUTPUT_DIR / "feature_documentation.csv"

    features.to_csv(feature_path, index=False)
    model_data.to_csv(model_path, index=False)
    create_feature_documentation().to_csv(documentation_path, index=False)

    print("\nOutput files created:")
    print(f"Feature dataset: {feature_path}")
    print(f"Model dataset: {model_path}")
    print(f"Feature documentation: {documentation_path}")

    print("\nFeature dataset shape:")
    print(features.shape)

    print("\nModel training dataset shape:")
    print(model_data.shape)

    print("\nTarget distribution:")
    print(
        features["is_at_risk"]
        .value_counts()
        .rename(index={0: "Not at risk (0)", 1: "At risk (1)"})
    )

    print("\nTarget percentages:")
    print(features["is_at_risk"].value_counts(normalize=True) * 100)


# ============================================================
# 14. MAIN PIPELINE
# ============================================================

def main():

    print("=" * 65)
    print("STUDENT FEATURE ENGINEERING PIPELINE (AUDITED)")
    print("=" * 65)

    datasets = load_datasets()

    students = prepare_data(datasets)

    print("\nCreating registration features...")
    features = create_registration_features(
        students,
        datasets["registration"]
    )

    print("Creating assessment features...")
    features = create_assessment_features(
        features,
        datasets["student_assessment"],
        datasets["assessments"]
    )

    print("Creating VLE engagement features...")
    features = create_engagement_features(
        features,
        datasets["student_vle"],
        datasets["vle"]
    )

    print("Creating course and enrollment features...")
    features = create_course_features(
        features,
        datasets["courses"],
        datasets["registration"]
    )

    print("Creating assessment schedule density...")
    features = add_schedule_density(features)

    print("Creating target variable...")
    features = create_target(features)

    print("Finalizing 40-column audited dataset...")
    features = finalize_feature_dataset(features)

    print("Creating model-ready dataset...")
    model_data = create_model_training_dataset(features)

    save_outputs(features, model_data)

    print("\nFeature engineering completed successfully.")


if __name__ == "__main__":
    main()
