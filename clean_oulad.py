
"""
OULAD COMPLETE DATA CLEANING PIPELINE
Project: Predictive Mentor Capacity Balancer

Input:
    data/oulad/

Output:
    cleaned_data/
        studentInfo_cleaned.csv
        studentRegistration_cleaned.csv
        studentAssessment_cleaned.csv
        assessments_cleaned.csv
        studentVle_cleaned.csv
        vle_cleaned.csv
        courses_cleaned.csv
        studentVle_enriched_cleaned.csv
        cleaning_report.csv
        missing_values_report.csv

Important:
    - Preserves legitimate missing values.
    - Does not create synthetic student records.
    - Does not modify academic outcomes.
    - Aggregates repeated student-resource-date interactions.
    - Validates foreign keys before joining datasets.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_DIR = BASE_DIR / "raw_dataset"
OUTPUT_DIR = BASE_DIR / "cleaned_data"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "studentInfo": "studentInfo.csv",
    "studentRegistration": "studentRegistration.csv",
    "studentAssessment": "studentAssessment.csv",
    "assessments": "assessments.csv",
    "studentVle": "studentVle.csv",
    "vle": "vle.csv",
    "courses": "courses.csv",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("OULAD_CLEANING")


# ============================================================
# 2. CLEANING REPORT STORAGE
# ============================================================

cleaning_records = []
missing_records = []


def record_cleaning(
    dataset,
    original_rows,
    cleaned_rows,
    duplicates_removed,
    invalid_rows_removed
):
    cleaning_records.append({
        "dataset": dataset,
        "original_rows": original_rows,
        "cleaned_rows": cleaned_rows,
        "duplicates_removed": duplicates_removed,
        "invalid_rows_removed": invalid_rows_removed,
        "rows_removed_total": original_rows - cleaned_rows
    })


def record_missing_values(dataset, df):

    for column in df.columns:

        missing_count = int(df[column].isna().sum())

        missing_records.append({
            "dataset": dataset,
            "feature": column,
            "missing_count": missing_count,
            "missing_percentage": round(
                (missing_count / len(df)) * 100, 2
            ) if len(df) else 0
        })


# ============================================================
# 3. LOAD DATASETS
# ============================================================

def load_dataset(dataset_name):

    filename = FILES[dataset_name]
    filepath = INPUT_DIR / filename

    if not filepath.exists():
        raise FileNotFoundError(
            f"Required OULAD file not found: {filepath}"
        )

    logger.info(f"Loading {filename}")

    df = pd.read_csv(
        filepath,
        low_memory=False
    )

    # Remove accidental whitespace from column names.
    df.columns = df.columns.str.strip()

    # Remove whitespace around string values.
    for column in df.select_dtypes(include=["object"]).columns:
        df[column] = df[column].map(
            lambda x: x.strip() if isinstance(x, str) else x
        )

    logger.info(
        f"{filename}: {len(df):,} rows, {len(df.columns)} columns"
    )

    return df


# ============================================================
# 4. GENERAL CLEANING FUNCTIONS
# ============================================================

def remove_duplicates(df, subset=None):

    before = len(df)

    if subset:
        df = df.drop_duplicates(
            subset=subset,
            keep="first"
        )
    else:
        df = df.drop_duplicates(keep="first")

    removed = before - len(df)

    return df, removed


def remove_invalid_identifiers(df, columns):

    before = len(df)

    for column in columns:
        df = df.dropna(subset=[column])

    removed = before - len(df)

    return df, removed


def convert_numeric(df, columns):

    for column in columns:

        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    return df


def save_dataset(df, filename):

    output_path = OUTPUT_DIR / filename

    df.to_csv(
        output_path,
        index=False
    )

    logger.info(
        f"Saved {filename} | {len(df):,} rows"
    )


# ============================================================
# 5. CLEAN COURSES
# ============================================================

def clean_courses(df):

    logger.info("Cleaning courses.csv")

    original_rows = len(df)

    required = [
        "code_module",
        "code_presentation"
    ]

    df, invalid = remove_invalid_identifiers(
        df,
        required
    )

    df, duplicates = remove_duplicates(
        df,
        subset=required
    )

    df = convert_numeric(
        df,
        ["module_presentation_length"]
    )

    if "module_presentation_length" in df.columns:

        df.loc[
            df["module_presentation_length"] <= 0,
            "module_presentation_length"
        ] = np.nan

    record_cleaning(
        "courses",
        original_rows,
        len(df),
        duplicates,
        invalid
    )

    record_missing_values("courses", df)

    return df


# ============================================================
# 6. CLEAN STUDENT INFORMATION
# ============================================================

def clean_student_info(df):

    logger.info("Cleaning studentInfo.csv")

    original_rows = len(df)

    required = [
        "id_student",
        "code_module",
        "code_presentation"
    ]

    df, invalid = remove_invalid_identifiers(
        df,
        required
    )

    df, duplicates = remove_duplicates(
        df,
        subset=required
    )

    numeric_columns = [
        "id_student",
        "num_of_prev_attempts",
        "studied_credits"
    ]

    df = convert_numeric(df, numeric_columns)

    # Invalid numeric identifiers are not usable.
    before = len(df)

    df = df[
        df["id_student"].notna()
        & (df["id_student"] >= 0)
    ]

    invalid += before - len(df)

    for column in [
        "num_of_prev_attempts",
        "studied_credits"
    ]:

        if column in df.columns:
            df.loc[
                df[column] < 0,
                column
            ] = np.nan

    if "gender" in df.columns:
        df["gender"] = df["gender"].replace(
            {"?": np.nan, "": np.nan}
        )

    if "imd_band" in df.columns:
        df["imd_band"] = df["imd_band"].replace(
            {"?": np.nan, "": np.nan}
        )

    if "age_band" in df.columns:
        df["age_band"] = df["age_band"].replace(
            {"?": np.nan, "": np.nan}
        )

    if "final_result" in df.columns:

        valid_results = [
            "Distinction",
            "Pass",
            "Fail",
            "Withdrawn"
        ]

        df.loc[
            ~df["final_result"].isin(valid_results),
            "final_result"
        ] = np.nan

    record_cleaning(
        "studentInfo",
        original_rows,
        len(df),
        duplicates,
        invalid
    )

    record_missing_values("studentInfo", df)

    return df


# ============================================================
# 7. CLEAN STUDENT REGISTRATION
# ============================================================

def clean_student_registration(df):

    logger.info("Cleaning studentRegistration.csv")

    original_rows = len(df)

    required = [
        "id_student",
        "code_module",
        "code_presentation"
    ]

    df, invalid = remove_invalid_identifiers(
        df,
        required
    )

    df, duplicates = remove_duplicates(
        df,
        subset=required
    )

    df = convert_numeric(
        df,
        [
            "id_student",
            "date_registration",
            "date_unregistration"
        ]
    )

    # Negative relative-day values are legitimate
    # in OULAD and must not be removed.

    record_cleaning(
        "studentRegistration",
        original_rows,
        len(df),
        duplicates,
        invalid
    )

    record_missing_values("studentRegistration", df)

    return df


# ============================================================
# 8. CLEAN ASSESSMENTS
# ============================================================

def clean_assessments(df):

    logger.info("Cleaning assessments.csv")

    original_rows = len(df)

    required = [
        "id_assessment",
        "code_module",
        "code_presentation"
    ]

    df, invalid = remove_invalid_identifiers(
        df,
        required
    )

    df, duplicates = remove_duplicates(
        df,
        subset=["id_assessment"]
    )

    df = convert_numeric(
        df,
        [
            "id_assessment",
            "date",
            "weight"
        ]
    )

    if "weight" in df.columns:

        df.loc[
            (df["weight"] < 0)
            | (df["weight"] > 100),
            "weight"
        ] = np.nan

    record_cleaning(
        "assessments",
        original_rows,
        len(df),
        duplicates,
        invalid
    )

    record_missing_values("assessments", df)

    return df


# ============================================================
# 9. CLEAN STUDENT ASSESSMENTS
# ============================================================

def clean_student_assessment(df):

    logger.info("Cleaning studentAssessment.csv")

    original_rows = len(df)

    required = [
        "id_assessment",
        "id_student"
    ]

    df, invalid = remove_invalid_identifiers(
        df,
        required
    )

    df, duplicates = remove_duplicates(
        df,
        subset=required
    )

    df = convert_numeric(
        df,
        [
            "id_assessment",
            "id_student",
            "date_submitted",
            "score",
            "is_banked"
        ]
    )

    if "is_banked" in df.columns:
        df["is_banked"] = df["is_banked"].fillna(0).astype(int)

    # Keep missing scores: they may represent
    # non-submission or unavailable assessment results.

    if "score" in df.columns:

        df.loc[
            (df["score"] < 0)
            | (df["score"] > 100),
            "score"
        ] = np.nan

    record_cleaning(
        "studentAssessment",
        original_rows,
        len(df),
        duplicates,
        invalid
    )

    record_missing_values("studentAssessment", df)

    return df


# ============================================================
# 10. CLEAN VLE RESOURCE DATA
# ============================================================

def clean_vle(df):

    logger.info("Cleaning vle.csv")

    original_rows = len(df)

    df, invalid = remove_invalid_identifiers(
        df,
        ["id_site"]
    )

    df, duplicates = remove_duplicates(
        df,
        subset=["id_site"]
    )

    df = convert_numeric(
        df,
        [
            "id_site",
            "week_from",
            "week_to"
        ]
    )

    # Preserve missing week information.

    record_cleaning(
        "vle",
        original_rows,
        len(df),
        duplicates,
        invalid
    )

    record_missing_values("vle", df)

    return df


# ============================================================
# 11. CLEAN STUDENT VLE INTERACTION DATA
# ============================================================

def clean_student_vle(df):

    logger.info("Cleaning studentVle.csv")

    original_rows = len(df)

    required = [
        "id_student",
        "code_module",
        "code_presentation",
        "id_site",
        "date",
        "sum_click"
    ]

    df, invalid = remove_invalid_identifiers(
        df,
        required
    )

    df = convert_numeric(
        df,
        [
            "id_student",
            "id_site",
            "date",
            "sum_click"
        ]
    )

    # Remove records with invalid numeric identifiers
    # or invalid interaction counts.

    before = len(df)

    df = df[
        df["id_student"].notna()
        & (df["id_student"] >= 0)
        & df["id_site"].notna()
        & df["date"].notna()
        & df["sum_click"].notna()
        & (df["sum_click"] >= 0)
    ]

    invalid += before - len(df)

    # OULAD may contain repeated records for the
    # same student, course, resource and date.
    #
    # These are aggregated by summing sum_click.
    # This preserves the total number of interactions.

    aggregation_keys = [
        "id_student",
        "code_module",
        "code_presentation",
        "id_site",
        "date"
    ]

    before_aggregation = len(df)

    df = (
        df.groupby(
            aggregation_keys,
            as_index=False,
            dropna=False
        )
        .agg({
            "sum_click": "sum"
        })
    )

    duplicates = before_aggregation - len(df)

    record_cleaning(
        "studentVle",
        original_rows,
        len(df),
        duplicates,
        invalid
    )

    record_missing_values("studentVle", df)

    return df


# ============================================================
# 12. DATASET RELATIONSHIP VALIDATION
# ============================================================

def validate_relationships(
    student_info,
    courses,
    assessments,
    student_assessment,
    vle,
    student_vle
):

    logger.info("Validating dataset relationships")

    # Validate student-course relationships.

    student_course_keys = student_info[
        [
            "id_student",
            "code_module",
            "code_presentation"
        ]
    ].drop_duplicates()

    registration_keys = set(
        zip(
            student_course_keys["id_student"],
            student_course_keys["code_module"],
            student_course_keys["code_presentation"]
        )
    )

    vle_keys = list(
        zip(
            student_vle["id_student"],
            student_vle["code_module"],
            student_vle["code_presentation"]
        )
    )

    invalid_student_vle = sum(
        key not in registration_keys
        for key in vle_keys
    )

    logger.info(
        "Student VLE records without matching "
        f"studentInfo records: {invalid_student_vle:,}"
    )

    # Validate course references.

    course_keys = set(
        zip(
            courses["code_module"],
            courses["code_presentation"]
        )
    )

    assessment_course_keys = list(
        zip(
            assessments["code_module"],
            assessments["code_presentation"]
        )
    )

    invalid_assessment_courses = sum(
        key not in course_keys
        for key in assessment_course_keys
    )

    logger.info(
        "Assessments without matching courses: "
        f"{invalid_assessment_courses:,}"
    )

    # Validate assessment references.

    valid_assessment_ids = set(
        assessments["id_assessment"].dropna()
    )

    invalid_assessment_refs = (
        ~student_assessment["id_assessment"].isin(
            valid_assessment_ids
        )
    ).sum()

    logger.info(
        "Student assessment records without matching "
        f"assessment definitions: {invalid_assessment_refs:,}"
    )

    # Validate VLE resource references.

    valid_site_ids = set(
        vle["id_site"].dropna()
    )

    invalid_site_refs = (
        ~student_vle["id_site"].isin(valid_site_ids)
    ).sum()

    logger.info(
        "Student VLE records without matching "
        f"resource definitions: {invalid_site_refs:,}"
    )


# ============================================================
# 13. ENRICH STUDENT VLE WITH RESOURCE INFORMATION
# ============================================================

def create_enriched_student_vle(student_vle, vle):

    logger.info("Creating enriched student VLE dataset")

    enriched = student_vle.merge(
        vle,
        on="id_site",
        how="left",
        validate="many_to_one",
        indicator=True
    )

    unmatched = (
        enriched["_merge"] != "both"
    ).sum()

    logger.info(
        f"Unmatched VLE resource records: {unmatched:,}"
    )

    enriched.drop(
        columns=["_merge"],
        inplace=True
    )

    return enriched


# ============================================================
# 14. GENERATE CLEANING REPORTS
# ============================================================

def generate_reports():

    cleaning_df = pd.DataFrame(cleaning_records)

    missing_df = pd.DataFrame(missing_records)

    cleaning_path = OUTPUT_DIR / "cleaning_report.csv"
    missing_path = OUTPUT_DIR / "missing_values_report.csv"

    cleaning_df.to_csv(
        cleaning_path,
        index=False
    )

    missing_df.to_csv(
        missing_path,
        index=False
    )

    logger.info("Cleaning report saved")

    print("\n" + "=" * 70)
    print("OULAD CLEANING SUMMARY")
    print("=" * 70)

    print(
        cleaning_df.to_string(index=False)
    )

    print("\n" + "=" * 70)
    print("MISSING VALUES SUMMARY")
    print("=" * 70)

    print(
        missing_df.sort_values(
            "missing_percentage",
            ascending=False
        ).head(30).to_string(index=False)
    )


# ============================================================
# 15. MAIN PIPELINE
# ============================================================

def main():

    start_time = datetime.now()

    logger.info("=" * 70)
    logger.info("STARTING OULAD CLEANING PIPELINE")
    logger.info("=" * 70)

    # --------------------------------------------------------
    # Load original datasets
    # --------------------------------------------------------

    courses = load_dataset("courses")
    student_info = load_dataset("studentInfo")
    registration = load_dataset("studentRegistration")
    assessments = load_dataset("assessments")
    student_assessment = load_dataset("studentAssessment")
    vle = load_dataset("vle")
    student_vle = load_dataset("studentVle")

    # --------------------------------------------------------
    # Clean datasets
    # --------------------------------------------------------

    courses = clean_courses(courses)

    student_info = clean_student_info(student_info)

    registration = clean_student_registration(
        registration
    )

    assessments = clean_assessments(assessments)

    student_assessment = clean_student_assessment(
        student_assessment
    )

    vle = clean_vle(vle)

    student_vle = clean_student_vle(student_vle)

    # --------------------------------------------------------
    # Validate dataset relationships
    # --------------------------------------------------------

    validate_relationships(
        student_info,
        courses,
        assessments,
        student_assessment,
        vle,
        student_vle
    )

    # --------------------------------------------------------
    # Create enriched VLE dataset
    # --------------------------------------------------------

    enriched_student_vle = create_enriched_student_vle(
        student_vle,
        vle
    )

    # --------------------------------------------------------
    # Save cleaned datasets
    # --------------------------------------------------------

    save_dataset(
        courses,
        "courses_cleaned.csv"
    )

    save_dataset(
        student_info,
        "studentInfo_cleaned.csv"
    )

    save_dataset(
        registration,
        "studentRegistration_cleaned.csv"
    )

    save_dataset(
        assessments,
        "assessments_cleaned.csv"
    )

    save_dataset(
        student_assessment,
        "studentAssessment_cleaned.csv"
    )

    save_dataset(
        vle,
        "vle_cleaned.csv"
    )

    save_dataset(
        student_vle,
        "studentVle_cleaned.csv"
    )

    save_dataset(
        enriched_student_vle,
        "studentVle_enriched_cleaned.csv"
    )

    # --------------------------------------------------------
    # Generate cleaning reports
    # --------------------------------------------------------

    generate_reports()

    end_time = datetime.now()

    duration = (end_time - start_time).total_seconds()

    logger.info("=" * 70)
    logger.info("OULAD CLEANING COMPLETED SUCCESSFULLY")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info(f"Execution time: {duration:.2f} seconds")
    logger.info("=" * 70)


# ============================================================
# 16. EXECUTION ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:
        main()

    except Exception as error:

        logger.exception(
            f"OULAD cleaning pipeline failed: {error}"
        )

        raise
