
import pandas as pd
import numpy as np
import random
import argparse
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

NUM_MENTORS = 100
COURSE_FILE = "./cleaned_data/courses_cleaned.csv"
OUTPUT_FILE = "mentor_data.csv"

RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================
# LOAD COURSE DATA
# ============================================================

def load_courses(course_file):

    course_path = Path(course_file)

    if not course_path.exists():
        raise FileNotFoundError(
            f"Course file not found: {course_path.resolve()}"
        )

    course_df = pd.read_csv(course_path)

    required_columns = [
        "code_module",
        "code_presentation",
        "module_presentation_length"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in course_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns in course.csv: {missing_columns}"
        )

    # Get unique course codes
    course_codes = sorted(
        course_df["code_module"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    if not course_codes:
        raise ValueError("No course codes found in course.csv.")

    print("\nCourses loaded from course.csv:")
    print(course_codes)

    return course_codes


# ============================================================
# MENTOR DATA GENERATION
# ============================================================

def generate_mentor_data(course_codes, num_mentors):

    mentors = []

    qualifications = [
        "B.Tech",
        "M.Tech",
        "M.Sc",
        "MBA",
        "Ph.D",
        "MCA"
    ]

    for i in range(1, num_mentors + 1):

        # Assign a course specialization from course.csv
        specialization = random.choice(course_codes)

        # Department information is not available in course.csv.
        # Therefore, use the course code as a synthetic department
        # identifier and label.
        department_id = f"DEPT_{specialization}"
        department_name = f"Department {specialization}"

        # Generate mentor experience
        experience_years = random.randint(1, 30)

        # Generate qualification based on experience
        if experience_years <= 3:
            qualification = random.choice([
                "B.Tech",
                "B.Sc",
                "MCA"
            ])

        elif experience_years <= 8:
            qualification = random.choice([
                "M.Tech",
                "M.Sc",
                "MBA",
                "MCA"
            ])

        else:
            qualification = random.choice([
                "M.Tech",
                "M.Sc",
                "MBA",
                "Ph.D"
            ])

        # Weekly mentoring availability
        availability_hours_per_week = random.choice([
            5, 8, 10, 12, 15, 18, 20, 25, 30
        ])

        # Maximum number of students a mentor can handle
        if availability_hours_per_week <= 10:

            max_student_capacity = random.randint(5, 12)

        elif availability_hours_per_week <= 20:

            max_student_capacity = random.randint(10, 20)

        else:

            max_student_capacity = random.randint(15, 30)

        mentor = {
            "mentor_id": f"M{i:04d}",

            "mentor_name": f"Mentor_{i:04d}",

            "department_id": department_id,

            "department_name": department_name,

            "specialization": specialization,

            "experience_years": experience_years,

            # "qualification": qualification,

            "availability_hours_per_week": availability_hours_per_week,

            "max_student_capacity": max_student_capacity
        }

        mentors.append(mentor)

    return pd.DataFrame(mentors)


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Generate synthetic mentor data using course.csv"
    )

    parser.add_argument(
        "--courses",
        default=COURSE_FILE,
        help="Path to course.csv"
    )

    parser.add_argument(
        "--num-mentors",
        type=int,
        default=NUM_MENTORS,
        help="Number of mentors to generate"
    )

    parser.add_argument(
        "--output",
        default=OUTPUT_FILE,
        help="Output mentor CSV filename"
    )

    args = parser.parse_args()

    if args.num_mentors <= 0:
        raise ValueError("Number of mentors must be greater than zero.")

    print("\n==========================================")
    print("         MENTOR DATA GENERATION")
    print("==========================================")

    # Load actual course codes
    course_codes = load_courses(args.courses)

    # Generate mentor dataset
    mentor_df = generate_mentor_data(
        course_codes,
        args.num_mentors
    )

    # Save CSV
    output_path = Path(args.output)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    mentor_df.to_csv(
        output_path,
        index=False
    )

    # ========================================================
    # GENERATION SUMMARY
    # ========================================================

    print("\nMentor dataset generated successfully!")

    print(f"\nTotal mentors generated: {len(mentor_df)}")

    print(f"Total course specializations: {len(course_codes)}")

    print(f"Output file: {output_path.resolve()}")

    print("\nGenerated columns:")

    for column in mentor_df.columns:
        print(f" - {column}")

    print("\nMentor distribution by specialization:")

    print(
        mentor_df["specialization"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nDataset preview:")

    print(
        mentor_df.head(10).to_string(index=False)
    )

    print("\n==========================================")
    print("       GENERATION COMPLETED")
    print("==========================================")


if __name__ == "__main__":
    main()
