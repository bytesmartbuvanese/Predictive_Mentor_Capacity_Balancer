#!/usr/bin/env python3
"""
Standalone Mentor Management CLI
--------------------------------
Reads:
  1. Student prediction CSV produced by the student-risk prediction pipeline.
  2. Mentor input CSV containing only static mentor information.
  3. (Optional) Existing mentor assignment CSV with mentor_id and
     current_assigned_students.

Writes separate allocation, mentor workload, and capacity summary CSV files
for current, six-month, and final-exam predictions.

Required mentor input columns:
mentor_id, mentor_name, department_id, department_name, specialization,
experience_years, availability_hours_per_week, max_student_capacity

Required student prediction columns:
student_id, course_code (optional), plus risk/probability columns for each
prediction horizon. Defaults:
current_is_at_risk, current_risk_probability
six_month_is_at_risk, six_month_risk_probability
final_exam_is_at_risk, final_exam_risk_probability

Use --help to see command-line options.
"""

from __future__ import annotations

import argparse
import heapq
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


MENTOR_REQUIRED_COLUMNS = [
    "mentor_id",
    "mentor_name",
    "department_id",
    "department_name",
    "specialization",
    "experience_years",
    "availability_hours_per_week",
    "max_student_capacity",
]

HORIZONS = {
    "current": ("current_is_at_risk", "current_risk_probability"),
    "six_month": ("six_month_is_at_risk", "six_month_risk_probability"),
    "final_exam": ("final_exam_is_at_risk", "final_exam_risk_probability"),
}


def parse_bool(value) -> bool:
    """Convert common CSV boolean representations into a Python bool."""
    if pd.isna(value):
        raise ValueError("Risk classification cannot be empty.")
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        if value in (0, 1):
            return bool(value)
        raise ValueError(
            f"Risk classification must be binary (0 or 1), got {value!r}."
        )
    normalized = str(value).strip().lower()
    if normalized == "1":
        return True
    if normalized == "0":
        return False
    raise ValueError(
        f"Risk classification must be binary (0 or 1), got {value!r}."
    )


def safe_filename_part(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value)


class MentorManagement:
    def __init__(
        self,
        mentor_df: pd.DataFrame,
        existing_assignments_df: Optional[pd.DataFrame] = None,
        *,
        use_embedded_assignments: bool = True,
    ):
        self.mentors = self._prepare_mentors(
            mentor_df,
            existing_assignments_df,
            use_embedded_assignments=use_embedded_assignments,
        )

    @staticmethod
    def _prepare_mentors(
        mentor_df: pd.DataFrame,
        existing_assignments_df: Optional[pd.DataFrame],
        *,
        use_embedded_assignments: bool,
    ) -> pd.DataFrame:
        mentors = mentor_df.copy()
        missing = [c for c in MENTOR_REQUIRED_COLUMNS if c not in mentors.columns]
        if missing:
            raise ValueError(
                "Mentor input CSV is missing required columns: " + ", ".join(missing)
            )
        if mentors.empty:
            raise ValueError("Mentor input CSV contains no mentor records.")
        if mentors["mentor_id"].isna().any() or mentors["mentor_id"].astype(str).duplicated().any():
            raise ValueError("mentor_id values must be present and unique.")

        numeric_columns = [
            "experience_years",
            "availability_hours_per_week",
            "max_student_capacity",
        ]
        for col in numeric_columns:
            mentors[col] = pd.to_numeric(mentors[col], errors="coerce")
            if mentors[col].isna().any() or not np.isfinite(mentors[col]).all():
                raise ValueError(f"Invalid or missing numeric values in mentor column '{col}'.")
            if (mentors[col] < 0).any():
                raise ValueError(f"Mentor column '{col}' cannot contain negative values.")

        if not np.equal(
            mentors["max_student_capacity"],
            np.floor(mentors["max_student_capacity"]),
        ).all():
            raise ValueError("max_student_capacity values must be whole numbers.")

        if use_embedded_assignments and "current_assigned_students" in mentors.columns:
            mentors["current_assigned_students"] = pd.to_numeric(
                mentors["current_assigned_students"],
                errors="coerce",
            )
            if (
                mentors["current_assigned_students"].isna().any()
                or not np.isfinite(mentors["current_assigned_students"]).all()
                or (mentors["current_assigned_students"] < 0).any()
            ):
                raise ValueError(
                    "current_assigned_students must contain valid non-negative numbers."
                )
        else:
            mentors["current_assigned_students"] = 0
        if existing_assignments_df is not None:
            if not {"mentor_id", "current_assigned_students"}.issubset(
                existing_assignments_df.columns
            ):
                raise ValueError(
                    "Existing assignments CSV must contain mentor_id and "
                    "current_assigned_students."
                )
            existing = existing_assignments_df[
                ["mentor_id", "current_assigned_students"]
            ].copy()
            if (
                existing["mentor_id"].isna().any()
                or existing["mentor_id"].astype(str).str.strip().eq("").any()
            ):
                raise ValueError(
                    "Existing assignments CSV contains an empty mentor_id."
                )
            if existing["mentor_id"].astype(str).duplicated().any():
                raise ValueError(
                    "Existing assignments CSV has duplicate mentor_id values."
                )
            unknown_mentor_ids = set(existing["mentor_id"].astype(str)) - set(
                mentors["mentor_id"].astype(str)
            )
            if unknown_mentor_ids:
                raise ValueError(
                    "Existing assignments CSV contains unknown mentor_id values: "
                    + ", ".join(sorted(unknown_mentor_ids)[:10])
                )
            existing["current_assigned_students"] = pd.to_numeric(
                existing["current_assigned_students"], errors="coerce"
            )
            if (
                existing["current_assigned_students"].isna().any()
                or not np.isfinite(existing["current_assigned_students"]).all()
                or (existing["current_assigned_students"] < 0).any()
            ):
                raise ValueError(
                    "current_assigned_students must contain valid non-negative numbers."
                )
            if not np.equal(
                existing["current_assigned_students"],
                np.floor(existing["current_assigned_students"]),
            ).all():
                raise ValueError(
                    "current_assigned_students values must be whole numbers."
                )
            mentors = mentors.drop(columns=["current_assigned_students"]).merge(
                existing, on="mentor_id", how="left", validate="one_to_one"
            )
            mentors["current_assigned_students"] = (
                mentors["current_assigned_students"].fillna(0)
            )

        if (
            mentors["current_assigned_students"] > mentors["max_student_capacity"]
        ).any():
            bad_ids = mentors.loc[
                mentors["current_assigned_students"] > mentors["max_student_capacity"],
                "mentor_id",
            ].astype(str).tolist()
            raise ValueError(
                "Existing assignments exceed maximum capacity for mentor(s): "
                + ", ".join(bad_ids)
            )

        if not np.equal(
            mentors["current_assigned_students"],
            np.floor(mentors["current_assigned_students"]),
        ).all():
            raise ValueError("current assignments must be whole numbers.")
        mentors["current_assigned_students"] = mentors[
            "current_assigned_students"
        ].astype(int)
        mentors["max_student_capacity"] = mentors["max_student_capacity"].astype(int)
        mentors["available_student_capacity"] = (
            mentors["max_student_capacity"] - mentors["current_assigned_students"]
        ).clip(lower=0).astype(int)
        mentors["current_workload_percentage"] = np.divide(
            mentors["current_assigned_students"] * 100,
            mentors["max_student_capacity"],
            out=np.zeros(len(mentors), dtype=float),
            where=mentors["max_student_capacity"].to_numpy() > 0,
        ).round(2)
        return mentors.reset_index(drop=True)

    def capacity_summary(self) -> dict:
        m = self.mentors
        total_max = int(m["max_student_capacity"].sum())
        total_assigned = int(m["current_assigned_students"].sum())
        total_available = int(m["available_student_capacity"].sum())
        return {
            "total_mentors": int(len(m)),
            "total_max_capacity": total_max,
            "current_assigned_students": total_assigned,
            "available_student_capacity": total_available,
            "currently_available_mentors": int((m["available_student_capacity"] > 0).sum()),
            "average_current_workload_percentage": round(
                float(m["current_workload_percentage"].mean()), 2
            ),
        }

    def allocate(
        self,
        students: pd.DataFrame,
        risk_column: str,
        probability_column: str,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
        """Allocate highest-risk students first using capacity-aware mentor scoring.

        A course_code-to-specialization match is preferred. If no matching
        mentor has capacity, the student is assigned to the best available
        mentor by projected workload, experience, and weekly availability.
        """
        required_student_cols = {"student_id", risk_column, probability_column}
        missing = required_student_cols - set(students.columns)
        if missing:
            raise ValueError(
                "Student prediction CSV is missing required column(s): "
                + ", ".join(sorted(missing))
            )

        if students.empty:
            raise ValueError("Student prediction CSV contains no student records.")
        if students["student_id"].isna().any() or (
            students["student_id"].astype(str).str.strip() == ""
        ).any():
            raise ValueError("student_id cannot contain empty values.")
        if students["student_id"].astype(str).duplicated().any():
            raise ValueError(
                "Student prediction CSV must contain at most one row per student."
            )

        classifications = students[risk_column].map(parse_bool).to_numpy(dtype=bool)
        probabilities = pd.to_numeric(
            students[probability_column],
            errors="coerce",
        )
        if probabilities.isna().any() or not np.isfinite(probabilities).all():
            raise ValueError(
                f"Risk probabilities in '{probability_column}' must be valid numbers for every student."
            )
        if ((probabilities < 0) | (probabilities > 1)).any():
            raise ValueError(f"Values in '{probability_column}' must be between 0 and 1.")

        student_ids = students["student_id"].astype(str).to_numpy()
        course_codes = (
            students["course_code"].to_numpy()
            if "course_code" in students.columns
            else np.full(len(students), None, dtype=object)
        )
        probabilities = probabilities.to_numpy(dtype=float)
        risky_indices = np.flatnonzero(classifications)
        risky_order = sorted(
            risky_indices.tolist(),
            key=lambda index: (-probabilities[index], student_ids[index]),
        )

        mentors = self.mentors.copy()
        n_mentors = len(mentors)
        assigned_new = [0] * n_mentors
        remaining = mentors["available_student_capacity"].astype(int).tolist()
        current_assigned = mentors["current_assigned_students"].astype(int).tolist()
        max_student_capacity = mentors["max_student_capacity"].astype(int).tolist()
        experience = mentors["experience_years"].astype(float).tolist()
        availability = mentors["availability_hours_per_week"].astype(float).tolist()
        mentor_ids = mentors["mentor_id"].tolist()
        mentor_names = mentors["mentor_name"].tolist()
        mentor_departments = mentors["department_name"].tolist()
        mentor_specializations = mentors["specialization"].tolist()
        generation = [0] * n_mentors
        global_heap: List[tuple] = []
        specialization_heaps: Dict[str, list] = {}

        def score(i: int) -> float:
            projected_load = (
                current_assigned[i] + assigned_new[i] + 1
            ) / max_student_capacity[i]
            # Lower score is preferred. Capacity utilization is the main factor.
            return (
                projected_load * 100
                - min(experience[i], 25) * 0.25
                - min(availability[i], 40) * 0.01
            )

        def push(i: int) -> None:
            if remaining[i] <= 0:
                return
            entry = (score(i), str(mentor_ids[i]), i, generation[i])
            heapq.heappush(global_heap, entry)
            spec = mentor_specializations[i]
            if pd.notna(spec) and str(spec).strip():
                heapq.heappush(
                    specialization_heaps.setdefault(
                        str(spec).strip().casefold(),
                        [],
                    ),
                    entry,
                )

        def pop_valid(heap: list):
            while heap:
                entry = heapq.heappop(heap)
                i = entry[2]
                if remaining[i] > 0 and entry[3] == generation[i]:
                    return i
            return None

        for i in range(n_mentors):
            push(i)

        allocation_rows = []
        for student_index in risky_order:
            student_id = student_ids[student_index]
            course = course_codes[student_index]
            probability = float(probabilities[student_index])
            course_key = (
                str(course).strip().casefold()
                if pd.notna(course) and str(course).strip()
                else None
            )
            mentor_index = (
                pop_valid(specialization_heaps.get(course_key, []))
                if course_key
                else None
            )
            match_type = "specialization"
            if mentor_index is None:
                mentor_index = pop_valid(global_heap)
                match_type = "capacity_fallback" if course_key else "capacity_only_course_unknown"

            if mentor_index is None:
                allocation_rows.append({
                    "student_id": student_id,
                    "course_code": course,
                    "risk_probability": probability,
                    "risk_classification": 1,
                    "allocation_status": "unallocated",
                    "mentor_id": None,
                    "mentor_name": None,
                    "assigned_mentor_id": None,
                    "assigned_mentor_name": None,
                    "match_type": "none",
                    "reason": "No mentor has remaining available capacity.",
                })
                continue

            remaining[mentor_index] -= 1
            assigned_new[mentor_index] += 1
            generation[mentor_index] += 1
            push(mentor_index)

            total_after = int(
                current_assigned[mentor_index] + assigned_new[mentor_index]
            )
            workload_after = round(
                total_after / max_student_capacity[mentor_index] * 100,
                2,
            )
            allocation_rows.append({
                "student_id": student_id,
                "course_code": course,
                "risk_probability": probability,
                "risk_classification": 1,
                "allocation_status": "allocated",
                "mentor_id": mentor_ids[mentor_index],
                "mentor_name": mentor_names[mentor_index],
                "assigned_mentor_id": mentor_ids[mentor_index],
                "assigned_mentor_name": mentor_names[mentor_index],
                "mentor_department": mentor_departments[mentor_index],
                "mentor_specialization": mentor_specializations[mentor_index],
                "match_type": match_type,
                "reason": (
                    "Matched course specialization and available capacity."
                    if match_type == "specialization"
                    else "Assigned using remaining capacity, projected workload, "
                         "experience, and availability."
                ),
                "mentor_total_assigned_after_allocation": total_after,
                "mentor_available_capacity_after_allocation": int(remaining[mentor_index]),
                "mentor_workload_after_allocation_percentage": workload_after,
            })

        allocations = pd.DataFrame(
            allocation_rows,
            columns=[
                "student_id",
                "course_code",
                "risk_probability",
                "risk_classification",
                "allocation_status",
                "mentor_id",
                "mentor_name",
                "assigned_mentor_id",
                "assigned_mentor_name",
                "mentor_department",
                "mentor_specialization",
                "match_type",
                "reason",
                "mentor_total_assigned_after_allocation",
                "mentor_available_capacity_after_allocation",
                "mentor_workload_after_allocation_percentage",
            ],
        )
        mentor_output = mentors.copy()
        mentor_output["newly_assigned_students"] = assigned_new
        mentor_output["total_assigned_students"] = (
            mentor_output["current_assigned_students"]
            + mentor_output["newly_assigned_students"]
        )
        mentor_output["available_student_capacity"] = (
            mentor_output["max_student_capacity"]
            - mentor_output["total_assigned_students"]
        ).clip(lower=0).astype(int)
        mentor_output["projected_workload_percentage"] = np.divide(
            mentor_output["total_assigned_students"] * 100,
            mentor_output["max_student_capacity"],
            out=np.zeros(len(mentor_output), dtype=float),
            where=mentor_output["max_student_capacity"].to_numpy() > 0,
        ).round(2)
        mentor_output["allocation_status"] = np.select(
            [
                mentor_output["newly_assigned_students"].eq(0),
                mentor_output["available_student_capacity"].eq(0),
            ],
            ["Not Allocated", "Fully Allocated"],
            default="Partially Available",
        )

        allocated_count = int(
            (allocations["allocation_status"] == "allocated").sum()
        ) if not allocations.empty else 0
        demand = len(risky_order)
        unallocated_count = int(demand - allocated_count)
        total_capacity = int(mentors["max_student_capacity"].sum())
        positive_capacities = mentors.loc[
            mentors["max_student_capacity"] > 0,
            "max_student_capacity",
        ]
        avg_capacity = (
            max(1, int(math.floor(positive_capacities.mean())))
            if not positive_capacities.empty
            else 1
        )
        additional_mentors = math.ceil(unallocated_count / avg_capacity) if unallocated_count else 0
        mentor_recommendations = []
        for mentor_index, newly_assigned in enumerate(assigned_new):
            if not newly_assigned:
                continue
            mentor_recommendations.append({
                "mentor_id": mentor_ids[mentor_index],
                "mentor_name": mentor_names[mentor_index],
                "specialization": mentor_specializations[mentor_index],
                "available_capacity": int(
                    self.mentors.iloc[mentor_index][
                        "available_student_capacity"
                    ]
                ),
                "recommended_students": newly_assigned,
                "future_workload_percentage": float(
                    mentor_output.iloc[mentor_index][
                        "projected_workload_percentage"
                    ]
                ),
            })
        summary = {
            **self.capacity_summary(),
            "total_students": int(len(students)),
            "predicted_at_risk_students": demand,
            "allocated_students": allocated_count,
            "unallocated_students": unallocated_count,
            "estimated_total_mentors_for_predicted_demand": (
                math.ceil(demand / avg_capacity) if demand else 0
            ),
            "additional_mentors_required": additional_mentors,
            "planning_capacity_per_mentor": avg_capacity,
            "successfully_allocated_students": allocated_count,
            "total_mentor_capacity": total_capacity,
            "currently_available_mentor_capacity": self.capacity_summary()[
                "available_student_capacity"
            ],
            "estimated_total_mentors_required": (
                math.ceil(demand / avg_capacity) if demand else 0
            ),
            "capacity_assumptions": (
                "Planning capacity is the floor of the mean positive mentor "
                "maximum capacity. Additional mentors are estimated from "
                "unallocated at-risk students divided by this planning capacity."
            ),
            "mentor_recommendations": mentor_recommendations,
            "projected_capacity_utilization_percentage": round(
                min(
                    (int(mentors["current_assigned_students"].sum()) + allocated_count)
                    / total_capacity * 100,
                    100,
                ),
                2,
            ) if total_capacity else 0.0,
        }
        return allocations, mentor_output, summary

    def calculate_capacity(self) -> dict:
        """Return the mentor capacity fields used by the existing API."""
        summary = self.capacity_summary()
        return {
            "total_mentors": summary["total_mentors"],
            "total_max_capacity": summary["total_max_capacity"],
            "currently_assigned": summary["current_assigned_students"],
            "available_capacity": summary["available_student_capacity"],
            "available_mentors": summary["currently_available_mentors"],
            "average_workload_percentage": summary[
                "average_current_workload_percentage"
            ],
        }

    def build_capacity_plan(
        self,
        student_forecasts,
        *,
        period,
        risk_key,
        probability_key,
    ) -> dict:
        """Build an API-compatible per-horizon capacity and allocation plan."""
        students = pd.DataFrame(student_forecasts)
        if risk_key not in students.columns or probability_key not in students.columns:
            raise ValueError(
                f"Student forecasts must contain '{risk_key}' and "
                f"'{probability_key}'."
            )
        risk_column = f"_api_{period}_risk"
        probability_column = f"_api_{period}_probability"
        risk_classifications = students[risk_key].map(parse_bool)
        students[risk_column] = risk_classifications.astype(int)
        students[probability_column] = students[probability_key]
        students = (
            students.sort_values(
                by=probability_column,
                ascending=False,
                kind="mergesort",
            )
            .drop_duplicates(subset=["student_id"], keep="first")
            .reset_index(drop=True)
        )
        mapping, _, summary = self.allocate(
            students,
            risk_column,
            probability_column,
        )
        assignments = []
        for record in mapping.to_dict(orient="records"):
            assignments.append({
                "student_id": record["student_id"],
                "course_code": record["course_code"],
                "risk_probability": record["risk_probability"],
                "status": record["allocation_status"],
                "mentor_id": record["assigned_mentor_id"],
                "mentor_name": record["assigned_mentor_name"],
                "match_type": record["match_type"],
                "reason": record["reason"],
                "future_workload_percentage": record.get(
                    "mentor_workload_after_allocation_percentage"
                ),
            })
        at_risk_students = students.loc[
            students[risk_column].eq(1)
        ]
        positive_capacity = self.mentors.loc[
            self.mentors["max_student_capacity"] > 0,
            "max_student_capacity",
        ]
        planning_capacity = (
            max(1, int(math.floor(positive_capacity.mean())))
            if not positive_capacity.empty
            else 1
        )
        demand = len(at_risk_students)
        capacity = self.calculate_capacity()
        utilization = (
            (
                capacity["currently_assigned"]
                + summary["allocated_students"]
            )
            / capacity["total_max_capacity"]
            * 100
            if capacity["total_max_capacity"]
            else 0
        )
        return {
            "period": period,
            "predicted_at_risk_students": demand,
            "students_identified_at_risk": [
                {
                    "student_id": row["student_id"],
                    "course_code": row.get("course_code"),
                    "risk_probability": row[probability_key],
                }
                for row in at_risk_students.to_dict(orient="records")
            ],
            "required_mentors": math.ceil(demand / planning_capacity),
            "planning_capacity_per_mentor": planning_capacity,
            "total_mentors": capacity["total_mentors"],
            "currently_available_mentors": capacity["available_mentors"],
            "available_student_capacity": capacity["available_capacity"],
            "capacity_utilization_percentage": round(
                min(utilization, 100),
                2,
            ),
            "additional_mentors_required": summary[
                "additional_mentors_required"
            ],
            "allocation": {
                "allocation_basis": "risk-prioritized capacity-aware mentor scoring",
                "students_requiring_mentoring": demand,
                "allocated_students": summary["allocated_students"],
                "unallocated_students": summary["unallocated_students"],
                "assignments": assignments,
                "mentor_recommendations": summary.get(
                    "mentor_recommendations",
                    [],
                ),
            },
        }

    def demand_analysis(
        self,
        current_demand,
        six_month_demand,
        final_exam_demand,
    ) -> dict:
        capacity = self.calculate_capacity()
        average_capacity = (
            capacity["total_max_capacity"] / capacity["total_mentors"]
            if capacity["total_mentors"]
            else 1
        )

        def gap_and_mentors(demand):
            gap = max(0, int(demand) - capacity["available_capacity"])
            additional = math.ceil(gap / average_capacity) if gap else 0
            return gap, additional

        six_gap, six_additional = gap_and_mentors(six_month_demand)
        final_gap, final_additional = gap_and_mentors(final_exam_demand)
        current_gap, current_additional = gap_and_mentors(current_demand)
        return {
            "current_demand": int(current_demand),
            "six_month_demand": int(six_month_demand),
            "final_exam_demand": int(final_exam_demand),
            "available_capacity": capacity["available_capacity"],
            "current_capacity_gap": current_gap,
            "six_month_capacity_gap": six_gap,
            "final_exam_capacity_gap": final_gap,
            "additional_mentors_needed_current": current_additional,
            "additional_mentors_needed_after_6_months": six_additional,
            "additional_mentors_needed_before_final_exam": final_additional,
        }

    def recommend_student_allocations(
        self,
        student_forecasts,
        *,
        risk_key="course_end_is_at_risk",
        probability_key="course_end_risk_probability",
    ) -> dict:
        students = pd.DataFrame(student_forecasts)
        mapping, _, summary = self.allocate(
            students,
            risk_key,
            probability_key,
        )
        assignments = []
        for record in mapping.to_dict(orient="records"):
            assignments.append({
                **record,
                "status": record["allocation_status"],
            })
        return {
            "allocation_basis": "risk-prioritized capacity-aware mentor scoring",
            "students_requiring_mentoring": summary[
                "predicted_at_risk_students"
            ],
            "allocated_students": summary["allocated_students"],
            "unallocated_students": summary["unallocated_students"],
            "assignments": assignments,
        }


def run_cli(args) -> None:
    student_path = Path(args.students)
    mentor_path = Path(args.mentors)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not student_path.is_file():
        raise FileNotFoundError(f"Student predictions file not found: {student_path}")
    if not mentor_path.is_file():
        raise FileNotFoundError(f"Mentor data file not found: {mentor_path}")

    students = pd.read_csv(student_path)
    mentors = pd.read_csv(mentor_path)
    existing = pd.read_csv(args.existing_assignments) if args.existing_assignments else None

    manager = MentorManagement(
        mentors,
        existing,
        use_embedded_assignments=False,
    )
    print("\n========== PREDICTIVE MENTOR MANAGEMENT ==========")
    print(f"Students file : {student_path}")
    print(f"Mentors file  : {mentor_path}")
    print(f"Student rows  : {len(students):,}")
    print(f"Mentor rows   : {len(mentors):,}")

    for horizon, (risk_col, probability_col) in HORIZONS.items():
        if risk_col not in students.columns or probability_col not in students.columns:
            raise ValueError(
                f"Student prediction CSV must contain '{risk_col}' and "
                f"'{probability_col}' for the {horizon} horizon."
            )

        allocation_df, mentor_output_df, summary = manager.allocate(
            students, risk_col, probability_col
        )
        prefix = safe_filename_part(horizon)
        allocation_file = output_dir / f"{prefix}_student_mentor_mapping.csv"
        mentor_file = output_dir / f"{prefix}_mentor_workload.csv"
        summary_file = output_dir / f"{prefix}_capacity_summary.csv"

        allocation_df.to_csv(allocation_file, index=False)
        mentor_output_df.to_csv(mentor_file, index=False)
        pd.DataFrame([summary]).to_csv(summary_file, index=False)

        print(f"\n---------- {horizon.replace('_', ' ').upper()} ----------")
        print(f"Students predicted at risk : {summary['predicted_at_risk_students']:,}")
        print(f"Students allocated         : {summary['allocated_students']:,}")
        print(f"Students unallocated       : {summary['unallocated_students']:,}")
        print(f"Existing available capacity: {summary['available_student_capacity']:,}")
        print(f"Additional mentors required: {summary['additional_mentors_required']:,}")
        print(f"Projected capacity usage   : {summary['projected_capacity_utilization_percentage']}%")
        print(f"Student mapping            : {allocation_file}")
        print(f"Mentor workload            : {mentor_file}")
        print(f"Capacity summary           : {summary_file}")

    print("\nProcessing complete.")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Allocate predicted at-risk students to mentors from the terminal."
    )
    parser.add_argument(
        "--students",
        default="student_predictions.csv",
        help="Student prediction CSV (default: student_predictions.csv)",
    )
    parser.add_argument(
        "--mentors",
        default="mentor_data.csv",
        help="Mentor input CSV (default: mentor_data.csv)",
    )
    parser.add_argument(
        "--existing-assignments",
        default=None,
        help=(
            "Optional CSV containing mentor_id,current_assigned_students. "
            "If omitted, existing assignments are initialized to zero."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="mentor_management_outputs",
        help="Directory for generated CSV outputs.",
    )
    return parser


if __name__ == "__main__":
    try:
        run_cli(build_parser().parse_args())
    except Exception as exc:
        raise SystemExit(f"\nERROR: {exc}")
