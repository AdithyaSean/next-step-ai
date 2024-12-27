"""generator package."""

import os
import random

import pandas as pd

from ..config.config import config


def generate_synthetic_data():
    """Generate synthetic student data with realistic patterns."""
    data = []
    for _ in range(config["num_students"]):
        student = {}

        # O/L Results (Grades: A, B, C, S, W)
        for subject in config["ol_subjects"]:
            grades = ["A", "B", "C", "S", "W"]
            student[f"OL_{subject}"] = random.choice(grades)

        # A/L Stream and Results
        stream = random.choice(list(config["al_streams"].keys()))
        student["AL_Stream"] = stream
        for subject in config["al_streams"][stream]:
            grades = ["A", "B", "C", "S", "F"]
            student[f"AL_{subject}"] = random.choice(grades)

        # University Course (Based on A/L Stream)
        courses = config["university_courses"][stream]
        chosen_course = random.choice(courses)
        student["University_Course"] = chosen_course
        student["Recommended_Career"] = random.choice(
            config["career_paths"][chosen_course]
        )

        data.append(student)

    df = pd.DataFrame(data)

    # Convert grades to numerical values (optional, but recommended)
    grade_mapping_ol = {"A": 9, "B": 7, "C": 5, "S": 3, "W": 1}
    grade_mapping_al = {"A": 10, "B": 8, "C": 6, "S": 4, "F": 0}

    for subject in config["ol_subjects"]:
        df[f"OL_{subject}_Numerical"] = df[f"OL_{subject}"].map(grade_mapping_ol)

    for stream, subjects in config["al_streams"].items():
        for subject in subjects:
            df[f"AL_{subject}_Numerical"] = df[f"AL_{subject}"].map(grade_mapping_al)

    # Create the directory if it does not exist
    os.makedirs(config["data_dir"], exist_ok=True)
    file_path = os.path.join(config["data_dir"], "synthetic_student_data.csv")
    df.to_csv(file_path, index=False)
    print(f"Synthetic data generated and saved to {file_path}")
    return df


if __name__ == "__main__":
    generate_synthetic_data()
