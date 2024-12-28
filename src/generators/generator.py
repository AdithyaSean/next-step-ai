"""Enhanced data generator."""

import os

import numpy as np
import pandas as pd

from ..config.config import AL_STREAMS, CAREERS, EDUCATION_LEVELS, OL_SUBJECTS, config


def generator():
    """Generate comprehensive student profiles with multiple education levels."""
    data = []

    total_students = config["num_students"]
    education_dist = config["education_level_dist"]

    def generate_ol_results():
        return {
            f"OL_subject_{idx}_score": np.random.randint(0, 100)
            for idx in config["ol_subjects"].values()
        }

    def generate_al_results(stream_id):
        return {
            f"AL_subject_{sub_id}_score": np.random.randint(0, 100)
            for sub_id in config["al_subjects"][stream_id]
        }

    def generate_career_probs(profile):
        """Generate career probabilities based on education and performance."""
        probs = {}

        for career_id, (base_low, base_high) in config["career_success_ranges"].items():
            prob = base_low
            edu_level = profile.get("education_level")
            stream = profile.get("AL_stream")
            gpa = profile.get("gpa", 0)

            # Normalize GPA to 0-100 scale for probability calculations
            gpa_normalized = (
                (gpa - config["gpa_range"]["min"])
                / (config["gpa_range"]["max"] - config["gpa_range"]["min"])
            ) * 100

            # Add OL subject contributions
            ol_math = profile.get(f"OL_subject_{OL_SUBJECTS['Maths']}_score", 0)
            ol_science = profile.get(f"OL_subject_{OL_SUBJECTS['Science']}_score", 0)
            ol_english = profile.get(f"OL_subject_{OL_SUBJECTS['English']}_score", 0)

            # Engineering careers
            if career_id == CAREERS["Engineering"]:
                if stream == AL_STREAMS["Physical Science"]:
                    math_score = profile.get("AL_subject_2_score", 0)  # Combined Maths
                    physics_score = profile.get("AL_subject_0_score", 0)  # Physics
                    prob += (math_score / 100) * 0.3 + (physics_score / 100) * 0.2
                # Add small boost for good OL math/science
                prob += (ol_math / 100) * 0.1 + (ol_science / 100) * 0.1

            # Medicine careers
            elif career_id == CAREERS["Medicine"]:
                if stream == AL_STREAMS["Biological Science"]:
                    bio_score = profile.get("AL_subject_0_score", 0)  # Biology
                    chem_score = profile.get("AL_subject_1_score", 0)  # Chemistry
                    prob += (bio_score / 100) * 0.3 + (chem_score / 100) * 0.2
                # Add small boost for good OL science
                prob += (ol_science / 100) * 0.15

            # IT careers
            elif career_id == CAREERS["IT"]:
                if stream in [AL_STREAMS["Physical Science"], AL_STREAMS["Technology"]]:
                    ict_score = profile.get("AL_subject_12_score", 0)  # ICT
                    prob += (ict_score / 100) * 0.4
                # Add small boost for good OL math/science
                prob += (ol_math / 100) * 0.1 + (ol_science / 100) * 0.1

            # Business careers
            elif career_id == CAREERS["Business"]:
                if stream == AL_STREAMS["Commerce"]:
                    accounting = profile.get("AL_subject_4_score", 0)  # Accounting
                    business = profile.get("AL_subject_5_score", 0)  # Business Studies
                    prob += (accounting / 100) * 0.2 + (business / 100) * 0.2

            # Teaching careers
            elif career_id == CAREERS["Teaching"]:
                # All streams can lead to teaching
                if edu_level in [EDUCATION_LEVELS["UNI"]]:
                    prob += (gpa_normalized / 100) * 0.3

            # Research careers
            elif career_id == CAREERS["Research"]:
                if edu_level in [EDUCATION_LEVELS["UNI"]]:
                    prob += (gpa_normalized / 100) * 0.4
                    if stream in [
                        AL_STREAMS["Physical Science"],
                        AL_STREAMS["Biological Science"],
                    ]:
                        prob += 0.2

            # All careers - small boost for good English
            prob += (ol_english / 100) * 0.05

            # Cap probability at base_high
            prob = min(prob, base_high)
            probs[f"career_{career_id}"] = prob

        return probs

    # Generate profiles for each education level
    for _ in range(total_students):
        profile = {"profile_id": _ + 1}
        edu_type = np.random.choice(
            list(education_dist.keys()), p=list(education_dist.values())
        )

        profile["education_level"] = edu_type

        if edu_type in [
            EDUCATION_LEVELS["OL"],
            EDUCATION_LEVELS["AL"],
            EDUCATION_LEVELS["UNI"],
        ]:
            profile.update(generate_ol_results())

        if edu_type in [EDUCATION_LEVELS["AL"], EDUCATION_LEVELS["UNI"]]:
            stream_id = np.random.choice(list(AL_STREAMS.values()))
            profile["AL_stream"] = stream_id
            profile.update(generate_al_results(stream_id))

        if edu_type in [EDUCATION_LEVELS["UNI"]]:
            profile["university_score"] = np.random.randint(60, 100)
            # Generate GPA with most values clustering around 2.8-3.5
            gpa = np.random.normal(3.2, 0.4)
            # Clip to valid GPA range
            gpa = np.clip(gpa, config["gpa_range"]["min"], config["gpa_range"]["max"])
            profile["gpa"] = round(gpa, 2)

        profile.update(generate_career_probs(profile))
        data.append(profile)

    df = pd.DataFrame(data)
    df = df.fillna(-1)

    os.makedirs(config["data_dir"], exist_ok=True)
    df.to_csv(f"{config['data_dir']}/student_profiles.csv", index=False)

    return df


if __name__ == "__main__":
    generator()
