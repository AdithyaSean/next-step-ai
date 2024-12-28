"""Enhanced data generator."""

import os

import numpy as np
import pandas as pd

from ..config.config import AL_STREAMS, EDUCATION_LEVELS, config


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

    def generate_career_probs():
        return {
            f"career_{career_id}": np.random.uniform(low, high)
            for career_id, (low, high) in config["career_success_ranges"].items()
        }

    # Generate profiles for each education level
    for _ in range(total_students):
        profile = {"profile_id": _ + 1}
        edu_type = np.random.choice(
            list(education_dist.keys()), p=list(education_dist.values())
        )

        profile["education_level"] = edu_type

        if edu_type in [
            EDUCATION_LEVELS["OL_only"],
            EDUCATION_LEVELS["OL_AL"],
            EDUCATION_LEVELS["OL_AL_UNI"],
        ]:
            profile.update(generate_ol_results())

        if edu_type in [EDUCATION_LEVELS["OL_AL"], EDUCATION_LEVELS["OL_AL_UNI"]]:
            stream_id = np.random.choice(list(AL_STREAMS.values()))
            profile["AL_stream"] = stream_id
            profile.update(generate_al_results(stream_id))

        if edu_type in [EDUCATION_LEVELS["OL_AL_UNI"], EDUCATION_LEVELS["UNI_only"]]:
            profile["university_score"] = np.random.randint(60, 100)

        profile.update(generate_career_probs())
        data.append(profile)

    df = pd.DataFrame(data)
    df = df.fillna(-1)

    os.makedirs(config["data_dir"], exist_ok=True)
    df.to_csv(f"{config['data_dir']}/student_profiles.csv", index=False)

    return df


if __name__ == "__main__":
    generator()
