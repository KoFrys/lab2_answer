"""Задачи второй части лабораторной: корреляционный анализ."""
from __future__ import annotations

import pandas as pd

from grader_contracts.correlation_tasks import BrainCorrelationSummary, BrainDataInput

FEATURES = ["FSIQ", "VIQ", "PIQ", "Weight", "Height"]
TARGET = "MRI_Count"


def _correlations(group: pd.DataFrame) -> dict[str, float]:
    """Корреляция Пирсона каждого признака с MRI_Count внутри группы."""
    result = {}
    for feature in FEATURES:
        pair = group[[feature, TARGET]].dropna()  # попарное удаление NA
        result[feature] = float(pair[feature].corr(pair[TARGET], method="pearson"))
    return result


def analyze_brain_correlations(data: BrainDataInput) -> BrainCorrelationSummary:
    """Проанализируйте brainsize.txt.

    Разделите наблюдения по полу и для каждой группы вычислите корреляции
    признаков FSIQ, VIQ, PIQ, Weight, Height с MRI_Count методом Пирсона.
    В strongest_mri_feature верните название признака с наибольшим модулем
    корреляции с MRI_Count среди объединённых результатов двух групп.
    """
    df = pd.read_csv(data.csv_path, sep="\t", na_values=["NA"])
    df["Gender"] = df["Gender"].str.strip()

    men = df[df["Gender"] == "Male"]
    women = df[df["Gender"] == "Female"]

    men_corr = _correlations(men)
    women_corr = _correlations(women)

    # Ищем признак с максимальным модулем корреляции среди обеих групп
    strongest_mri_feature = ""
    strongest_value = -1.0
    for corr in (men_corr, women_corr):
        for feature, value in corr.items():
            if abs(value) > strongest_value:
                strongest_value = abs(value)
                strongest_mri_feature = feature

    return BrainCorrelationSummary(
        men_count=int(len(men)),
        women_count=int(len(women)),
        women_mri_correlation=women_corr,
        men_mri_correlation=men_corr,
        strongest_mri_feature=strongest_mri_feature,
    )
