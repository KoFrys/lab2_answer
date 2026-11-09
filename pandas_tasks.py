"""Задачи первой части лабораторной: Pandas и Titanic."""
from __future__ import annotations

import pandas as pd

from grader_contracts.pandas_tasks import TitanicInput, TitanicSummary


def analyze_titanic(data: TitanicInput) -> TitanicSummary:
    """Выполните загрузку и анализ датасета Titanic.

    Нужно: посчитать пропуски, число пассажиров старше 30 лет, средний возраст
    и долю выживших по классам, а также пять наибольших тарифов по убыванию.
    """
    df = pd.read_csv(data.csv_path)

    # Пропуски по столбцам
    missing_by_column = {}
    missing_counts = df.isna().sum()
    for column, count in missing_counts.items():
        missing_by_column[str(column)] = int(count)

    # NaN > 30 даёт False, поэтому пропуски возраста не учитываются
    adults_over_30_count = int((df["Age"] > 30).sum())

    # mean() игнорирует NaN, возраст 0 никто не подставляет
    mean_age_by_pclass = {}
    mean_ages = df.groupby("Pclass")["Age"].mean()
    for pclass, mean_age in mean_ages.items():
        mean_age_by_pclass[int(pclass)] = float(mean_age)

    # Доля выживших по классам
    survival_rate_by_pclass = {}
    survival_rates = df.groupby("Pclass")["Survived"].mean()
    for pclass, rate in survival_rates.items():
        survival_rate_by_pclass[int(pclass)] = float(rate)

    # Пять наибольших тарифов по убыванию
    highest_fares = []
    top_fares = df["Fare"].nlargest(5)
    for fare in top_fares:
        highest_fares.append(float(fare))

    return TitanicSummary(
        row_count=int(len(df)),
        missing_by_column=missing_by_column,
        adults_over_30_count=adults_over_30_count,
        mean_age_by_pclass=mean_age_by_pclass,
        survival_rate_by_pclass=survival_rate_by_pclass,
        highest_fares=highest_fares,
    )
