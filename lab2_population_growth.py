from __future__ import annotations

import csv
from pathlib import Path
from textwrap import dedent

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pyspark.sql import SparkSession, functions as F

DATA_PATH = Path("data.csv")
REPORT_DIR = Path("reports")
IMG_DIR = Path("img")
REPORT_DIR.mkdir(parents=True, exist_ok=True)
IMG_DIR.mkdir(parents=True, exist_ok=True)

spark = (
    SparkSession.builder.appName("RosstatPopulationGrowth")
    .config("spark.sql.execution.arrow.pyspark.enabled", "true")
    .getOrCreate()
)


def _load_records(path: Path) -> list[tuple[str, int, float]]:
    if not path.exists():
        raise FileNotFoundError(f"Не найден файл с данными: {path}")

    with path.open(encoding="utf-8-sig") as fh:
        reader = csv.reader(fh, delimiter=";")
        rows = [row for row in reader if any(cell.strip() for cell in row)]

    if len(rows) < 4:
        raise ValueError("Файл не похож на выгрузку Росстата")

    raw_years = [cell.strip() for cell in rows[1][1:] if cell.strip()]
    years = []
    for cell in raw_years:
        year_part = cell.split()[0]
        if year_part.isdigit():
            years.append(int(year_part))

    records: list[tuple[str, int, float]] = []
    for row in rows[3:]:
        region = row[0].strip()
        if not region:
            continue
        for year, raw_value in zip(years, row[1:]):
            if year < 2018 or year > 2023:
                continue
            value = raw_value.strip().replace("\u00a0", "").replace(" ", "")
            if not value:
                continue
            value = float(value.replace(",", "."))
            records.append((region, year, value))
    return records


records = _load_records(DATA_PATH)
if not records:
    raise ValueError("Не удалось извлечь ни одной записи из csv")

df = (
    spark.createDataFrame(records, schema="Region STRING, Year INT, TotalGrowth DOUBLE")
    .withColumn(
        "Level",
        F.when(F.col("Region") == F.lit("Российская Федерация"), F.lit("country"))
        .when(F.lower(F.col("Region")).contains("федеральный округ"), F.lit("district"))
        .otherwise(F.lit("region")),
    )
)
df.cache()
df.createOrReplaceTempView("growth")

dataset_rows = df.count()
year_stats = df.agg(F.min("Year").alias("min_year"), F.max("Year").alias("max_year")).first()
regions_total = df.select("Region").distinct().count()
districts_total = (
    df.filter(F.col("Level") == "district").select("Region").distinct().count()
)


def render_table(pdf: pd.DataFrame) -> str:
    return pdf.to_markdown(index=False)


report_parts = [
    "# Лабораторная работа №2, Хотинская Арина, ИС-М24",
    "## Описание набора данных",
    f"- Источник: выгрузка Росстата «Общий прирост постоянного населения»",
    f"- Фактически доступный интервал: {year_stats.min_year}–{year_stats.max_year} (в исходной выгрузке нет данных за 2000–2017)",
    f"- Количество строк (Region, Year): {dataset_rows}",
    f"- Уникальных территорий: {regions_total}",
    f"- В том числе федеральных округов: {districts_total}",
]

sample_df = spark.sql(
    """
    SELECT Year, Region, TotalGrowth
    FROM growth
    ORDER BY Year, Region
    LIMIT 10
    """
).toPandas()
report_parts.append("### Пример записей")
report_parts.append(render_table(sample_df))


def save_plot(fig_name: str) -> Path:
    path = IMG_DIR / fig_name
    fig = plt.gcf()
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


sns.set_theme(style="whitegrid")
plt.rcParams.update(
    {
        "savefig.dpi": 300,
        "figure.dpi": 150,
    }
)

# Exercise 1: национальная динамика
ex1_sql = dedent(
    """
    SELECT Year, TotalGrowth
    FROM growth
    WHERE Region = 'Российская Федерация'
    ORDER BY Year
    """
)
ex1_pdf = spark.sql(ex1_sql).toPandas()
plt.figure(figsize=(12, 6))
sns.lineplot(data=ex1_pdf, x="Year", y="TotalGrowth", marker="o")
plt.title("Общий прирост населения РФ, 2018–2023")
plt.ylabel("чел.")
plt.axhline(0, color="black", linewidth=0.8)
ex1_img = save_plot("ex1_total_growth.png")

report_parts.append("## Упражнение 1. Национальная динамика")
report_parts.append("```sql\n" + ex1_sql.strip() + "\n```")
report_parts.append(
    "Линейный график показывает смену знака прироста: после пика 2019 года "
    "страна уходит в устойчивое сокращение населения."
)
report_parts.append(f"![ex1]({ex1_img})")

# Exercise 2: динамика по федеральным округам
ex2_sql = dedent(
    """
    SELECT Year, Region AS District, TotalGrowth
    FROM growth
    WHERE Level = 'district'
    ORDER BY Year, District
    """
)
ex2_pdf = spark.sql(ex2_sql).toPandas()
plt.figure(figsize=(14, 6))
sns.lineplot(data=ex2_pdf, x="Year", y="TotalGrowth", hue="District", marker="o")
plt.title("Динамика прироста по федеральным округам")
plt.ylabel("чел.")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
ex2_img = save_plot("ex2_district_trends.png")

report_parts.append("## Упражнение 2. Федеральные округа во времени")
report_parts.append("```sql\n" + ex2_sql.strip() + "\n```")
report_parts.append(
    "Большинство округов показывают отрицательные значения, за исключением "
    "Северо-Кавказского округа"
)
report_parts.append(f"![ex2]({ex2_img})")

# Exercise 3: тепловая карта округ × год
heatmap_df = ex2_pdf.pivot(index="District", columns="Year", values="TotalGrowth")
plt.figure(figsize=(14, 6))
sns.heatmap(heatmap_df, cmap="RdYlBu", center=0, annot=True, fmt=".0f")
plt.title("Тепловая карта прироста по округам")
ex3_img = save_plot("ex3_district_heatmap.png")

report_parts.append("## Упражнение 3. Тепловая карта округов")
report_parts.append(
    "Тепловая карта усиливает различия: Северный Кавказ и отдельные годы в ЦФО "
    "остаются зонами роста, тогда как Приволжский и Сибирский округа стабильно в минусе."
)
report_parts.append(f"![ex3]({ex3_img})")

# Exercise 4: топ/анти-топ субъектов 2023
ex4_sql = dedent(
    """
    WITH ranked AS (
        SELECT Region,
               TotalGrowth,
               DENSE_RANK() OVER (ORDER BY TotalGrowth DESC) AS rk_pos,
               DENSE_RANK() OVER (ORDER BY TotalGrowth ASC)  AS rk_neg
        FROM growth
        WHERE Level = 'region' AND Year = 2023
    )
    SELECT CASE WHEN rk_pos <= 5 THEN 'Top-5' ELSE 'Bottom-5' END AS Segment,
           Region,
           TotalGrowth
    FROM ranked
    WHERE rk_pos <= 5 OR rk_neg <= 5
    ORDER BY Segment, TotalGrowth DESC
    """
)
ex4_pdf = spark.sql(ex4_sql).toPandas()
plt.figure(figsize=(12, 6))
sns.barplot(
    data=ex4_pdf,
    y="Region",
    x="TotalGrowth",
    hue="Segment",
    dodge=False,
)
plt.title("Топ/анти-топ субъектов по приросту (2023)")
plt.xlabel("чел.")
ex4_img = save_plot("ex4_regions_2023.png")

report_parts.append("## Упражнение 4. Крайние регионы 2023 года")
report_parts.append("```sql\n" + ex4_sql.strip() + "\n```")
report_parts.append(render_table(ex4_pdf))
report_parts.append(
    "Горизонтальная диаграмма подчёркивает сверхпозитивную динамику в Москве и "
    "Московской области на фоне глубокого спада в большинстве регионов Центральной России."
)
report_parts.append(f"![ex4]({ex4_img})")

# Exercise 5: доля положительных лет по округам
ex5_sql = dedent(
    """
    SELECT Region AS District,
           SUM(CASE WHEN TotalGrowth > 0 THEN 1 ELSE 0 END) AS PositiveYears,
           COUNT(*) AS TotalYears,
           ROUND(SUM(CASE WHEN TotalGrowth > 0 THEN 1 ELSE 0 END) / COUNT(*), 3)
               AS PositiveShare
    FROM growth
    WHERE Level = 'district'
    GROUP BY Region
    ORDER BY PositiveShare DESC
    """
)
ex5_pdf = spark.sql(ex5_sql).toPandas()
plt.figure(figsize=(12, 6))
sns.barplot(data=ex5_pdf, x="PositiveShare", y="District", orient="h")
plt.title("Доля лет с положительным приростом (2018–2023)")
plt.xlabel("Доля положительных лет")
ex5_img = save_plot("ex5_positive_share.png")

report_parts.append("## Упражнение 5. Устойчивость прироста по округам")
report_parts.append("```sql\n" + ex5_sql.strip() + "\n```")
report_parts.append(render_table(ex5_pdf))
report_parts.append(
    "Северо-Кавказский округ остаётся единственным регионом, где все годы выборки "
    "показывают положительный прирост, тогда как большинство округов имеют долю 0."
)
report_parts.append(f"![ex5]({ex5_img})")

report_path = REPORT_DIR / "lab2_report.md"
report_path.write_text("\n\n".join(report_parts), encoding="utf-8")
print(f"Отчёт сохранён в {report_path}")

spark.stop()