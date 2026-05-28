from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
TABLES_DIR = ROOT / "outputs" / "tables"
PLOTS_DIR = ROOT / "outputs" / "plots"

DATA_URL = "https://archive.ics.uci.edu/static/public/942/data.csv"
RAW_PATH = RAW_DIR / "rt_iot2022.csv"

NORMAL_LABELS = {"Thing_Speak", "MQTT_Publish", "Wipro_bulb"}
PLOT_FEATURES = [
    "flow_duration",
    "flow_pkts_per_sec",
    "payload_bytes_per_second",
    "fwd_init_window_size",
]
REGRESSION_X = "flow_pkts_per_sec"
REGRESSION_Y = "payload_bytes_per_second"
TTEST_FEATURE = "payload_bytes_per_second"
NORMALITY_FEATURE = "flow_duration"
RANDOM_STATE = 42
PLOT_SAMPLE_SIZE = 5000
TEST_SAMPLE_SIZE = 5000


# Create the output directory structure used by the analysis.
def ensure_dirs() -> None:
    for path in (RAW_DIR, TABLES_DIR, PLOTS_DIR):
        path.mkdir(parents=True, exist_ok=True)


# Load the dataset from disk, or download and cache it on first run.
def load_dataset() -> pd.DataFrame:
    if RAW_PATH.exists():
        return pd.read_csv(RAW_PATH)

    df = pd.read_csv(DATA_URL)
    df.to_csv(RAW_PATH, index=False)
    return df


# Add binary labels and clean categorical types for later analysis.
def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    prepared["is_attack"] = (~prepared["Attack_type"].isin(NORMAL_LABELS)).astype(int)
    prepared["traffic_class"] = prepared["is_attack"].map({0: "Normal", 1: "Attack"})
    prepared["proto"] = prepared["proto"].astype("category")
    prepared["service"] = prepared["service"].astype("category")
    return prepared


# Split columns into numeric and categorical feature groups.
def get_feature_sets(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_cols = [
        col
        for col in df.select_dtypes(include=[np.number]).columns
        if col not in {"id", "is_attack"}
    ]
    categorical_cols = [
        col
        for col in df.columns
        if col not in numeric_cols and col not in {"id"}
    ]
    return numeric_cols, categorical_cols


# Save basic dataset overview tables and categorical frequency summaries.
def save_overview_tables(
    df: pd.DataFrame, numeric_cols: list[str], categorical_cols: list[str]
) -> None:
    original_columns = {"id", "Attack_type", "proto", "service"} | set(numeric_cols)
    overview = pd.DataFrame(
        {
            "metric": [
                "observations",
                "original_columns_total",
                "derived_columns_added",
                "analyzed_columns_total",
                "numeric_features",
                "categorical_features",
                "missing_values_total",
                "attack_share",
                "normal_share",
            ],
            "value": [
                len(df),
                len(original_columns),
                2,
                len(df.columns),
                len(numeric_cols),
                len(categorical_cols),
                int(df.isna().sum().sum()),
                df["is_attack"].mean(),
                1 - df["is_attack"].mean(),
            ],
        }
    )
    overview.to_csv(TABLES_DIR / "overview.csv", index=False)

    dtypes = pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(dtype) for dtype in df.dtypes],
            "missing_values": [int(df[col].isna().sum()) for col in df.columns],
            "unique_values": [int(df[col].nunique(dropna=False)) for col in df.columns],
        }
    )
    dtypes.to_csv(TABLES_DIR / "column_types.csv", index=False)

    descriptive = df[numeric_cols].describe().T
    descriptive["variance"] = df[numeric_cols].var()
    descriptive.to_csv(TABLES_DIR / "descriptive_statistics.csv")

    category_frames = []
    for column in ["traffic_class", "Attack_type", "proto", "service"]:
        counts = (
            df[column]
            .value_counts(dropna=False)
            .rename_axis("category")
            .reset_index(name="count")
        )
        counts["share"] = counts["count"] / len(df)
        counts["variable"] = column
        category_frames.append(counts[["variable", "category", "count", "share"]])
    pd.concat(category_frames, ignore_index=True).to_csv(
        TABLES_DIR / "categorical_frequencies.csv", index=False
    )


# Compute correlations with the attack label and top feature-to-feature links.
def save_correlation_tables(df: pd.DataFrame, numeric_cols: list[str]) -> None:
    corr_target = df[numeric_cols + ["is_attack"]].corr(numeric_only=True)["is_attack"]
    corr_target = corr_target.drop("is_attack").sort_values(
        key=lambda s: s.abs(), ascending=False
    )
    corr_table = pd.DataFrame(
        {
            "feature": corr_target.index,
            "correlation_with_is_attack": corr_target.values,
            "abs_correlation": corr_target.abs().values,
        }
    )
    corr_table.to_csv(TABLES_DIR / "correlation_with_attack.csv", index=False)

    corr = df[numeric_cols].corr(numeric_only=True).abs().copy()
    upper_mask = np.triu(np.ones(corr.shape, dtype=bool))
    pairwise = (
        corr.mask(upper_mask)
        .stack()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"level_0": "feature_a", "level_1": "feature_b", 0: "correlation"})
    )
    pairwise.to_csv(TABLES_DIR / "top_feature_correlations.csv", index=False)


# Detect outliers with the interquartile range rule for each numeric feature.
def save_outlier_table(df: pd.DataFrame, numeric_cols: list[str]) -> None:
    q1 = df[numeric_cols].quantile(0.25)
    q3 = df[numeric_cols].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outlier_mask = (df[numeric_cols] < lower) | (df[numeric_cols] > upper)

    outliers = pd.DataFrame(
        {
            "feature": numeric_cols,
            "q1": q1.values,
            "q3": q3.values,
            "iqr": iqr.values,
            "lower_bound": lower.values,
            "upper_bound": upper.values,
            "outlier_count": outlier_mask.sum().values,
        }
    )
    outliers["outlier_share"] = outliers["outlier_count"] / len(df)
    outliers = outliers.sort_values("outlier_count", ascending=False)
    outliers.to_csv(TABLES_DIR / "outliers_iqr.csv", index=False)


# Run the required hypothesis tests for mean comparison and normality.
def run_hypothesis_tests(df: pd.DataFrame) -> None:
    attack_values = df.loc[df["is_attack"] == 1, TTEST_FEATURE]
    normal_values = df.loc[df["is_attack"] == 0, TTEST_FEATURE]
    t_stat, t_pvalue = stats.ttest_ind(
        attack_values,
        normal_values,
        equal_var=False,
        nan_policy="omit",
    )

    pooled_std = np.sqrt(
        (
            attack_values.var(ddof=1) * (len(attack_values) - 1)
            + normal_values.var(ddof=1) * (len(normal_values) - 1)
        )
        / (len(attack_values) + len(normal_values) - 2)
    )
    cohens_d = (attack_values.mean() - normal_values.mean()) / pooled_std

    normality_sample = (
        df.loc[df["traffic_class"] == "Normal", NORMALITY_FEATURE]
        .sample(n=min(TEST_SAMPLE_SIZE, (df["traffic_class"] == "Normal").sum()), random_state=RANDOM_STATE)
        .astype(float)
    )
    k2_stat, normality_pvalue = stats.normaltest(normality_sample)

    tests = pd.DataFrame(
        [
            {
                "test_name": "Welch t-test",
                "feature": TTEST_FEATURE,
                "null_hypothesis": "mean attack traffic equals mean normal traffic",
                "statistic": t_stat,
                "p_value": t_pvalue,
                "decision_alpha_0_05": "reject" if t_pvalue < 0.05 else "fail_to_reject",
                "group_1_mean": attack_values.mean(),
                "group_2_mean": normal_values.mean(),
                "effect_size_cohens_d": cohens_d,
            },
            {
                "test_name": "D'Agostino K^2 normality test",
                "feature": NORMALITY_FEATURE,
                "null_hypothesis": "normal traffic follows a normal distribution",
                "statistic": k2_stat,
                "p_value": normality_pvalue,
                "decision_alpha_0_05": "reject" if normality_pvalue < 0.05 else "fail_to_reject",
                "group_1_mean": normality_sample.mean(),
                "group_2_mean": np.nan,
                "effect_size_cohens_d": np.nan,
            },
        ]
    )
    tests.to_csv(TABLES_DIR / "hypothesis_tests.csv", index=False)


# Fit and save a simple linear regression between two selected features.
def run_regression(df: pd.DataFrame) -> None:
    regression_df = (
        df[[REGRESSION_X, REGRESSION_Y]]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .astype(float)
    )
    result = stats.linregress(regression_df[REGRESSION_X], regression_df[REGRESSION_Y])

    regression_summary = pd.DataFrame(
        [
            {
                "x_feature": REGRESSION_X,
                "y_feature": REGRESSION_Y,
                "slope": result.slope,
                "intercept": result.intercept,
                "r_value": result.rvalue,
                "r_squared": result.rvalue**2,
                "p_value": result.pvalue,
                "std_err": result.stderr,
            }
        ]
    )
    regression_summary.to_csv(TABLES_DIR / "linear_regression_summary.csv", index=False)


# Plot the class balance and the most frequent traffic or attack labels.
def plot_class_distribution(df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="traffic_class", hue="traffic_class", palette="Set2", legend=False)
    plt.title("Traffic Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "class_distribution.png", dpi=200)
    plt.close()

    top_attacks = df["Attack_type"].value_counts().head(10)
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=top_attacks.values,
        y=top_attacks.index,
        hue=top_attacks.index,
        palette="crest",
        legend=False,
    )
    plt.title("Top 10 Attack Types / Traffic Sources")
    plt.xlabel("Count")
    plt.ylabel("Attack_type")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "top_attack_types.png", dpi=200)
    plt.close()


# Plot trimmed histograms with KDE to show the main distribution shapes.
def plot_histograms(df: pd.DataFrame) -> None:
    sample = df.sample(n=min(PLOT_SAMPLE_SIZE, len(df)), random_state=RANDOM_STATE)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.ravel()

    for ax, feature in zip(axes, PLOT_FEATURES, strict=True):
        feature_sample = sample[feature].astype(float)
        upper_limit = feature_sample.quantile(0.99)
        sns.histplot(feature_sample, bins=40, kde=True, ax=ax, color="#2a9d8f")
        ax.set_title(f"{feature} (trimmed to 99th pct for reading)")
        ax.set_xlim(feature_sample.min(), upper_limit)
        ax.set_xlabel(feature)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "histograms_with_kde.png", dpi=200)
    plt.close()


# Plot boxplots by traffic class for the selected key features.
def plot_boxplots(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.ravel()

    for ax, feature in zip(axes, PLOT_FEATURES, strict=True):
        plot_df = df[["traffic_class", feature]].copy()
        plot_df["plot_value"] = np.log1p(plot_df[feature].astype(float))

        sns.boxplot(
            data=plot_df,
            x="traffic_class",
            y="plot_value",
            hue="traffic_class",
            ax=ax,
            palette="Set2",
            legend=False,
            showfliers=False,
        )
        ax.set_title(f"{feature} by class (log1p scale)")
        ax.set_xlabel("Traffic class")
        ax.set_ylabel(f"log(1 + {feature})")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "boxplots_by_class.png", dpi=200)
    plt.close()


# Plot the regression scatter chart with a fitted linear trend line.
def plot_scatter_with_regression(df: pd.DataFrame) -> None:
    sample = df.sample(n=min(PLOT_SAMPLE_SIZE, len(df)), random_state=RANDOM_STATE).copy()
    x_upper = sample[REGRESSION_X].quantile(0.99)
    y_upper = sample[REGRESSION_Y].quantile(0.99)

    plt.figure(figsize=(9, 6))
    sns.scatterplot(
        data=sample,
        x=REGRESSION_X,
        y=REGRESSION_Y,
        hue="traffic_class",
        alpha=0.5,
        s=25,
        palette="Set2",
    )
    sns.regplot(
        data=sample,
        x=REGRESSION_X,
        y=REGRESSION_Y,
        scatter=False,
        color="black",
        line_kws={"linewidth": 2},
    )
    plt.xlim(sample[REGRESSION_X].min(), x_upper)
    plt.ylim(sample[REGRESSION_Y].min(), y_upper)
    plt.title("Linear relationship between packet rate and payload throughput")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "scatter_regression.png", dpi=200)
    plt.close()


# Orchestrate the full workflow from data loading to saved outputs.
def main() -> None:
    ensure_dirs()
    raw_df = load_dataset()
    df = prepare_dataset(raw_df)
    numeric_cols, categorical_cols = get_feature_sets(df)

    save_overview_tables(df, numeric_cols, categorical_cols)
    save_correlation_tables(df, numeric_cols)
    save_outlier_table(df, numeric_cols)
    run_hypothesis_tests(df)
    run_regression(df)

    sns.set_theme(style="whitegrid")
    plot_class_distribution(df)
    plot_histograms(df)
    plot_boxplots(df)
    plot_scatter_with_regression(df)

    print("Analysis complete.")
    print(f"Raw data: {RAW_PATH}")
    print(f"Tables: {TABLES_DIR}")
    print(f"Plots: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
