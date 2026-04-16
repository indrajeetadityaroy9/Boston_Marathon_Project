from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class MarathonConfig:
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent)
    data_csv: Path = field(init=False)
    weather_csv: Path = field(init=False)
    metrics_json: Path = field(init=False)
    figures_dir: Path = field(init=False)

    seed: int = 0
    bca_confidence_level: float = 0.95
    n_bootstrap_resamples: int = 1000
    analysis_start_year: int = 2000

    pre_race_train_years: tuple = tuple(range(2000, 2017))
    pre_race_test_years: tuple = (2017,)
    mixed_effects_train_end_year: int = 2016
    in_race_train_years: tuple = (2015, 2016)
    in_race_test_year: int = 2017

    pacing_even_tolerance: float = 0.01
    conformal_alpha: float = 0.10
    conformal_split_ratio: float = 0.20

    def __post_init__(self):
        pr = self.project_root
        self.data_csv = pr / "data" / "processed" / "boston_marathon_cleaned.csv"
        self.weather_csv = pr / "data" / "processed" / "race_day_weather.csv"
        self.metrics_json = pr / "results" / "pipeline_metrics.json"
        self.figures_dir = pr / "results" / "figures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        self.year_center = int(round(sum(self.pre_race_train_years) / len(self.pre_race_train_years)))

        self.cumulative_split_time_columns = [
            "5k_seconds", "10k_seconds", "15k_seconds", "20k_seconds", "half_seconds",
            "25k_seconds", "30k_seconds", "35k_seconds", "40k_seconds",
        ]
        self.checkpoint_labels = [c.replace("_seconds", "").upper() for c in self.cumulative_split_time_columns]
        self.pipeline_columns = ["year", "display_name", "age", "gender", "seconds", "age_imputed", "bib"] + self.cumulative_split_time_columns

        self.age_group_bins = [0, 30, 35, 40, 45, 50, 55, 60, 65, 120]
        self.age_group_labels = ["<30", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60-64", "65+"]
        self.weather_age_group_bins = [0, 35, 40, 45, 50, 55, 60, 65, 70, 75, 120]
        self.weather_age_group_labels = ["<35", "35-39", "40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70-74", "75+"]

        self.pre_race_fixed_effect_features = [
            "age_centered", "age_centered_squared", "female", "year_centered",
            "age_centered_female_interaction", "age_centered_squared_female_interaction",
        ]
        self.segment_distances_m = [5000, 5000, 5000, 5000, 1097.5, 3902.5, 5000, 5000, 5000]
        self.lgbm_deterministic_params = {"deterministic": True, "force_col_wise": True, "num_threads": 1, "verbosity": -1}

    def to_dict(self):
        d = {k: str(v) if isinstance(v, Path) else list(v) if isinstance(v, tuple) else v
             for k, v in asdict(self).items()}
        d["year_center"] = self.year_center
        return d
