# %% [markdown]
# # Pearson correlations across profile types
#
# Correlate features across matched well aggregates, per patient/tumor and pooled.

# %%
import pathlib

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from notebook_init_utils import init_notebook
from scipy.stats import pearsonr

root_dir, in_notebook = init_notebook()

# %%
# Define the comparisons to run
PROJECTIONS = ("max_projection",)

# profile type: (2D handcrafted resolution, 3D file prefix, plot label)
# MorphEM uses projected 2D crops but is stored under profiles_3D.
PROFILE_TYPES = {
    "organoid_handcrafted": ("organoid", "organoid", "Organoid / ZedProfiler"),
    "organoid_sammed": ("organoid", "sammed_organoid", "Organoid / SAM-Med3D"),
    "sc_handcrafted": ("sc", "sc", "Single-cell / ZedProfiler"),
    "sc_sammed": ("sc", "sammed_sc", "Single-cell / SAM-Med3D"),
    "sc_sammed_nucleocentric": (
        "sc",
        "sammed_nucleocentric",
        "Single-cell / Nucleocentric SAM-Med3D",
    ),
    "sc_nucleocentric_morphem": (
        "sc",
        "nucleocentric_morphem",
        "Single-cell / Nucleocentric MorphEM",
    ),
}

PATIENT_TUMORS = None  # None = every shared patient/tumor; or a list of IDs.
INCLUDE_ALL_PATIENTS = True  # Pools every shared, non-excluded sample.
EXCLUDED_PATIENT_TUMORS = ("NF0037_T1_CQ1",)
OUTLIER_CUTOFF = 100
MIN_PAIRS = 3  # A validity floor, not evidence of a reliable estimate.
FEATURE_BLOCK_SIZE = 128

# Wells are matched on these keys, and these annotations travel with them
MATCH_KEYS = ["Metadata_patient_tumor", "Metadata_Well"]
ANNOTATION_COLUMNS = ["Metadata_treatment", "Metadata_dose", "Metadata_dose_unit"]

# The 3D profiles use a different metadata naming convention than the 2D ones
METADATA_3D_RENAME = {
    "Metadata_Biology_PatientTumor": "Metadata_patient_tumor",
    "Metadata_Experiment_Well": "Metadata_Well",
    "Metadata_Experiment_Treatment": "Metadata_treatment",
    "Metadata_Experiment_Dose": "Metadata_dose",
    "Metadata_Experiment_Unit": "Metadata_dose_unit",
}

# %%
# Define input paths
# 2D profiles (one sub-directory per projection)
input_2d_dir = pathlib.Path(f"{root_dir}/data/profiles_2D/all_patients").resolve(
    strict=True
)

# 3D profiles
input_3d_dir = pathlib.Path(
    f"{root_dir}/data/profiles_3D/all_patients/2.aggregated_profiles"
).resolve(strict=True)

# Define output paths
results_dir = pathlib.Path(
    f"{root_dir}/2.2d_vs_3d_analysis/results/correlation"
).resolve()
results_dir.mkdir(parents=True, exist_ok=True)

manifest_path = results_dir / "correlation_manifest.parquet"
matching_summary_path = results_dir / "matching_summary.parquet"
feature_summary_path = results_dir / "feature_cleanup_summary.parquet"


# %% [markdown]
# ## Clean profiles and match wells


# %%
def load_profile(path: pathlib.Path, dimension: str) -> tuple[pd.DataFrame, dict]:
    """
    Load a well-aggregated profile and apply the same feature filters used in
    the EDA, then report what was dropped.

    Parameters
    ----------
    path : pathlib.Path
        Path to the aggregated profile parquet file.
    dimension : str
        Either "2D" or "3D". 3D metadata columns are renamed to the 2D
        convention so both profiles can be matched on the same keys.

    Returns
    ----------
    tuple[pd.DataFrame, dict]
        The cleaned profile (metadata columns followed by feature columns) and
        a summary of the feature cleanup.
    """
    df = pd.read_parquet(path)
    if dimension == "3D":
        df = df.rename(columns=METADATA_3D_RENAME)

    # Drop excluded samples before counting anything
    n_input_wells = len(df)
    df = df.loc[~df["Metadata_patient_tumor"].isin(EXCLUDED_PATIENT_TUMORS)].copy()

    features = [col for col in df if not col.startswith("Metadata_")]
    texture_features = [col for col in features if "_Texture_" in col]

    # Texture features are dropped up front, then anything non-numeric or
    # infinite becomes NaN so the correlations only see real measurements
    values = (
        df[[col for col in features if col not in texture_features]]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )

    # Same magnitude rule as pycytominer's drop_outliers operation in the EDA
    extreme_features = values.columns[values.abs().gt(OUTLIER_CUTOFF).any()]
    values = values.drop(columns=extreme_features)

    # A feature with one value (or none) cannot correlate with anything
    uninformative_features = values.columns[values.nunique(dropna=True).le(1)]
    values = values.drop(columns=uninformative_features)

    metadata = df[[col for col in df if col.startswith("Metadata_")]]

    summary = {
        "input_path": str(path.relative_to(root_dir)),
        "dimension": dimension,
        "n_input_wells": n_input_wells,
        "n_excluded_wells": n_input_wells - len(df),
        "n_features_input": len(features),
        "n_texture_dropped": len(texture_features),
        "n_outlier_dropped": len(extreme_features),
        "n_constant_or_empty_dropped": len(uninformative_features),
        "n_features_retained": values.shape[1],
    }

    return pd.concat([metadata, values], axis=1).reset_index(drop=True), summary


def match_profiles(
    df_2d: pd.DataFrame, df_3d: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Align 2D and 3D well aggregates one-to-one on patient/tumor and well, and
    audit which wells matched.

    Parameters
    ----------
    df_2d : pd.DataFrame
        Cleaned 2D profile data with MATCH_KEYS and ANNOTATION_COLUMNS.
    df_3d : pd.DataFrame
        Cleaned 3D profile data with the same metadata structure.

    Returns
    ----------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        The 2D and 3D profiles restricted to shared wells and in the same row
        order, plus the full outer-join audit of matched and unmatched wells.
    """
    # match_status records whether each well is in both, 2D only, or 3D only
    audit = df_2d[MATCH_KEYS + ANNOTATION_COLUMNS].merge(
        df_3d[MATCH_KEYS + ANNOTATION_COLUMNS],
        on=MATCH_KEYS,
        how="outer",
        suffixes=("_2d", "_3d"),
        indicator="match_status",
        validate="one_to_one",
        sort=True,
    )

    shared = audit.loc[audit["match_status"].eq("both")]
    if len(shared) == 0:
        raise ValueError("No shared patient/tumor-well keys between 2D and 3D.")

    # Index both profiles by the shared keys so the rows line up
    keys = pd.MultiIndex.from_frame(shared[MATCH_KEYS])
    aligned_2d = df_2d.set_index(MATCH_KEYS).loc[keys].reset_index()
    aligned_3d = df_3d.set_index(MATCH_KEYS).loc[keys].reset_index()

    assert list(aligned_2d[MATCH_KEYS].apply(tuple, axis=1)) == list(
        aligned_3d[MATCH_KEYS].apply(tuple, axis=1)
    ), "Wells are not aligned after matching"

    return aligned_2d, aligned_3d, audit


# %% [markdown]
# ## Calculate Pearson correlations and valid pair counts


# %%
def standardize(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Center each column and scale it to unit norm, ignoring missing values.
    This is NA aware!

    Missing values are filled with 0 after centering, which drops them from the
    dot products used to build the correlations. The returned mask records where
    the real observations were, so pair counts can be recovered per feature pair.

    Parameters
    ----------
    values : np.ndarray
        The input array to be normalized in the following dimensionality: (nxm)

    Returns
    ----------
    tuple[np.ndarray, np.ndarray]
        The centered, unit-norm array of the same dimensionality (nxm), and the
        matching finite-observation mask (nxm) as floats.
    """
    finite = np.isfinite(values)
    counts = finite.sum(axis=0)
    means = np.divide(
        np.where(finite, values, 0).sum(axis=0),
        counts,
        out=np.zeros(values.shape[1]),
        where=counts > 0,
    )
    centered = np.where(finite, values - means, 0)
    norms = np.linalg.norm(centered, axis=0)

    # Repeated decimals can have tiny nonzero residuals after mean subtraction,
    # so constant columns are found on the raw values instead of their norm
    constant = np.where(finite, values, np.inf).min(axis=0) >= np.where(
        finite, values, -np.inf
    ).max(axis=0)
    norms[constant] = 0

    centered = np.divide(centered, norms, out=np.zeros_like(centered), where=norms > 0)
    return centered, finite.astype(float)


def correlation_blocks(
    features_2d: pd.DataFrame,
    features_3d: pd.DataFrame,
    min_pairs: int = MIN_PAIRS,
    block_size: int = FEATURE_BLOCK_SIZE,
):
    """
    Yield long-form Pearson correlations and their finite pair counts, a block
    of 2D features at a time.

    The full matrix of every 2D feature against every 3D feature is too large to
    hold at once, so it is built with matrix products over blocks of 2D features
    and streamed out block by block.

    Parameters
    ----------
    features_2d : pd.DataFrame
        2D feature columns, one row per matched well.
    features_3d : pd.DataFrame
        3D feature columns, with the same rows in the same order.
    min_pairs : int, optional
        Minimum number of finite pairs a correlation needs to be reported.
    block_size : int, optional
        Number of 2D features correlated per block.

    Yields
    ----------
    pd.DataFrame
        Long-form feature_2d, feature_3d, pearson_r, and n_pairs for one block.
    """
    assert len(features_2d) == len(features_3d), (
        "Feature tables must have the same aligned observations"
    )

    raw_x = features_2d.to_numpy(dtype=float)
    raw_y = features_3d.to_numpy(dtype=float)
    x, mask_x = standardize(raw_x)
    y, mask_y = standardize(raw_y)

    # With no missing values every pair shares every well, which allows the
    # cheaper form below
    complete = bool(mask_x.all() and mask_y.all())

    for start in range(0, x.shape[1], block_size):
        block = slice(start, start + block_size)
        xb, mask_xb = x[:, block], mask_x[:, block]

        if complete:
            counts = np.full((xb.shape[1], y.shape[1]), len(x), dtype=np.int32)
            numerator = xb.T @ y
            var_x = (xb**2).sum(axis=0)[:, None]
            var_y = (y**2).sum(axis=0)[None, :]
        else:
            # Each feature pair is centered on its own overlap
            counts = (mask_xb.T @ mask_y).astype(np.int32)
            safe_counts = np.maximum(counts, 1)
            sums_x, sums_y = xb.T @ mask_y, mask_xb.T @ y
            numerator = xb.T @ y - sums_x * sums_y / safe_counts
            var_x = (xb**2).T @ mask_y - sums_x**2 / safe_counts
            var_y = mask_xb.T @ (y**2) - sums_y**2 / safe_counts

        denominator = np.sqrt(np.maximum(var_x, 0) * np.maximum(var_y, 0))
        valid = (counts >= min_pairs) & (denominator > 0)
        correlations = np.divide(
            numerator,
            denominator,
            out=np.full(numerator.shape, np.nan),
            where=valid,
        )

        if not complete:
            # A feature may be constant only on its overlap with another.
            # Recalculate cancellation-prone pairs directly on that overlap.
            unstable = (
                (counts >= min_pairs)
                & ((var_x < 1e-12) | (var_y < 1e-12))
                & (xb != 0).any(axis=0)[:, None]
                & (y != 0).any(axis=0)[None, :]
            )
            for i, j in zip(*np.where(unstable)):
                paired = (mask_xb[:, i] > 0) & (mask_y[:, j] > 0)
                a, b = raw_x[paired, start + i], raw_y[paired, j]
                correlations[i, j] = (
                    pearsonr(a, b).statistic
                    if np.ptp(a) > 0 and np.ptp(b) > 0
                    else np.nan
                )

        yield pd.DataFrame(
            {
                "feature_2d": np.repeat(features_2d.columns[block], y.shape[1]),
                "feature_3d": np.tile(features_3d.columns, xb.shape[1]),
                "pearson_r": np.clip(correlations, -1, 1).ravel(),
                "n_pairs": counts.ravel(),
            }
        )


def save_correlation(
    features_2d: pd.DataFrame, features_3d: pd.DataFrame, output_path: pathlib.Path
) -> dict:
    """
    Stream a complete 2D vs. 3D correlation matrix to a long-form parquet file.

    Parameters
    ----------
    features_2d : pd.DataFrame
        2D feature columns, one row per matched well.
    features_3d : pd.DataFrame
        3D feature columns, with the same rows in the same order.
    output_path : pathlib.Path
        Path to save the long-form parquet file to.

    Returns
    ----------
    dict
        A summary of the saved matrix, for the manifest.
    """
    schema = pa.schema(
        [
            ("feature_2d", pa.string()),
            ("feature_3d", pa.string()),
            ("pearson_r", pa.float64()),
            ("n_pairs", pa.int32()),
        ]
    )

    n_defined = 0
    min_n_pairs, max_n_pairs = len(features_2d), 0

    # Written block by block so the whole matrix never has to be in memory
    with pq.ParquetWriter(output_path, schema, compression="zstd") as writer:
        for block in correlation_blocks(
            features_2d,
            features_3d,
            min_pairs=MIN_PAIRS,
            block_size=FEATURE_BLOCK_SIZE,
        ):
            writer.write_table(
                pa.Table.from_pandas(block, schema=schema, preserve_index=False)
            )
            n_defined += int(block["pearson_r"].notna().sum())
            min_n_pairs = min(min_n_pairs, int(block["n_pairs"].min()))
            max_n_pairs = max(max_n_pairs, int(block["n_pairs"].max()))

    return {
        "n_features_2d": features_2d.shape[1],
        "n_features_3d": features_3d.shape[1],
        "n_defined_correlations": n_defined,
        "min_n_pairs": min_n_pairs,
        "max_n_pairs": max_n_pairs,
    }


# %% [markdown]
# ## Save correlations and matching summaries

# %%
# Profiles are cached because the same file is reused across profile types
profile_cache = {}
feature_summaries = []
matching_summaries = []
manifest = []

for projection in PROJECTIONS:
    for profile_type, (resolution, prefix, label) in PROFILE_TYPES.items():
        path_2d = input_2d_dir / projection / f"{resolution}_agg_profiles.parquet"
        path_3d = input_3d_dir / f"{prefix}_norm_sc_agg_profiles.parquet"

        for path, dimension in ((path_2d, "2D"), (path_3d, "3D")):
            if path not in profile_cache:
                profile_cache[path], summary = load_profile(path, dimension)
                feature_summaries.append(summary)

        aligned_2d, aligned_3d, audit = match_profiles(
            profile_cache[path_2d], profile_cache[path_3d]
        )

        # Save the well-matching audit next to this comparison's correlations
        pair_dir = results_dir / projection / profile_type
        pair_dir.mkdir(parents=True, exist_ok=True)
        audit.to_parquet(pair_dir / "matched_wells.parquet", index=False)

        for patient_tumor, group in audit.groupby("Metadata_patient_tumor"):
            counts = group["match_status"].value_counts()
            matching_summaries.append(
                {
                    "projection": projection,
                    "profile_type": profile_type,
                    "patient_tumor": patient_tumor,
                    "n_matched_wells": int(counts.get("both", 0)),
                    "n_2d_only": int(counts.get("left_only", 0)),
                    "n_3d_only": int(counts.get("right_only", 0)),
                }
            )

        # Correlate each patient/tumor on its own, and optionally all pooled
        shared_patient_tumors = sorted(aligned_2d["Metadata_patient_tumor"].unique())
        if PATIENT_TUMORS is not None:
            missing = set(PATIENT_TUMORS) - set(shared_patient_tumors)
            if missing:
                raise ValueError(
                    f"{projection}/{profile_type}: unmatched IDs {missing}."
                )
            shared_patient_tumors = [
                patient_tumor
                for patient_tumor in shared_patient_tumors
                if patient_tumor in PATIENT_TUMORS
            ]

        cohorts = ["all_patients"] if INCLUDE_ALL_PATIENTS else []
        cohorts = cohorts + shared_patient_tumors

        features_2d = [col for col in aligned_2d if not col.startswith("Metadata_")]
        features_3d = [col for col in aligned_3d if not col.startswith("Metadata_")]

        for cohort in cohorts:
            if cohort == "all_patients":
                selected = np.ones(len(aligned_2d), dtype=bool)
            else:
                selected = aligned_2d["Metadata_patient_tumor"].eq(cohort).to_numpy()

            output_path = pair_dir / f"{cohort}_correlation.parquet"
            summary = save_correlation(
                aligned_2d.loc[selected, features_2d],
                aligned_3d.loc[selected, features_3d],
                output_path,
            )

            manifest.append(
                {
                    "projection": projection,
                    "profile_type": profile_type,
                    "profile_label": label,
                    "cohort": cohort,
                    "analysis_unit": "patient_tumor_well_aggregate",
                    "n_matched_wells": int(selected.sum()),
                    "min_pairs_required": MIN_PAIRS,
                    "correlation_file": str(output_path.relative_to(results_dir)),
                    **summary,
                }
            )

            print(
                f"{projection} / {profile_type} / {cohort}: "
                f"{selected.sum()} matched wells; "
                f"{summary['n_defined_correlations']:,} defined correlations",
                flush=True,
            )

# %%
# Save the manifest notebook 2 plots from, plus the cleanup and matching audits
correlation_manifest = pd.DataFrame(manifest)

pd.DataFrame(feature_summaries).to_parquet(feature_summary_path, index=False)
pd.DataFrame(matching_summaries).to_parquet(matching_summary_path, index=False)
correlation_manifest.to_parquet(manifest_path, index=False)

correlation_manifest
