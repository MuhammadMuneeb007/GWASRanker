#!/usr/bin/env python3

import argparse
import contextlib
import io
import math
import subprocess
import sys
import tempfile
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

import Variation5 as v5


NORMAL = NormalDist()
CHI2_MEDIAN_DF1 = 0.454936423119572
DEFAULT_LDSC_REF = "/data/ascher01/uqmmune1/finalheritability/LDSCFILES/eur_ref_ld_chr/"
DEFAULT_BUILD = "auto"
DEFAULT_HM3_REFERENCE = "auto"
CHR_LENGTHS_HG19 = {
    "1": 249250621, "2": 243199373, "3": 198022430, "4": 191154276,
    "5": 180915260, "6": 171115067, "7": 159138663, "8": 146364022,
    "9": 141213431, "10": 135534747, "11": 135006516, "12": 133851895,
    "13": 115169878, "14": 107349540, "15": 102531392, "16": 90354753,
    "17": 81195210, "18": 78077248, "19": 59128983, "20": 63025520,
    "21": 48129895, "22": 51304566,
}
CHR_LENGTHS_HG38 = {
    "1": 248956422, "2": 242193529, "3": 198295559, "4": 190214555,
    "5": 181538259, "6": 170805979, "7": 159345973, "8": 145138636,
    "9": 138394717, "10": 133797422, "11": 135086622, "12": 133275309,
    "13": 114364328, "14": 107043718, "15": 101991189, "16": 90338345,
    "17": 83257441, "18": 80373285, "19": 58617616, "20": 64444167,
    "21": 46709983, "22": 50818468,
}


class TeeTextIO:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def clean_text(x):
    return v5.clean_text(x)


def percent(numerator, denominator):
    return v5.percent(numerator, denominator)


def numeric_column(df, column):
    return v5.get_numeric_series(df, column)


def text_column(df, column):
    return v5.get_text_series(df, column)


def finite_series(series):
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()


def resolve_reference_path(reference):
    reference = clean_text(reference)
    if not reference:
        return ""

    path = Path(reference)
    if path.exists():
        return str(path)

    completed = subprocess.run(
        ["gwaslab", "path", reference],
        check=False,
        capture_output=True,
        text=True,
    )

    for line in completed.stdout.splitlines():
        candidate = clean_text(line)
        if candidate and Path(candidate).exists():
            return candidate

    return reference


def read_reference_table(reference):
    path = resolve_reference_path(reference)

    if not path or not Path(path).exists():
        return pd.DataFrame(), path

    suffixes = "".join(Path(path).suffixes).lower()

    if path.endswith(".parquet"):
        return pd.read_parquet(path), path

    if ".vcf" in suffixes:
        return pd.DataFrame(), path

    compression = "gzip" if path.endswith(".gz") else None
    return pd.read_csv(path, sep=None, engine="python", compression=compression), path


def choose_column(columns, candidates):
    lower_to_original = {str(col).lower(): col for col in columns}

    for candidate in candidates:
        if candidate.lower() in lower_to_original:
            return lower_to_original[candidate.lower()]

    return ""


def choose_af_column(columns, preferred):
    if preferred:
        chosen = choose_column(columns, [preferred])
        if chosen:
            return chosen

    return choose_column(columns, [
        "EUR", "EUR_AF", "EAF_EUR", "AF_EUR",
        "EAF", "AF", "ALT_AF", "MAF", "FREQ",
    ])


def make_variant_key_frame(df, id_column="SNPID", af_column="EAF"):
    data = pd.DataFrame()

    if id_column in df.columns:
        data["SNPID"] = text_column(df, id_column)
    elif "rsID" in df.columns:
        data["SNPID"] = text_column(df, "rsID")
    else:
        data["SNPID"] = ""

    if af_column in df.columns:
        data["AF"] = numeric_column(df, af_column)
    else:
        data["AF"] = np.nan

    if {"CHR", "POS"}.issubset(df.columns):
        data["CHR"] = text_column(df, "CHR").str.replace("^chr", "", case=False, regex=True)
        data["POS"] = numeric_column(df, "POS")

    return data


def load_prior_build(output_feature_file):
    output_dir = Path(output_feature_file).parent

    for feature_name in ["Feature3.csv", "Feature5.csv"]:
        path = output_dir / feature_name
        if not path.exists():
            continue

        data = pd.read_csv(path)
        if data.empty or "genome_build" not in data.columns:
            continue

        build = clean_text(data.iloc[0].get("genome_build", ""))
        if build:
            return build, str(path)

    return "", ""


def normalise_build_for_gwaslab(build):
    text = clean_text(build).lower()

    if text in ["19", "hg19", "grch37", "build37"]:
        return "19"

    if text in ["38", "hg38", "grch38", "build38"]:
        return "38"

    return ""


def select_hm3_reference(build):
    build = normalise_build_for_gwaslab(build)

    if build == "38":
        return "1kg_hm3_hg38_eaf"

    return "1kg_hm3_hg19_eaf"


def normalise_chr_values(series):
    return (
        series.astype(str)
        .map(clean_text)
        .str.upper()
        .str.replace("^CHR", "", regex=True)
        .str.strip()
        .replace({"M": "MT"})
    )


def chromosome_lengths_for_build(build):
    return CHR_LENGTHS_HG38 if normalise_build_for_gwaslab(build) == "38" else CHR_LENGTHS_HG19


def p_to_z_abs(p):
    if pd.isna(p) or p <= 0 or p >= 1:
        return np.nan

    quantile = 1 - (p / 2)
    quantile = min(max(quantile, np.nextafter(0.0, 1.0)), np.nextafter(1.0, 0.0))
    return NORMAL.inv_cdf(quantile)


def p_series_to_z_abs(series):
    p = pd.to_numeric(series, errors="coerce")
    result = pd.Series(np.nan, index=p.index, dtype=float)
    valid = p.notna() & (p > 0) & (p < 1)

    if not valid.any():
        return result

    quantile = 1 - (p.loc[valid] / 2)
    quantile = quantile.clip(
        lower=np.nextafter(0.0, 1.0),
        upper=np.nextafter(1.0, 0.0),
    )
    result.loc[valid] = quantile.map(NORMAL.inv_cdf)
    return result


def calculate_maf(df):
    if "MAF" in df.columns:
        maf = numeric_column(df, "MAF")
        return maf.where((maf >= 0) & (maf <= 0.5))

    if "EAF" in df.columns:
        eaf = numeric_column(df, "EAF")
        eaf = eaf.where((eaf >= 0) & (eaf <= 1))
        return eaf.map(lambda x: min(x, 1 - x) if pd.notna(x) else np.nan)

    return pd.Series(dtype=float)


def calculate_sample_size_metrics(df):
    result = {
        "N_total_samples": "",
        "N_effective_case_control": "",
        "N_effective_case_control_available": False,
    }

    if "N" in df.columns:
        n = finite_series(numeric_column(df, "N"))
        if not n.empty:
            result["N_total_samples"] = int(round(float(n.median())))

    if {"N_CASES", "N_CONTROLS"}.issubset(df.columns):
        n_cases = finite_series(numeric_column(df, "N_CASES"))
        n_controls = finite_series(numeric_column(df, "N_CONTROLS"))

        if not n_cases.empty and not n_controls.empty:
            cases = float(n_cases.median())
            controls = float(n_controls.median())
            if cases > 0 and controls > 0:
                result["N_effective_case_control"] = round(4 / ((1 / cases) + (1 / controls)), 0)
                result["N_effective_case_control_available"] = True

    return result


def calculate_variant_quality_metrics(df):
    result = {
        "total_variants_M": len(df),
    }

    return result


def calculate_chi_square(df):
    # 1) Prefer existing Z if valid
    if "Z" in df.columns:
        z = finite_series(numeric_column(df, "Z"))
        if not z.empty:
            return z ** 2

    # 2) Try BETA / SE only if it produces valid values
    if {"BETA", "SE"}.issubset(df.columns):
        beta = numeric_column(df, "BETA")
        se = numeric_column(df, "SE").replace(0, np.nan)
        chi2 = finite_series((beta / se) ** 2)
        if not chi2.empty:
            return chi2

    # 3) Fallback to P-value
    if "P" in df.columns:
        p = numeric_column(df, "P")
        z_abs = p_series_to_z_abs(p)
        chi2 = finite_series(z_abs ** 2)
        if not chi2.empty:
            return chi2

    return pd.Series(dtype=float)


def calculate_statistical_quality_metrics(df):
    result = {
        "lambda_gc": "",
        "mean_chi2": "",
        "inconsistent_z_pct": "",
        "z_consistency_fail_gt_10pct": "",
    }

    chi2 = calculate_chi_square(df)
    if not chi2.empty:
        result["lambda_gc"] = round(float(chi2.median() / CHI2_MEDIAN_DF1), 6)
        result["mean_chi2"] = round(float(chi2.mean()), 6)

    if {"BETA", "SE", "P"}.issubset(df.columns):
        beta = numeric_column(df, "BETA")
        se = numeric_column(df, "SE").replace(0, np.nan)
        p = numeric_column(df, "P")

        z_expected = beta / se
        z_from_p = p_series_to_z_abs(p) * np.sign(beta)
        valid = pd.DataFrame({"z_expected": z_expected, "z_from_p": z_from_p}).replace(
            [np.inf, -np.inf], np.nan
        ).dropna()

        if not valid.empty:
            inconsistent_pct = percent(
                (valid["z_expected"] - valid["z_from_p"]).abs().gt(0.1).sum(),
                len(valid),
            )
            result["inconsistent_z_pct"] = inconsistent_pct
            result["z_consistency_fail_gt_10pct"] = bool(inconsistent_pct > 10)

    return result


def calculate_tail_lambda_metrics(df):
    result = {
        "lambda_gc_top_1pct": "",
        "lambda_top1_to_median_ratio": "",
    }

    chi2 = calculate_chi_square(df)
    if chi2.empty:
        return result

    chi2 = chi2.sort_values()
    n_tail = max(1, int(len(chi2) * 0.01))
    tail = chi2.tail(n_tail)

    median_lambda = chi2.median() / CHI2_MEDIAN_DF1
    tail_lambda = tail.median() / CHI2_MEDIAN_DF1

    result["lambda_gc_top_1pct"] = round(float(tail_lambda), 6)
    result["lambda_top1_to_median_ratio"] = round(float(tail_lambda / median_lambda), 6) if median_lambda else ""

    return result


def chi2_df1_ppf(probability):
    probability = min(max(float(probability), np.nextafter(0.0, 1.0)), np.nextafter(1.0, 0.0))
    normal_quantile = NORMAL.inv_cdf((probability + 1) / 2)
    return normal_quantile ** 2


def expected_chi2_tail_median(n_values, start_index):
    if n_values <= 0:
        return np.nan

    tail_length = n_values - start_index
    if tail_length <= 0:
        return np.nan

    median_position = start_index + ((tail_length - 1) / 2)
    probability = (median_position + 0.5) / n_values
    return chi2_df1_ppf(probability)


def calculate_qq_metrics(df):
    result = {
        "qq_lambda_median": "",
        "qq_lambda_top5pct": "",
        "qq_lambda_top1pct": "",
        "qq_top1_to_median_ratio": "",
        "qq_inflation_pattern": "",
    }

    chi2 = calculate_chi_square(df)
    if chi2.empty:
        result["qq_inflation_pattern"] = "missing_chi_square_inputs"
        return result

    chi2_sorted = np.sort(chi2.to_numpy(dtype=float))
    n_values = len(chi2_sorted)
    if n_values < 100:
        result["qq_inflation_pattern"] = "insufficient_variants"
        return result

    lambda_median = float(np.median(chi2_sorted) / CHI2_MEDIAN_DF1)

    top5_start = min(n_values - 1, int(n_values * 0.95))
    top1_start = min(n_values - 1, int(n_values * 0.99))

    observed_top5_median = float(np.median(chi2_sorted[top5_start:]))
    observed_top1_median = float(np.median(chi2_sorted[top1_start:]))
    expected_top5_median = expected_chi2_tail_median(n_values, top5_start)
    expected_top1_median = expected_chi2_tail_median(n_values, top1_start)

    lambda_top5 = observed_top5_median / expected_top5_median if expected_top5_median > 0 else np.nan
    lambda_top1 = observed_top1_median / expected_top1_median if expected_top1_median > 0 else np.nan

    result["qq_lambda_median"] = round(lambda_median, 6)
    result["qq_lambda_top5pct"] = round(float(lambda_top5), 6) if np.isfinite(lambda_top5) else ""
    result["qq_lambda_top1pct"] = round(float(lambda_top1), 6) if np.isfinite(lambda_top1) else ""
    result["qq_top1_to_median_ratio"] = round(float(lambda_top1 / lambda_median), 6) if lambda_median and np.isfinite(lambda_top1) else ""

    if np.isfinite(lambda_top1) and np.isfinite(lambda_top5):
        if lambda_top1 > lambda_top5 > (lambda_median * 1.5):
            pattern = "strong_signal_top_driven"
        elif lambda_median > 1.1 and lambda_top1 < (lambda_median * 2):
            pattern = "uniform_inflation_possible_confounding"
        elif lambda_median <= 1.05 and lambda_top1 > 5:
            pattern = "clean_null_with_real_signal"
        else:
            pattern = "mixed"
    else:
        pattern = "not_classified"

    result["qq_inflation_pattern"] = pattern
    return result


def calculate_beta_distribution_metrics(df):
    result = {
        "beta_mean": "",
        "beta_std": "",
        "beta_skewness": "",
        "beta_kurtosis": "",
        "beta_outlier_pct_abs_z_gt_5": "",
        "beta_outlier_pct_abs_z_gt_10": "",
        "beta_p95_abs": "",
        "beta_p99_abs": "",
        "beta_distribution_flag": "",
    }

    if "BETA" not in df.columns:
        result["beta_distribution_flag"] = "missing_beta"
        return result

    beta = finite_series(numeric_column(df, "BETA"))
    if len(beta) < 100:
        result["beta_distribution_flag"] = "insufficient_valid_beta"
        return result

    mean = float(beta.mean())
    std = float(beta.std())
    skewness = float(beta.skew())
    kurtosis = float(beta.kurtosis())

    result["beta_mean"] = round(mean, 8)
    result["beta_std"] = round(std, 8)
    result["beta_skewness"] = round(skewness, 8)
    result["beta_kurtosis"] = round(kurtosis, 8)
    result["beta_p95_abs"] = round(float(beta.abs().quantile(0.95)), 8)
    result["beta_p99_abs"] = round(float(beta.abs().quantile(0.99)), 8)

    outlier_gt_10_pct = ""
    if std > 0:
        z = (beta - mean) / std
        result["beta_outlier_pct_abs_z_gt_5"] = percent((z.abs() > 5).sum(), len(z))
        outlier_gt_10_pct = percent((z.abs() > 10).sum(), len(z))
        result["beta_outlier_pct_abs_z_gt_10"] = outlier_gt_10_pct

    if abs(skewness) > 2:
        flag = "highly_skewed_check_effect_scale"
    elif kurtosis > 50:
        flag = "extreme_kurtosis_possible_outliers"
    elif clean_text(outlier_gt_10_pct) and float(outlier_gt_10_pct) > 0.1:
        flag = "extreme_outliers_present"
    elif std > 1.0:
        flag = "large_beta_std_check_scale_log_or_vs_beta"
    else:
        flag = "normal"

    result["beta_distribution_flag"] = flag
    return result


def calculate_se_distribution_metrics(df):
    result = {
        "se_mean": "",
        "se_std": "",
        "se_cv": "",
        "se_min": "",
        "se_max": "",
        "se_zero_count": "",
        "se_negative_count": "",
        "se_bimodal_flag": False,
        "se_distribution_flag": "",
    }

    if "SE" not in df.columns:
        result["se_distribution_flag"] = "missing_se"
        return result

    raw_se = pd.to_numeric(df["SE"], errors="coerce")
    result["se_zero_count"] = int((raw_se == 0).sum())
    result["se_negative_count"] = int((raw_se < 0).sum())

    se = finite_series(raw_se)
    se = se[se > 0]
    if len(se) < 100:
        result["se_distribution_flag"] = "insufficient_valid_positive_se"
        return result

    mean = float(se.mean())
    std = float(se.std())
    cv = std / mean if mean > 0 else np.nan

    result["se_mean"] = round(mean, 8)
    result["se_std"] = round(std, 8)
    result["se_cv"] = round(float(cv), 6) if np.isfinite(cv) else ""
    result["se_min"] = round(float(se.min()), 8)
    result["se_max"] = round(float(se.max()), 8)

    q25 = float(se.quantile(0.25))
    q75 = float(se.quantile(0.75))
    result["se_bimodal_flag"] = bool(q25 > 0 and (q75 / q25) > 5)

    if np.isfinite(cv) and cv > 2:
        flag = "high_cv_suggests_mixed_cohorts_or_batch_effects"
    elif result["se_bimodal_flag"]:
        flag = "wide_se_iqr_check_for_batch_effects"
    elif result["se_zero_count"] > 0:
        flag = "zero_se_present_problematic_for_ldsc"
    elif result["se_negative_count"] > 0:
        flag = "negative_se_present_invalid"
    else:
        flag = "normal"

    result["se_distribution_flag"] = flag
    return result


def calculate_chromosome_coverage_metrics(df, build):
    lengths = chromosome_lengths_for_build(build)
    result = {
        "autosomes_present_count": "",
        "all_22_autosomes_present": "",
        "missing_autosomes": "",
        "variants_per_chr_signature": "",
        "min_autosomal_chr_variant_count": "",
        "chr22_variant_count": "",
        "variant_density_median_per_mb": "",
        "variant_density_min_chr_per_mb": "",
        "zero_coverage_1mb_bin_percentage": "",
    }

    if not {"CHR", "POS"}.issubset(df.columns):
        return result

    data = pd.DataFrame({
        "CHR": normalise_chr_values(df["CHR"]),
        "POS": pd.to_numeric(df["POS"], errors="coerce"),
    }).dropna()
    data = data[data["CHR"].isin(lengths.keys())]
    data = data[(data["POS"] > 0)]

    if data.empty:
        return result

    counts = data["CHR"].value_counts().to_dict()
    counts = {chrom: int(counts.get(chrom, 0)) for chrom in sorted(lengths.keys(), key=int)}
    present = [chrom for chrom, count in counts.items() if count > 0]
    missing = [chrom for chrom, count in counts.items() if count == 0]

    result["autosomes_present_count"] = len(present)
    result["all_22_autosomes_present"] = len(present) == 22
    result["missing_autosomes"] = " | ".join(missing)
    result["variants_per_chr_signature"] = " | ".join(f"{chrom}:{count}" for chrom, count in counts.items())
    result["min_autosomal_chr_variant_count"] = min(counts.values())
    result["chr22_variant_count"] = counts.get("22", 0)

    densities = []
    zero_bins = 0
    total_bins = 0

    for chrom, length in lengths.items():
        chrom_data = data[data["CHR"] == chrom]
        densities.append(counts[chrom] / (length / 1_000_000))

        n_bins = math.ceil(length / 1_000_000)
        total_bins += n_bins
        if chrom_data.empty:
            zero_bins += n_bins
            continue

        bins_present = ((chrom_data["POS"].astype(int) - 1) // 1_000_000).clip(lower=0, upper=n_bins - 1)
        zero_bins += n_bins - bins_present.nunique()

    result["variant_density_median_per_mb"] = round(float(pd.Series(densities).median()), 4)
    result["variant_density_min_chr_per_mb"] = round(float(min(densities)), 4)
    result["zero_coverage_1mb_bin_percentage"] = percent(zero_bins, total_bins)

    return result


def calculate_effect_frequency_metrics(df):
    result = {
        "abs_beta_log_maf_spearman": "",
        "effect_frequency_relationship_note": "",
    }

    if "BETA" not in df.columns:
        return result

    maf = calculate_maf(df)
    beta = pd.to_numeric(df["BETA"], errors="coerce")
    valid = pd.DataFrame({"maf": maf, "abs_beta": beta.abs()}).replace([np.inf, -np.inf], np.nan).dropna()
    valid = valid[(valid["maf"] > 0) & (valid["maf"] <= 0.5)]

    if len(valid) < 100:
        result["effect_frequency_relationship_note"] = "insufficient_valid_beta_maf"
        return result

    valid["log_maf"] = np.log10(valid["maf"])
    result["abs_beta_log_maf_spearman"] = round(float(valid["abs_beta"].corr(valid["log_maf"], method="spearman")), 6)
    result["effect_frequency_relationship_note"] = "descriptive_qc_not_hard_failure"

    return result


def calculate_info_by_maf_metrics(df):
    result = {
        "info_mean_maf_0_01_0_05": "",
        "info_mean_maf_0_05_0_5": "",
        "info_low_0_8_maf_0_01_0_05_pct": "",
        "info_low_0_8_maf_0_05_0_5_pct": "",
    }

    if "INFO" not in df.columns:
        return result

    maf = calculate_maf(df)
    info = pd.to_numeric(df["INFO"], errors="coerce")
    valid = pd.DataFrame({"maf": maf, "info": info}).replace([np.inf, -np.inf], np.nan).dropna()
    valid = valid[(valid["maf"] >= 0) & (valid["maf"] <= 0.5) & (valid["info"] >= 0) & (valid["info"] <= 1)]

    if valid.empty:
        return result

    low_freq = valid[(valid["maf"] >= 0.01) & (valid["maf"] < 0.05)]
    common = valid[(valid["maf"] >= 0.05) & (valid["maf"] <= 0.5)]

    if not low_freq.empty:
        result["info_mean_maf_0_01_0_05"] = round(float(low_freq["info"].mean()), 6)
        result["info_low_0_8_maf_0_01_0_05_pct"] = percent((low_freq["info"] < 0.8).sum(), len(low_freq))

    if not common.empty:
        result["info_mean_maf_0_05_0_5"] = round(float(common["info"].mean()), 6)
        result["info_low_0_8_maf_0_05_0_5_pct"] = percent((common["info"] < 0.8).sum(), len(common))

    return result


def count_window_lead_loci(df, p_threshold, window_kb):
    if not {"CHR", "POS", "P"}.issubset(df.columns):
        return "", "missing_CHR_POS_or_P"

    hits = pd.DataFrame({
        "CHR": text_column(df, "CHR").str.replace("^chr", "", case=False, regex=True).str.upper(),
        "POS": numeric_column(df, "POS"),
        "P": numeric_column(df, "P"),
    }).dropna()

    hits = hits[(hits["P"] > 0) & (hits["P"] <= p_threshold)]
    if hits.empty:
        return 0, "window_greedy_no_ld"

    hits = hits.sort_values("P")
    selected = []
    window_bp = int(window_kb * 1000)

    for _, row in hits.iterrows():
        same_locus = False
        for lead in selected:
            if row["CHR"] == lead["CHR"] and abs(row["POS"] - lead["POS"]) <= window_bp:
                same_locus = True
                break
        if not same_locus:
            selected.append(row)

    return len(selected), "window_greedy_no_ld"


def calculate_reference_af_metrics(df, ref_af_path, reference_af_column=""):
    result = {
        "reference_af_file": clean_text(ref_af_path),
        "reference_af_resolved_path": "",
        "reference_af_id_column": "",
        "reference_af_column": "",
        "reference_overlap_pct": "",
        "reference_overlap_count": "",
        "reference_af_correlation": "",
        "reference_af_correlation_below_0_95": "",
        "reference_mean_abs_af_difference": "",
    }

    if not ref_af_path:
        return result

    ref, resolved_path = read_reference_table(ref_af_path)
    result["reference_af_resolved_path"] = resolved_path

    if ref.empty:
        return result

    id_col = choose_column(ref.columns, ["SNPID", "SNP", "rsID", "RSID", "variant_id", "ID"])
    af_col = choose_af_column(ref.columns, reference_af_column)

    if not id_col or not af_col or "SNPID" not in df.columns or "EAF" not in df.columns:
        return result

    result["reference_af_id_column"] = id_col
    result["reference_af_column"] = af_col

    left = pd.DataFrame({
        "SNPID": text_column(df, "SNPID"),
        "gwas_af": numeric_column(df, "EAF"),
    }).dropna()
    right = pd.DataFrame({
        "SNPID": ref[id_col].astype(str).map(clean_text),
        "ref_af": pd.to_numeric(ref[af_col], errors="coerce"),
    }).dropna()

    merged = left.merge(right, on="SNPID", how="inner")
    result["reference_overlap_count"] = len(merged)
    result["reference_overlap_pct"] = percent(len(merged), len(left))

    if len(merged) >= 3:
        correlation = round(float(merged["gwas_af"].corr(merged["ref_af"])), 6)
        result["reference_af_correlation"] = correlation
        result["reference_af_correlation_below_0_95"] = bool(correlation < 0.95)
        result["reference_mean_abs_af_difference"] = round(float((merged["gwas_af"] - merged["ref_af"]).abs().mean()), 6)

    return result


def calculate_hapmap3_overlap_metrics(df, hm3_reference):
    result = {
        "hapmap3_reference": clean_text(hm3_reference),
        "hapmap3_reference_resolved_path": "",
        "hapmap3_reference_id_column": "",
        "hapmap3_overlap_count": "",
        "hapmap3_overlap_percentage_of_gwas": "",
        "hapmap3_overlap_percentage_of_reference": "",
        "hapmap3_match_method": "",
        "hapmap3_rsid_available_in_gwas": False,
        "hapmap3_rsid_available_in_reference": False,
        "hapmap3_chrpos_fallback_used": False,
    }

    if not hm3_reference:
        return result

    ref, resolved_path = read_reference_table(hm3_reference)
    result["hapmap3_reference_resolved_path"] = resolved_path

    if ref.empty:
        result["hapmap3_match_method"] = "failed_reference_empty"
        return result

    ref_id_col = choose_column(ref.columns, ["SNPID", "SNP", "rsID", "RSID", "variant_id", "ID"])

    gwas_rsids = pd.Series(dtype=str)
    for col_name in ["SNPID", "rsID", "SNP", "SNPID_raw"]:
        if col_name in df.columns:
            candidate = df[col_name].astype(str).map(clean_text)
            valid = candidate[candidate.str.match(r"^rs\d+$", case=False, na=False)]
            if len(valid) > len(gwas_rsids):
                gwas_rsids = valid
    gwas_rsids = gwas_rsids.drop_duplicates()

    ref_rsids = pd.Series(dtype=str)
    if ref_id_col:
        candidate = ref[ref_id_col].astype(str).map(clean_text)
        ref_rsids = candidate[candidate.str.match(r"^rs\d+$", case=False, na=False)].drop_duplicates()

    result["hapmap3_rsid_available_in_gwas"] = len(gwas_rsids) > 100
    result["hapmap3_rsid_available_in_reference"] = len(ref_rsids) > 100

    if len(gwas_rsids) > 100 and len(ref_rsids) > 100:
        ref_rsid_set = set(ref_rsids.str.lower())
        overlap = gwas_rsids[gwas_rsids.str.lower().isin(ref_rsid_set)]

        result["hapmap3_overlap_count"] = int(len(overlap))
        result["hapmap3_overlap_percentage_of_gwas"] = percent(len(overlap), len(gwas_rsids))
        result["hapmap3_overlap_percentage_of_reference"] = percent(len(overlap), len(ref_rsids))
        result["hapmap3_match_method"] = "rsid"
        result["hapmap3_reference_id_column"] = ref_id_col or ""
        return result

    result["hapmap3_chrpos_fallback_used"] = True

    gwas_chr_col = choose_column(df.columns, ["CHR", "CHROM", "chromosome", "#CHROM"])
    gwas_pos_col = choose_column(df.columns, ["POS", "BP", "base_pair_location", "position", "POS_b37"])
    ref_chr_col = choose_column(ref.columns, ["CHR", "CHROM", "chromosome", "#CHROM"])
    ref_pos_col = choose_column(ref.columns, ["POS", "BP", "base_pair_location", "position", "POS_b37"])

    if not gwas_chr_col or not gwas_pos_col:
        result["hapmap3_match_method"] = "failed_no_chrpos_in_gwas"
        return result

    if not ref_chr_col or not ref_pos_col:
        result["hapmap3_match_method"] = "failed_no_chrpos_in_reference"
        return result

    def normalise_chr(series):
        return (
            series.astype(str)
            .map(clean_text)
            .str.upper()
            .str.replace("^CHR", "", regex=True)
            .str.strip()
            .replace({"M": "MT"})
        )

    valid_chromosomes = [str(i) for i in range(1, 23)] + ["X", "Y", "MT"]

    gwas_keys = pd.DataFrame({
        "CHR": normalise_chr(df[gwas_chr_col]),
        "POS": pd.to_numeric(df[gwas_pos_col], errors="coerce"),
    }).dropna()
    gwas_keys = gwas_keys[gwas_keys["CHR"].isin(valid_chromosomes)]
    gwas_keys["key"] = gwas_keys["CHR"] + ":" + gwas_keys["POS"].astype(int).astype(str)
    gwas_key_set = set(gwas_keys["key"].drop_duplicates())

    ref_keys = pd.DataFrame({
        "CHR": normalise_chr(ref[ref_chr_col]),
        "POS": pd.to_numeric(ref[ref_pos_col], errors="coerce"),
    }).dropna()
    ref_keys = ref_keys[ref_keys["CHR"].isin(valid_chromosomes)]
    ref_keys["key"] = ref_keys["CHR"] + ":" + ref_keys["POS"].astype(int).astype(str)
    ref_key_set = set(ref_keys["key"].drop_duplicates())

    if not gwas_key_set or not ref_key_set:
        result["hapmap3_match_method"] = "failed_chrpos_keys_empty"
        return result

    overlap_count = len(gwas_key_set & ref_key_set)
    result["hapmap3_overlap_count"] = int(overlap_count)
    result["hapmap3_overlap_percentage_of_gwas"] = percent(overlap_count, len(gwas_key_set))
    result["hapmap3_overlap_percentage_of_reference"] = percent(overlap_count, len(ref_key_set))
    result["hapmap3_match_method"] = "chrpos"
    result["hapmap3_reference_id_column"] = f"{ref_chr_col}:{ref_pos_col}"

    return result


def h2_obs_to_liability(h2_obs, sample_prevalence, population_prevalence):
    import gwaslab as gl

    result = gl.h2_obs_to_liab(
        h2_obs,
        P=sample_prevalence,
        K=population_prevalence,
    )

    if isinstance(result, tuple):
        return result[0]

    return result


def calculate_n_times_h2(row):
    result = {
        "n_times_h2": "",
        "n_times_h2_available": False,
        "n_times_h2_source": "",
    }

    h2 = clean_text(row.get("ldsc_h2", ""))
    if h2 == "":
        return result

    n_value = row.get("N_effective_case_control", "") or row.get("N_total_samples", "")
    if clean_text(n_value) == "":
        return result

    result["n_times_h2"] = round(float(n_value) * float(h2), 6)
    result["n_times_h2_available"] = True
    result["n_times_h2_source"] = "N_effective_case_control_x_ldsc_h2" if clean_text(row.get("N_effective_case_control", "")) else "N_total_samples_x_ldsc_h2"

    return result


def extract_ldsc_h2_fields(ldsc_h2):
    result = {
        "ldsc_h2": "",
        "ldsc_h2_se": "",
        "ldsc_lambda_gc": "",
        "ldsc_mean_chi2": "",
        "ldsc_intercept": "",
        "ldsc_intercept_se": "",
        "ldsc_ratio": "",
        "ldsc_ratio_se": "",
    }

    if ldsc_h2 is None:
        return result

    if isinstance(ldsc_h2, pd.DataFrame):
        if ldsc_h2.empty:
            return result
        row = ldsc_h2.iloc[0]
    else:
        row = pd.Series(ldsc_h2)

    mapping = {
        "ldsc_h2": ["h2_obs", "h2", "Heritability"],
        "ldsc_h2_se": ["h2_se", "h2_obs_se", "Heritability_se"],
        "ldsc_lambda_gc": ["Lambda_gc", "lambda_gc"],
        "ldsc_mean_chi2": ["Mean_chi2", "mean_chi2"],
        "ldsc_intercept": ["Intercept", "intercept"],
        "ldsc_intercept_se": ["Intercept_se", "intercept_se"],
        "ldsc_ratio": ["Ratio", "ratio"],
        "ldsc_ratio_se": ["Ratio_se", "ratio_se"],
    }

    for output_col, candidates in mapping.items():
        for candidate in candidates:
            if candidate in row.index and pd.notna(row[candidate]):
                result[output_col] = row[candidate]
                break

    return result


def run_gwaslab_ldsc(sumstats, ref_ld_chr, w_ld_chr, log_file):
    log_buffer = io.StringIO()
    tee_stdout = TeeTextIO(sys.stdout, log_buffer)
    tee_stderr = TeeTextIO(sys.stderr, log_buffer)

    with contextlib.redirect_stdout(tee_stdout), contextlib.redirect_stderr(tee_stderr):
        sumstats.estimate_h2_by_ldsc(ref_ld_chr=ref_ld_chr, w_ld_chr=w_ld_chr)

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(log_buffer.getvalue())

    return extract_ldsc_h2_fields(getattr(sumstats, "ldsc_h2", None))


def materialize_tabular_input_for_gwaslab(path, tmpdir):
    path = Path(path)

    if path.suffix.lower() != ".csv":
        return path, "direct_file"

    df = pd.read_csv(path, low_memory=False)
    df = df.loc[:, ~df.columns.duplicated()].copy()

    # GWASLab auto mode can map both rsID and SNPID to SNPID, which creates
    # duplicate renamed columns and crashes later in datatype conversion.
    if "SNPID" in df.columns and "rsID" in df.columns:
        df = df.drop(columns=["rsID"])
    elif "SNPID" not in df.columns and "rsID" in df.columns:
        df = df.rename(columns={"rsID": "SNPID"})

    rewritten_path = Path(tmpdir) / f"{path.stem}.tsv"
    df.to_csv(rewritten_path, sep="\t", index=False)
    return rewritten_path, "csv_rewritten_to_tsv"


def load_sumstats_for_variation6(file_path, output_dir, build):
    import gwaslab as gl

    compression = v5.detect_compression(file_path)
    log_file = output_dir / "Feature6_gwaslab.log"

    tmpdir = None
    logical_path = file_path
    load_source = "direct_file"

    if compression == "zip":
        tmpdir = tempfile.TemporaryDirectory(prefix="variation6_gwaslab_")
        logical_path, load_source = v5.extract_zip_member(file_path, tmpdir.name)
    elif compression in ["tar", "tar.gz"]:
        tmpdir = tempfile.TemporaryDirectory(prefix="variation6_gwaslab_")
        logical_path, load_source = v5.extract_tar_member(file_path, tmpdir.name)
    elif Path(file_path).suffix.lower() == ".csv":
        tmpdir = tempfile.TemporaryDirectory(prefix="variation6_gwaslab_")
        logical_path, load_source = materialize_tabular_input_for_gwaslab(file_path, tmpdir.name)

    log_buffer = io.StringIO()
    tee_stdout = TeeTextIO(sys.stdout, log_buffer)
    tee_stderr = TeeTextIO(sys.stderr, log_buffer)

    kwargs = {
        "fmt": "auto",
        "build": build,
        "verbose": True,
    }

    with contextlib.redirect_stdout(tee_stdout), contextlib.redirect_stderr(tee_stderr):
        sumstats = gl.Sumstats(str(logical_path), **kwargs)

    log_file.write_text(log_buffer.getvalue(), encoding="utf-8")
    return sumstats, load_source, tmpdir, kwargs


def prepare_sumstats_for_ldsc(sumstats):
    df = v5.get_gwaslab_data(sumstats)

    result = {
        "ldsc_snpid_source": "SNPID",
        "ldsc_beta_source": "BETA",
        "ldsc_z_source": "existing_Z",
        "ldsc_se_source": "existing_SE",
    }

    # Prefer true rsID if available
    if "rsID" in df.columns:
        rsid = df["rsID"].astype(str).map(clean_text)
        valid_rsid = rsid.str.match(r"^rs\d+$", case=False, na=False)
        if valid_rsid.sum() > 0:
            df.loc[valid_rsid, "SNPID"] = rsid[valid_rsid]
            result["ldsc_snpid_source"] = "rsID_copied_to_SNPID"

    # If SNPID itself is not rsID, LDSC will fail unless you map CHR:POS to rsID
    if "SNPID" in df.columns:
        snpid = df["SNPID"].astype(str).map(clean_text)
        result["ldsc_valid_rsid_count_before_cleaning"] = int(
            snpid.str.match(r"^rs\d+$", case=False, na=False).sum()
        )

    # Create BETA from OR if BETA missing
    beta_missing_or_empty = (
        "BETA" not in df.columns
        or pd.to_numeric(df["BETA"], errors="coerce").notna().sum() == 0
    )

    if beta_missing_or_empty and "OR" in df.columns:
        or_values = pd.to_numeric(df["OR"], errors="coerce")
        df["BETA"] = np.where(or_values > 0, np.log(or_values), np.nan)
        result["ldsc_beta_source"] = "log_OR"

    # Create signed Z from BETA and P if Z is missing or empty
    z_missing_or_empty = (
        "Z" not in df.columns
        or pd.to_numeric(df["Z"], errors="coerce").notna().sum() == 0
    )

    if z_missing_or_empty and {"BETA", "P"}.issubset(df.columns):
        beta = pd.to_numeric(df["BETA"], errors="coerce")
        p = pd.to_numeric(df["P"], errors="coerce")
        z_abs = p_series_to_z_abs(p)
        df["Z"] = z_abs * np.sign(beta)
        result["ldsc_z_source"] = "signed_Z_from_P_and_BETA"

    # If SE is missing/invalid, derive SE = abs(BETA / Z)
    se_missing_or_empty = (
        "SE" not in df.columns
        or (pd.to_numeric(df["SE"], errors="coerce") > 0).sum() == 0
    )

    if se_missing_or_empty and {"BETA", "Z"}.issubset(df.columns):
        beta = pd.to_numeric(df["BETA"], errors="coerce")
        z = pd.to_numeric(df["Z"], errors="coerce").replace(0, np.nan)
        df["SE"] = (beta / z).abs()
        result["ldsc_se_source"] = "derived_abs_BETA_div_Z"

    return result


def clean_sumstats_for_ldsc(sumstats):
    df = v5.get_gwaslab_data(sumstats)
    pre_count = len(df)

    result = {
        "ldsc_clean_pre_variant_count": pre_count,
        "ldsc_clean_post_variant_count": 0,
        "ldsc_clean_removed_variant_count": pre_count,
        "ldsc_clean_removed_variant_percentage": 100.0 if pre_count else 0.0,
    }

    if df.empty:
        sumstats.data = df
        return result

    mask = pd.Series(True, index=df.index)

    if "SNPID" in df.columns:
        snpid = df["SNPID"].astype(str).map(clean_text)
        mask &= snpid.str.match(r"^rs\d+$", case=False, na=False)
    else:
        mask &= False

    if "CHR" in df.columns:
        chr_text = df["CHR"].astype(str).map(clean_text).str.replace("^chr", "", case=False, regex=True)
        mask &= chr_text.isin([str(i) for i in range(1, 23)])
    else:
        mask &= False

    if "POS" in df.columns:
        pos = pd.to_numeric(df["POS"], errors="coerce")
        mask &= pos.notna() & (pos > 0)
    else:
        mask &= False

    if {"EA", "NEA"}.issubset(df.columns):
        ea = df["EA"].astype(str).map(clean_text).str.upper()
        nea = df["NEA"].astype(str).map(clean_text).str.upper()
        mask &= ea.str.match(r"^[ACGT]$", na=False)
        mask &= nea.str.match(r"^[ACGT]$", na=False)
        mask &= ea.ne(nea)
    else:
        mask &= False

    if "BETA" in df.columns:
        beta = pd.to_numeric(df["BETA"], errors="coerce")
        mask &= beta.notna() & np.isfinite(beta)
    else:
        mask &= False

    if "SE" in df.columns:
        se = pd.to_numeric(df["SE"], errors="coerce")
        mask &= se.notna() & np.isfinite(se) & (se > 0)
    else:
        mask &= False

    if "P" in df.columns:
        p = pd.to_numeric(df["P"], errors="coerce")
        mask &= p.notna() & np.isfinite(p) & (p > 0) & (p < 1)

    if "Z" in df.columns:
        z = pd.to_numeric(df["Z"], errors="coerce")
        mask &= z.notna() & np.isfinite(z)

    cleaned = df.loc[mask].copy()

    if "SNPID" in cleaned.columns:
        cleaned = cleaned.drop_duplicates(subset=["SNPID"], keep="first")

    if {"CHR", "POS", "EA", "NEA"}.issubset(cleaned.columns):
        cleaned = cleaned.drop_duplicates(subset=["CHR", "POS", "EA", "NEA"], keep="first")

    sumstats.data = cleaned.reset_index(drop=True)

    post_count = len(sumstats.data)
    result["ldsc_clean_post_variant_count"] = post_count
    result["ldsc_clean_removed_variant_count"] = pre_count - post_count
    result["ldsc_clean_removed_variant_percentage"] = percent(pre_count - post_count, pre_count)

    return result


def calculate_all_metrics(df, args, reference_af_file, hm3_reference, build):
    row = {}
    row.update(calculate_sample_size_metrics(df))
    row.update(calculate_variant_quality_metrics(df))
    row.update(calculate_statistical_quality_metrics(df))
    row.update(calculate_tail_lambda_metrics(df))
    row.update(calculate_qq_metrics(df))
    row.update(calculate_beta_distribution_metrics(df))
    row.update(calculate_se_distribution_metrics(df))
    row.update(calculate_chromosome_coverage_metrics(df, build))
    row.update(calculate_effect_frequency_metrics(df))
    row.update(calculate_info_by_maf_metrics(df))

    n_signals, signal_method = count_window_lead_loci(
        df,
        p_threshold=args.p_threshold,
        window_kb=args.signal_window_kb,
    )
    row["n_genome_wide_significant_window_loci"] = n_signals
    row["n_signals_method"] = signal_method

    row.update(calculate_reference_af_metrics(df, reference_af_file, args.reference_af_column))
    row.update(calculate_hapmap3_overlap_metrics(df, hm3_reference))
    return row


def profile_variation6(job_index, phenotype, accession, file_path, final_gwas_file, output_feature_file, args):
    file_path = Path(file_path)
    final_gwas_file = Path(final_gwas_file)
    output_feature_file = Path(output_feature_file)
    output_dir = output_feature_file.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    detected_build, build_source = load_prior_build(output_feature_file)
    requested_build = clean_text(args.build)
    gwaslab_build = normalise_build_for_gwaslab(requested_build)

    if requested_build.lower() == "auto":
        gwaslab_build = normalise_build_for_gwaslab(detected_build)

    if not gwaslab_build:
        gwaslab_build = "19"

    hm3_reference = args.hm3_reference
    if clean_text(hm3_reference).lower() == "auto":
        hm3_reference = select_hm3_reference(gwaslab_build)

    reference_af_file = args.reference_af_file
    if clean_text(reference_af_file).lower() == "auto":
        reference_af_file = hm3_reference

    log_file = output_dir / "Feature6_gwaslab.log"
    sumstats, load_source, tmpdir, gwaslab_kwargs = load_sumstats_for_variation6(
        file_path=final_gwas_file,
        output_dir=output_dir,
        build=gwaslab_build,
    )

    try:
        ldsc_preparation = prepare_sumstats_for_ldsc(sumstats)
        df = v5.get_gwaslab_data(sumstats)
        metrics = calculate_all_metrics(df, args, reference_af_file, hm3_reference, gwaslab_build)
        ldsc_cleaning = clean_sumstats_for_ldsc(sumstats)

        ldsc_requested = bool(args.ref_ld_chr and args.w_ld_chr and not args.skip_ldsc)
        ldsc_result = {
            "ldsc_requested": ldsc_requested,
            "ldsc_h2_attempted": False,
            "ldsc_h2_success": False,
            "ldsc_ref_ld_chr": clean_text(args.ref_ld_chr),
            "ldsc_w_ld_chr": clean_text(args.w_ld_chr),
            "ldsc_h2": "",
            "ldsc_h2_se": "",
            "ldsc_lambda_gc": "",
            "ldsc_mean_chi2": "",
            "ldsc_intercept": "",
            "ldsc_intercept_se": "",
            "ldsc_ratio": "",
            "ldsc_ratio_se": "",
            "liability_h2": "",
            "liability_h2_available": False,
            "liability_prevalence_source": "",
            "ldsc_skip_reason": "",
        }

        if ldsc_requested:
            if ldsc_cleaning["ldsc_clean_post_variant_count"] == 0:
                ldsc_result["ldsc_skip_reason"] = "no_variants_after_ldsc_cleaning"
            else:
                ldsc_result["ldsc_h2_attempted"] = True
                ldsc_result.update(run_gwaslab_ldsc(sumstats, args.ref_ld_chr, args.w_ld_chr, log_file))
                ldsc_result["ldsc_h2_success"] = clean_text(ldsc_result["ldsc_h2"]) != ""

            if (
                clean_text(ldsc_result["ldsc_h2"]) != ""
                and args.sample_prevalence is not None
                and args.population_prevalence is not None
            ):
                ldsc_result["liability_h2"] = h2_obs_to_liability(
                    float(ldsc_result["ldsc_h2"]),
                    args.sample_prevalence,
                    args.population_prevalence,
                )
                ldsc_result["liability_h2_available"] = True
                ldsc_result["liability_prevalence_source"] = "cli_sample_and_population_prevalence"

        row = {
            "job_index": job_index,
            "phenotype": phenotype,
            "accessionId": accession,
            "file_path": str(file_path),
            "file_name": file_path.name,
            "final_gwas_file": str(final_gwas_file),
            "final_gwas_name": final_gwas_file.name,
            "gwaslab_load_source": load_source,
            "gwaslab_log_file": str(log_file),
            "gwaslab_kwargs_used": " | ".join(f"{key}={value}" for key, value in gwaslab_kwargs.items()),
            "detected_build_from_prior_feature": detected_build,
            "detected_build_source": build_source,
            "gwaslab_build_used": gwaslab_build,
            "gwaslab_loaded_variant_count": len(df),
            "hm3_reference_used": hm3_reference,
            "reference_af_file_used": reference_af_file,
            "heritability_attempted": False,
            "heritability_success": False,
            "heritability_missing_reason": "" if ldsc_requested else "missing_ref_ld_chr_or_w_ld_chr_or_skip_ldsc",
        }

        row.update(metrics)
        row.update(ldsc_preparation)
        row.update(ldsc_cleaning)
        row.update(ldsc_result)
        row["heritability_attempted"] = row["ldsc_h2_attempted"]
        row["heritability_success"] = row["ldsc_h2_success"]
        row.update(calculate_n_times_h2(row))

        pd.DataFrame([row]).to_csv(output_feature_file, index=False)
        return row

    finally:
        if tmpdir is not None:
            tmpdir.cleanup()


def run_one_job(index, jobs_file, force, args):
    jobs_path = Path(jobs_file)

    if not jobs_path.exists():
        raise FileNotFoundError(f"Missing jobs file: {jobs_path}")

    jobs = pd.read_csv(jobs_path)
    required = ["job_index", "phenotype", "accessionId", "file_path", "output_feature_file"]
    missing = [c for c in required if c not in jobs.columns]

    if missing:
        raise ValueError(f"Missing columns in {jobs_path}: {missing}")

    sub = jobs[jobs["job_index"].astype(int) == int(index)]
    if sub.empty:
        raise ValueError(f"No job found for index: {index}")

    job = sub.iloc[0]
    phenotype = clean_text(job["phenotype"])
    accession = clean_text(job["accessionId"])
    file_path = Path(clean_text(job["file_path"]))
    base_output = Path(clean_text(job["output_feature_file"]))
    final_gwas_file = base_output.parent / "FinalGWAS.csv"
    output_feature_file = base_output.parent / "Feature6.csv"

    print("=" * 100)
    print("VARIATION 6: GWAS QUALITY METRICS AND OPTIONAL GWASLAB LDSC HERITABILITY")
    print("=" * 100)
    print(f"[JOB INDEX]  {index}")
    print(f"[PHENOTYPE]  {phenotype}")
    print(f"[ACCESSION]  {accession}")
    print(f"[GWAS FILE]  {file_path}")
    print(f"[FINAL GWAS] {final_gwas_file}")
    print(f"[OUTPUT]     {output_feature_file}")
    print("=" * 100)

    if not file_path.exists():
        raise FileNotFoundError(f"Missing GWAS file: {file_path}")
    if not final_gwas_file.exists():
        raise FileNotFoundError(f"Missing FinalGWAS.csv: {final_gwas_file}")

    if output_feature_file.exists() and not force:
        print(f"[SKIP] Feature6.csv already exists: {output_feature_file}")
        return

    row = profile_variation6(
        job_index=index,
        phenotype=phenotype,
        accession=accession,
        file_path=file_path,
        final_gwas_file=final_gwas_file,
        output_feature_file=output_feature_file,
        args=args,
    )

    print()
    print("[SAVED]")
    print(output_feature_file)
    print()
    print("[KEY RESULTS]")
    print(f"lambda_gc:                    {row['lambda_gc']}")
    print(f"mean_chi2:                    {row['mean_chi2']}")
    print(f"lambda_gc_top_1pct:           {row['lambda_gc_top_1pct']}")
    print(f"qq_inflation_pattern:         {row['qq_inflation_pattern']}")
    print(f"beta_distribution_flag:       {row['beta_distribution_flag']}")
    print(f"se_distribution_flag:         {row['se_distribution_flag']}")
    print(f"N_effective_case_control:     {row['N_effective_case_control']}")
    print(f"n_times_h2:                   {row['n_times_h2']}")
    print(f"autosomes_present_count:      {row['autosomes_present_count']}")
    print(f"HapMap3 overlap:              {row['hapmap3_overlap_count']}")
    print(f"reference AF correlation:     {row['reference_af_correlation']}")
    print(f"LDSC cleaned variants:        {row['ldsc_clean_post_variant_count']}")
    print(f"gwaslab_build_used:           {row['gwaslab_build_used']}")
    print(f"ldsc_h2_attempted:            {row['ldsc_h2_attempted']}")
    print(f"ldsc_h2_success:              {row['ldsc_h2_success']}")
    print(f"ldsc_skip_reason:             {row['ldsc_skip_reason']}")
    print(f"ldsc_h2:                      {row['ldsc_h2']}")
    print(f"ldsc_intercept:               {row['ldsc_intercept']}")
    print(f"n_window_loci:                {row['n_genome_wide_significant_window_loci']}")
    print("=" * 100)


def combine_feature6(jobs_file, output_file):
    jobs = pd.read_csv(jobs_file)
    rows = []

    for _, job in jobs.iterrows():
        base_output = Path(clean_text(job["output_feature_file"]))
        feature6 = base_output.parent / "Feature6.csv"
        if feature6.exists():
            rows.append(pd.read_csv(feature6))

    out_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    out_df.to_csv(output_file, index=False)

    print("=" * 100)
    print("[COMBINED Feature6.csv FILES]")
    print(f"[FILES COMBINED] {len(rows)}")
    print(f"[OUTPUT] {output_file}")
    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(
        description="Variation6: GWAS quality metrics and optional GWASLab LDSC heritability."
    )
    parser.add_argument("index", nargs="?", type=int, help="GWAS job index from GWAS_jobs.csv")
    parser.add_argument("--jobs-file", default="GWAS_jobs.csv", help="GWAS job manifest")
    parser.add_argument("--force", action="store_true", help="Overwrite Feature6.csv")
    parser.add_argument("--combine", action="store_true", help="Combine all per-GWAS Feature6.csv files")
    parser.add_argument("--combine-output", default="Variation6_allGWAS.csv")
    parser.add_argument(
        "--build",
        default=DEFAULT_BUILD,
        help="Genome build passed to GWASLab: auto, 19, or 38. Default: auto",
    )
    parser.add_argument("--ref-ld-chr", default=DEFAULT_LDSC_REF, help="LDSC ref_ld_chr folder/prefix")
    parser.add_argument("--w-ld-chr", default=DEFAULT_LDSC_REF, help="LDSC w_ld_chr folder/prefix")
    parser.add_argument("--skip-ldsc", action="store_true", help="Skip GWASLab LDSC heritability even though default LDSC paths are set")
    parser.add_argument("--sample-prevalence", type=float, default=None)
    parser.add_argument("--population-prevalence", type=float, default=None)
    parser.add_argument(
        "--reference-af-file",
        default=DEFAULT_HM3_REFERENCE,
        help="Reference AF table/path/GWASLab keyword. Default: auto build-matched HapMap3 EAF",
    )
    parser.add_argument("--reference-af-column", default="", help="Reference AF column to use, e.g. EUR or EAF")
    parser.add_argument(
        "--hm3-reference",
        default=DEFAULT_HM3_REFERENCE,
        help="HapMap3 reference table/path/GWASLab keyword. Default: auto build-matched HapMap3 EAF",
    )
    parser.add_argument("--p-threshold", type=float, default=5e-8)
    parser.add_argument("--signal-window-kb", type=int, default=500)

    args = parser.parse_args()

    if args.combine:
        combine_feature6(args.jobs_file, args.combine_output)
        return

    if args.index is None:
        parser.error("index is required unless --combine is used")

    run_one_job(args.index, args.jobs_file, args.force, args)


if __name__ == "__main__":
    main()
