#!/usr/bin/env python3

"""
PRS2-Calculation-Merge.py

Merge PRS outputs across all GWAS jobs for both:
    - PRS_Plink
    - PRS_PRSice-2

This script writes rows even when files are missing so you can see exactly
which jobs/tools did not produce outputs.

Outputs:
    PRS2_All_GWAS_Run_Summaries.csv
    PRS2_All_GWAS_Best_Results.csv
    PRS2_All_GWAS_Merge_Status.csv

Run:
    python PRS2-Calculation-Merge.py
"""

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_JOBS_FILE = "GWAS_jobs.csv"
PRS_TOOLS = [
    {"tool_name": "PRS_Plink", "label": "Plink"},
    {"tool_name": "PRS_PRSice-2", "label": "PRSice-2"},
]
TOOL_PREFIX = {
    "PRS_Plink": "plink",
    "PRS_PRSice-2": "prsice2",
}


def clean_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def load_jobs(jobs_file):
    jobs_file = Path(jobs_file)
    if not jobs_file.exists():
        raise FileNotFoundError(f"Missing jobs file: {jobs_file}")

    jobs = pd.read_csv(jobs_file)
    required = ["job_index", "file_path"]
    missing = [c for c in required if c not in jobs.columns]
    if missing:
        raise ValueError(f"Missing required columns in {jobs_file}: {missing}")

    jobs["job_index"] = pd.to_numeric(jobs["job_index"], errors="coerce").astype("Int64")
    jobs = jobs.dropna(subset=["job_index"]).copy()
    jobs["job_index"] = jobs["job_index"].astype(int)
    return jobs


def get_gwas_output_dir(job):
    file_path = Path(clean_text(job["file_path"]))
    output_feature_file = clean_text(job.get("output_feature_file", ""))
    if output_feature_file:
        return Path(output_feature_file).parent
    return file_path.parent


def get_phenotype(job):
    phenotype = clean_text(job.get("phenotype", ""))
    if phenotype:
        return phenotype

    file_path = Path(clean_text(job["file_path"]))
    if file_path.parts:
        return file_path.parts[0]
    return ""


def get_accession(job, gwas_output_dir):
    accession = clean_text(job.get("accessionId", ""))
    if accession:
        return accession
    return Path(gwas_output_dir).name


def base_metadata_row(job, gwas_output_dir, tool):
    return {
        "job_index": int(job["job_index"]),
        "phenotype": get_phenotype(job),
        "accessionId": get_accession(job, gwas_output_dir),
        "file_path": clean_text(job["file_path"]),
        "gwas_output_dir": str(gwas_output_dir),
        "prs_tool_name": tool["tool_name"],
        "prs_tool_label": tool["label"],
        "prs_dir": str(Path(gwas_output_dir) / "PRS" / tool["tool_name"]),
    }


def read_single_row_csv(path):
    df = pd.read_csv(path)
    if df.empty:
        return None, "empty", "CSV exists but is empty"
    return df.head(1).copy(), "success", ""


def reorder_columns(df, preferred):
    if df.empty:
        return df
    first_cols = [c for c in preferred if c in df.columns]
    other_cols = [c for c in df.columns if c not in first_cols]
    return df[first_cols + other_cols]


def add_sum_columns(df, prefix):
    train_pure = f"{prefix}_Train_pure_prs_mean"
    test_pure = f"{prefix}_Test_pure_prs_mean"
    train_null = f"{prefix}_Train_null_model_mean"
    test_null = f"{prefix}_Test_null_model_mean"
    train_best = f"{prefix}_Train_best_model_mean"
    test_best = f"{prefix}_Test_best_model_mean"

    if train_pure in df.columns and test_pure in df.columns:
        df[f"{prefix}_pure_train_test_sum"] = pd.to_numeric(df[train_pure], errors="coerce") + pd.to_numeric(df[test_pure], errors="coerce")
    if train_null in df.columns and test_null in df.columns:
        df[f"{prefix}_null_train_test_sum"] = pd.to_numeric(df[train_null], errors="coerce") + pd.to_numeric(df[test_null], errors="coerce")
    if train_best in df.columns and test_best in df.columns:
        df[f"{prefix}_best_train_test_sum"] = pd.to_numeric(df[train_best], errors="coerce") + pd.to_numeric(df[test_best], errors="coerce")
    return df


def read_optional_csv(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def infer_plink_reason(prs_dir):
    fold_metrics = read_optional_csv(Path(prs_dir) / "PRS_fold_metrics.csv")
    run_summary = read_optional_csv(Path(prs_dir) / "PRS_run_summary.csv")

    if not run_summary.empty:
        row = run_summary.iloc[0]
        valid_rows = pd.to_numeric(pd.Series([row.get("gwas_for_plink_valid_rows", pd.NA)]), errors="coerce").iloc[0]
        if pd.notna(valid_rows) and float(valid_rows) <= 0:
            return "no valid GWAS variants after FinalGWAS-to-PLINK preparation"

    if fold_metrics.empty:
        return "PRS output files missing"

    if "train_bfile_missing" in fold_metrics.columns:
        missing_train = fold_metrics["train_bfile_missing"].fillna("").astype(str).str.strip()
        if (missing_train != "").any():
            return "missing train genotype bfile components"

    if "test_bfile_missing" in fold_metrics.columns:
        missing_test = fold_metrics["test_bfile_missing"].fillna("").astype(str).str.strip()
        if (missing_test != "").any():
            return "missing test genotype bfile components"

    if "prune_in_snps" in fold_metrics.columns:
        prune = pd.to_numeric(fold_metrics["prune_in_snps"], errors="coerce").fillna(0)
        if (prune <= 0).all():
            return "no variants left after pruning"

    if "valid_snps_for_scoring" in fold_metrics.columns:
        valid_score = pd.to_numeric(fold_metrics["valid_snps_for_scoring"], errors="coerce").fillna(0)
        if (valid_score <= 0).all():
            return "no variants left after clumping or overlap filtering"

    if "clone_train_success" in fold_metrics.columns and (~fold_metrics["clone_train_success"].fillna(False).astype(bool)).any():
        return "failed to create train clumped/pruned genotype files"

    if "clone_test_success" in fold_metrics.columns and (~fold_metrics["clone_test_success"].fillna(False).astype(bool)).any():
        return "failed to create test clumped/pruned genotype files"

    if "score_train_success" in fold_metrics.columns and (~fold_metrics["score_train_success"].fillna(False).astype(bool)).any():
        return "PLINK score failed on train fold(s)"

    if "score_test_success" in fold_metrics.columns and (~fold_metrics["score_test_success"].fillna(False).astype(bool)).any():
        return "PLINK score failed on test fold(s)"

    return "PRS results missing but no specific PLINK reason inferred"


def infer_prsice_reason(prs_dir):
    fold_metrics = read_optional_csv(Path(prs_dir) / "PRS_fold_metrics.csv")
    run_summary = read_optional_csv(Path(prs_dir) / "PRS_run_summary.csv")

    if not run_summary.empty:
        row = run_summary.iloc[0]
        valid_rows = pd.to_numeric(pd.Series([row.get("gwas_for_prsice_valid_rows", pd.NA)]), errors="coerce").iloc[0]
        if pd.notna(valid_rows) and float(valid_rows) <= 0:
            return "no valid GWAS variants after FinalGWAS-to-PRSice preparation"

    if fold_metrics.empty:
        return "PRS output files missing"

    if "train_bfile_missing" in fold_metrics.columns:
        missing_train = fold_metrics["train_bfile_missing"].fillna("").astype(str).str.strip()
        if (missing_train != "").any():
            return "missing train genotype bfile components"

    if "test_bfile_missing" in fold_metrics.columns:
        missing_test = fold_metrics["test_bfile_missing"].fillna("").astype(str).str.strip()
        if (missing_test != "").any():
            return "missing test genotype bfile components"

    if "train_cov_exists_before_run" in fold_metrics.columns:
        train_cov = fold_metrics["train_cov_exists_before_run"].fillna(False).astype(bool)
        if (~train_cov).any():
            return "train PRSice covariate file was not created"

    if "test_cov_exists_before_run" in fold_metrics.columns:
        test_cov = fold_metrics["test_cov_exists_before_run"].fillna(False).astype(bool)
        if (~test_cov).any():
            return "test PRSice covariate file was not created"

    if "train_success" in fold_metrics.columns and (~fold_metrics["train_success"].fillna(False).astype(bool)).any():
        return "PRSice execution failed on train fold(s)"

    if "test_success" in fold_metrics.columns and (~fold_metrics["test_success"].fillna(False).astype(bool)).any():
        return "PRSice execution failed on test fold(s)"

    return "PRS results missing but no specific PRSice reason inferred"


def infer_failure_reason(tool_name, prs_dir, run_summary_status, best_result_status):
    if run_summary_status == "success" and best_result_status == "success":
        return ""

    if tool_name == "PRS_Plink":
        return infer_plink_reason(prs_dir)
    if tool_name == "PRS_PRSice-2":
        return infer_prsice_reason(prs_dir)
    return "unknown reason"


def build_performance_comparison(jobs, status_out, best_out):
    base_rows = []
    for _, job in jobs.sort_values("job_index").iterrows():
        gwas_output_dir = get_gwas_output_dir(job)
        base_rows.append({
            "job_index": int(job["job_index"]),
            "phenotype": get_phenotype(job),
            "accessionId": get_accession(job, gwas_output_dir),
            "file_path": clean_text(job["file_path"]),
            "gwas_output_dir": str(gwas_output_dir),
        })

    comparison = pd.DataFrame(base_rows)

    for tool in PRS_TOOLS:
        tool_name = tool["tool_name"]
        prefix = TOOL_PREFIX[tool_name]

        tool_status = status_out[status_out["prs_tool_name"] == tool_name].copy() if not status_out.empty else pd.DataFrame()
        if not tool_status.empty:
            status_keep = [
                "job_index",
                "run_summary_status",
                "best_result_status",
                "likely_reason",
                "run_summary_error",
                "best_result_error",
                "run_summary_file",
                "best_result_file",
            ]
            tool_status = tool_status[status_keep].copy()
            tool_status = tool_status.rename(columns={
                "run_summary_status": f"{prefix}_run_summary_status",
                "best_result_status": f"{prefix}_best_result_status",
                "likely_reason": f"{prefix}_likely_reason",
                "run_summary_error": f"{prefix}_run_summary_error",
                "best_result_error": f"{prefix}_best_result_error",
                "run_summary_file": f"{prefix}_run_summary_file",
                "best_result_file": f"{prefix}_best_result_file",
            })
            comparison = comparison.merge(tool_status, on="job_index", how="left")

        tool_best = best_out[best_out["prs_tool_name"] == tool_name].copy() if not best_out.empty else pd.DataFrame()
        if not tool_best.empty:
            perf_cols = [
                "job_index",
                "performance_metric",
                "pvalue_label",
                "pvalue",
                "prsice_model",
                "n_folds",
                "Train_pure_prs_mean",
                "Test_pure_prs_mean",
                "Train_null_model_mean",
                "Test_null_model_mean",
                "Train_best_model_mean",
                "Test_best_model_mean",
                "generalisation_gap",
                "best_model_generalisation_gap",
                "test_incremental_auc_best_minus_null",
                "test_incremental_auc_best_minus_pure_prs",
            ]
            perf_cols = [c for c in perf_cols if c in tool_best.columns]
            tool_best = tool_best[perf_cols].copy()
            rename_map = {c: f"{prefix}_{c}" for c in perf_cols if c != "job_index"}
            tool_best = tool_best.rename(columns=rename_map)
            comparison = comparison.merge(tool_best, on="job_index", how="left")

        comparison = add_sum_columns(comparison, prefix)

    return reorder_columns(comparison, [
        "job_index", "phenotype", "accessionId", "file_path", "gwas_output_dir",
        "plink_run_summary_status", "plink_best_result_status", "plink_likely_reason",
        "plink_performance_metric", "plink_pvalue_label", "plink_pvalue",
        "plink_Train_pure_prs_mean", "plink_Test_pure_prs_mean", "plink_pure_train_test_sum",
        "plink_Train_null_model_mean", "plink_Test_null_model_mean", "plink_null_train_test_sum",
        "plink_Train_best_model_mean", "plink_Test_best_model_mean", "plink_best_train_test_sum",
        "prsice2_run_summary_status", "prsice2_best_result_status", "prsice2_likely_reason",
        "prsice2_performance_metric", "prsice2_prsice_model", "prsice2_pvalue_label", "prsice2_pvalue",
        "prsice2_Train_pure_prs_mean", "prsice2_Test_pure_prs_mean", "prsice2_pure_train_test_sum",
        "prsice2_Train_null_model_mean", "prsice2_Test_null_model_mean", "prsice2_null_train_test_sum",
        "prsice2_Train_best_model_mean", "prsice2_Test_best_model_mean", "prsice2_best_train_test_sum",
    ])


def main():
    parser = argparse.ArgumentParser(
        description="Merge PRS run summaries and best results for Plink and PRSice-2."
    )
    parser.add_argument(
        "--jobs-file",
        default=DEFAULT_JOBS_FILE,
        help="Default: GWAS_jobs.csv",
    )
    parser.add_argument(
        "--summary-output",
        default="PRS2_All_GWAS_Run_Summaries.csv",
        help="Merged run-summary output CSV.",
    )
    parser.add_argument(
        "--best-output",
        default="PRS2_All_GWAS_Best_Results.csv",
        help="Merged best-result output CSV.",
    )
    parser.add_argument(
        "--status-output",
        default="PRS2_All_GWAS_Merge_Status.csv",
        help="Merge status output CSV.",
    )
    parser.add_argument(
        "--comparison-output",
        default="PRS2_All_GWAS_Performance_Comparison.csv",
        help="One-row-per-GWAS comparison CSV for Plink vs PRSice-2 performance.",
    )
    args = parser.parse_args()

    jobs = load_jobs(args.jobs_file)

    summary_rows = []
    best_rows = []
    status_rows = []

    print("=" * 100)
    print("MERGE PRS OUTPUTS FOR PLINK AND PRSICE-2")
    print("=" * 100)
    print(f"[JOBS FILE] {args.jobs_file}")
    print(f"[N JOBS]    {len(jobs)}")
    print("=" * 100)

    for _, job in jobs.sort_values("job_index").iterrows():
        gwas_output_dir = get_gwas_output_dir(job)

        for tool in PRS_TOOLS:
            meta = base_metadata_row(job, gwas_output_dir, tool)
            prs_dir = Path(meta["prs_dir"])
            run_summary_file = prs_dir / "PRS_run_summary.csv"
            best_result_file = prs_dir / "PRS_best_result.csv"

            status_entry = dict(meta)
            status_entry.update({
                "run_summary_file": str(run_summary_file),
                "best_result_file": str(best_result_file),
                "run_summary_status": "",
                "best_result_status": "",
                "run_summary_error": "",
                "best_result_error": "",
                "likely_reason": "",
            })

            if run_summary_file.exists():
                try:
                    summary_df, summary_status, summary_error = read_single_row_csv(run_summary_file)
                    status_entry["run_summary_status"] = summary_status
                    status_entry["run_summary_error"] = summary_error
                    if summary_df is not None:
                        summary_df.insert(0, "job_index", meta["job_index"])
                        summary_df.insert(1, "phenotype", meta["phenotype"])
                        summary_df.insert(2, "accessionId", meta["accessionId"])
                        summary_df.insert(3, "prs_tool_name", meta["prs_tool_name"])
                        summary_df.insert(4, "prs_tool_label", meta["prs_tool_label"])
                        summary_df.insert(5, "gwas_output_dir", meta["gwas_output_dir"])
                        summary_df.insert(6, "prs_dir", meta["prs_dir"])
                        summary_df.insert(7, "run_summary_file", str(run_summary_file))
                        summary_df.insert(8, "merge_status", summary_status)
                        summary_rows.append(summary_df)
                except Exception as e:
                    status_entry["run_summary_status"] = "failed"
                    status_entry["run_summary_error"] = f"{type(e).__name__}: {e}"
                    summary_rows.append(pd.DataFrame([{
                        **meta,
                        "run_summary_file": str(run_summary_file),
                        "merge_status": "failed",
                        "merge_error": f"{type(e).__name__}: {e}",
                    }]))
            else:
                status_entry["run_summary_status"] = "missing"
                status_entry["run_summary_error"] = "PRS_run_summary.csv not found"
                summary_rows.append(pd.DataFrame([{
                    **meta,
                    "run_summary_file": str(run_summary_file),
                    "merge_status": "missing",
                    "merge_error": "PRS_run_summary.csv not found",
                }]))

            if best_result_file.exists():
                try:
                    best_df, best_status, best_error = read_single_row_csv(best_result_file)
                    status_entry["best_result_status"] = best_status
                    status_entry["best_result_error"] = best_error
                    if best_df is not None:
                        best_df.insert(0, "job_index", meta["job_index"])
                        best_df.insert(1, "phenotype", meta["phenotype"])
                        best_df.insert(2, "accessionId", meta["accessionId"])
                        best_df.insert(3, "prs_tool_name", meta["prs_tool_name"])
                        best_df.insert(4, "prs_tool_label", meta["prs_tool_label"])
                        best_df.insert(5, "gwas_output_dir", meta["gwas_output_dir"])
                        best_df.insert(6, "prs_dir", meta["prs_dir"])
                        best_df.insert(7, "best_result_file", str(best_result_file))
                        best_df.insert(8, "merge_status", best_status)
                        best_rows.append(best_df)
                except Exception as e:
                    status_entry["best_result_status"] = "failed"
                    status_entry["best_result_error"] = f"{type(e).__name__}: {e}"
                    best_rows.append(pd.DataFrame([{
                        **meta,
                        "best_result_file": str(best_result_file),
                        "merge_status": "failed",
                        "merge_error": f"{type(e).__name__}: {e}",
                    }]))
            else:
                status_entry["best_result_status"] = "missing"
                status_entry["best_result_error"] = "PRS_best_result.csv not found"
                best_rows.append(pd.DataFrame([{
                    **meta,
                    "best_result_file": str(best_result_file),
                    "merge_status": "missing",
                    "merge_error": "PRS_best_result.csv not found",
                }]))

            status_entry["likely_reason"] = infer_failure_reason(
                tool_name=meta["prs_tool_name"],
                prs_dir=prs_dir,
                run_summary_status=status_entry["run_summary_status"],
                best_result_status=status_entry["best_result_status"],
            )
            status_rows.append(status_entry)
            print(
                f"[{meta['prs_tool_label']}] job={meta['job_index']} "
                f"phenotype={meta['phenotype']} accession={meta['accessionId']} "
                f"summary={status_entry['run_summary_status']} "
                f"best={status_entry['best_result_status']} "
                f"reason={status_entry['likely_reason']}"
            )

    summary_out = pd.concat(summary_rows, ignore_index=True) if summary_rows else pd.DataFrame()
    best_out = pd.concat(best_rows, ignore_index=True) if best_rows else pd.DataFrame()
    status_out = pd.DataFrame(status_rows)

    summary_out = reorder_columns(summary_out, [
        "job_index", "phenotype", "accessionId", "prs_tool_name", "prs_tool_label",
        "merge_status", "merge_error", "gwas_output_dir", "prs_dir", "run_summary_file",
        "n_folds_found", "n_folds_train_success", "n_folds_test_success", "n_folds_with_results",
        "score_model", "prsice_stat_mode", "finalgwas_input_rows",
        "gwas_for_plink_valid_rows", "gwas_for_prsice_valid_rows",
    ])

    best_out = reorder_columns(best_out, [
        "job_index", "phenotype", "accessionId", "prs_tool_name", "prs_tool_label",
        "merge_status", "merge_error", "gwas_output_dir", "prs_dir", "best_result_file",
        "prsice_model", "pvalue_label", "pvalue", "trait_type", "performance_metric",
        "n_folds", "Train_pure_prs_mean", "Test_pure_prs_mean",
        "Train_null_model_mean", "Test_null_model_mean",
        "Train_best_model_mean", "Test_best_model_mean",
        "generalisation_gap", "best_model_generalisation_gap",
        "test_incremental_auc_best_minus_null", "test_incremental_auc_best_minus_pure_prs",
    ])

    status_out = reorder_columns(status_out, [
        "job_index", "phenotype", "accessionId", "prs_tool_name", "prs_tool_label",
        "gwas_output_dir", "prs_dir",
        "likely_reason",
        "run_summary_status", "run_summary_error", "run_summary_file",
        "best_result_status", "best_result_error", "best_result_file",
    ])

    summary_out.to_csv(args.summary_output, index=False)
    best_out.to_csv(args.best_output, index=False)
    status_out.to_csv(args.status_output, index=False)
    comparison_out = build_performance_comparison(jobs, status_out, best_out)
    comparison_out.to_csv(args.comparison_output, index=False)

    print("=" * 100)
    print("[DONE]")
    print(f"[SUMMARY OUTPUT] {args.summary_output}")
    print(f"[BEST OUTPUT]    {args.best_output}")
    print(f"[STATUS OUTPUT]  {args.status_output}")
    print(f"[COMPARE OUTPUT] {args.comparison_output}")
    print("=" * 100)

    if not status_out.empty:
        print()
        print("[STATUS PREVIEW]")
        preview_cols = [
            "job_index", "phenotype", "accessionId", "prs_tool_label",
            "run_summary_status", "best_result_status",
        ]
        preview_cols = [c for c in preview_cols if c in status_out.columns]
        print(status_out[preview_cols].head(40).to_string(index=False))

    if not comparison_out.empty:
        print()
        print("[COMPARISON PREVIEW]")
        preview_cols = [
            "job_index", "phenotype", "accessionId",
            "plink_best_result_status", "plink_Test_best_model_mean",
            "prsice2_best_result_status", "prsice2_Test_best_model_mean",
        ]
        preview_cols = [c for c in preview_cols if c in comparison_out.columns]
        print(comparison_out[preview_cols].head(40).to_string(index=False))


if __name__ == "__main__":
    main()
