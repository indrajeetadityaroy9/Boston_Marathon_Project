"""Publication LaTeX tables for the Boston Marathon finish-time analysis."""
import pandas as pd


def _write(content, outpath):
    with open(outpath, 'w') as f:
        f.write(content)
    print(f"  wrote {outpath}")


def generate_data_summary(stats, outpath):
    """Dataset description table."""
    lines = [
        r'\begin{tabular}{lr}',
        r'\toprule',
        r'Property & Value \\',
        r'\midrule',
        f"Rows & {stats['n_rows']:,} \\\\",
        f"Columns & {stats['n_columns']} \\\\",
        f"Years & {stats['years']} \\\\",
        f"Distinct runners & {stats['n_distinct_runners']:,} \\\\",
        f"Complete split records & {stats['n_complete_splits']:,} \\\\",
        f"Repeat-runner ICC & {stats['icc']:.4f} \\\\",
        f"Log-seconds skewness & {stats['log_skewness']:.4f} \\\\",
        r'\bottomrule',
        r'\end{tabular}',
    ]
    _write('\n'.join(lines), outpath)


def generate_ablation_table(ablation_df, scale, outpath):
    """M0-M5 nested model comparison table."""
    lines = [
        r'\begin{tabular}{llrrr}',
        r'\toprule',
        r'Stage & Model & RMSE (s) & $\Delta$ (s) & $\times$scale \\',
        r'\midrule',
    ]
    for _, r in ablation_df.iterrows():
        rmse_str = f"{r['rmse_seconds']:.1f}"
        if pd.isna(r['improvement_seconds']):
            lines.append(f"{r['stage']} & {r['description']} & {rmse_str} & --- & --- \\\\")
        else:
            delta = f"{r['improvement_seconds']:+.1f}"
            xs = f"{r['improvement_seconds'] / scale:+.2f}$\\times$"
            lines.append(f"{r['stage']} & {r['description']} & {rmse_str} & {delta} & {xs} \\\\")
    lines += [r'\bottomrule', r'\end{tabular}']
    _write('\n'.join(lines), outpath)


def generate_checkpoint_table(cp_df, outpath):
    """Per-checkpoint RMSE table."""
    no_hist = cp_df[cp_df['variant'] == 'no_runner_history']
    lines = [
        r'\begin{tabular}{lrr}',
        r'\toprule',
        r'Checkpoint & $\alpha$ & RMSE (s) \\',
        r'\midrule',
    ]
    for _, r in no_hist.iterrows():
        lines.append(f"{r['checkpoint']} & {r['alpha']:.2f} & {r['rmse_seconds']:.1f} \\\\")
    lines += [r'\bottomrule', r'\end{tabular}']
    _write('\n'.join(lines), outpath)


def generate_coverage_table(coverage_rows, outpath):
    """Coverage diagnostic table."""
    lines = [
        r'\begin{tabular}{lr}',
        r'\toprule',
        r'Method & Coverage (\%) \\',
        r'\midrule',
    ]
    for r in coverage_rows:
        lines.append(f"{r['method']} & {r['coverage'] * 100:.1f} \\\\")
    lines += [r'\bottomrule', r'\end{tabular}']
    _write('\n'.join(lines), outpath)


def generate_recalibration_table(recal_df, outpath):
    """Honest recalibration results table."""
    lines = [
        r'\begin{tabular}{llrrr}',
        r'\toprule',
        r'Method & Checkpoint & Baseline (\%) & Corrected (\%) & Width (s) \\',
        r'\midrule',
    ]
    for _, r in recal_df.iterrows():
        lines.append(f"{r['method']} & {r['checkpoint']} & {r['baseline_coverage'] * 100:.1f} & {r['corrected_coverage'] * 100:.1f} & {r['interval_width']:.0f} \\\\")
    lines += [r'\bottomrule', r'\end{tabular}']
    _write('\n'.join(lines), outpath)
