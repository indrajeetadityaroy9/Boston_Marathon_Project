"""Publication figures for the Boston Marathon finish-time analysis."""
import numpy as np
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

# AISTATS formatting: 3.25in column width, 9pt base font
_RC = {
    'figure.figsize': (3.25, 2.4),
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
}


def plot_ablation_bar(ablation_df, scale, outpath):
    """M0-M5 RMSE bar chart with BCa CI error bars."""
    with plt.rc_context(_RC):
        fig, ax = plt.subplots()
        stages = ablation_df['stage'].tolist()
        rmses = ablation_df['rmse_seconds'].to_numpy()
        lo = rmses - ablation_df['rmse_ci_lower'].to_numpy()
        hi = ablation_df['rmse_ci_upper'].to_numpy() - rmses
        colors = ['0.7'] * 4 + ['0.4'] * 2
        ax.barh(stages[::-1], rmses[::-1], xerr=[lo[::-1], hi[::-1]],
                color=colors[::-1], edgecolor='black', linewidth=0.5, capsize=2)
        ax.set_xlabel('RMSE (seconds)')
        ax.set_title('Nested Model Comparison')
        ax.spines[['top', 'right']].set_visible(False)
        fig.savefig(outpath)
        plt.close(fig)
    print(f"  wrote {outpath}")


def plot_checkpoint_rmse(cp_df, outpath):
    """RMSE vs checkpoint curve."""
    with plt.rc_context(_RC):
        fig, ax = plt.subplots()
        no_hist = cp_df[cp_df['variant'] == 'no_runner_history']
        ax.plot(range(len(no_hist)), no_hist['rmse_seconds'].to_numpy(), 'o-', color='black', markersize=4, linewidth=1.2)
        ax.set_xticks(range(len(no_hist)))
        ax.set_xticklabels(no_hist['checkpoint'].tolist(), rotation=45, ha='right')
        ax.set_ylabel('RMSE (seconds)')
        ax.set_title('Prediction Error by Checkpoint')
        ax.spines[['top', 'right']].set_visible(False)
        fig.savefig(outpath)
        plt.close(fig)
    print(f"  wrote {outpath}")


def plot_coverage_diagnostic(coverage_rows, outpath):
    """Horizontal bar chart of empirical coverage with 90% reference line."""
    with plt.rc_context({**_RC, 'figure.figsize': (3.25, 3.0)}):
        fig, ax = plt.subplots()
        labels = [r['method'] for r in coverage_rows]
        covs = [r['coverage'] for r in coverage_rows]
        y = range(len(labels))
        ax.barh(list(y)[::-1], covs[::-1], color='0.6', edgecolor='black', linewidth=0.5)
        ax.axvline(0.90, color='black', linestyle='--', linewidth=0.8, label='nominal 90%')
        ax.set_yticks(list(y)[::-1])
        ax.set_yticklabels(labels[::-1], fontsize=7)
        ax.set_xlabel('Empirical Coverage')
        ax.set_title('Coverage Diagnostic')
        ax.legend(loc='lower right', fontsize=7)
        ax.spines[['top', 'right']].set_visible(False)
        fig.savefig(outpath)
        plt.close(fig)
    print(f"  wrote {outpath}")


def plot_conformity_shift(cal_scores, test_scores, outpath):
    """Overlaid density plots of conformity score distributions."""
    with plt.rc_context(_RC):
        fig, ax = plt.subplots()
        bins = np.linspace(min(cal_scores.min(), test_scores.min()), np.percentile(np.concatenate([cal_scores, test_scores]), 99), 50)
        ax.hist(cal_scores, bins=bins, density=True, alpha=0.5, color='0.6', edgecolor='black', linewidth=0.3, label='calibration')
        ax.hist(test_scores, bins=bins, density=True, alpha=0.5, color='0.3', edgecolor='black', linewidth=0.3, label='test')
        ax.set_xlabel('Conformity Score')
        ax.set_ylabel('Density')
        ax.set_title('Conformity Score Shift at 5K')
        ax.legend(fontsize=7)
        ax.spines[['top', 'right']].set_visible(False)
        fig.savefig(outpath)
        plt.close(fig)
    print(f"  wrote {outpath}")
