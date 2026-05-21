import json
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('dark_background')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'axes.facecolor': '#121212',
    'figure.facecolor': '#121212',
    'grid.color': '#333333',
    'text.color': '#e0e0e0',
    'axes.labelcolor': '#e0e0e0',
    'xtick.color': '#e0e0e0',
    'ytick.color': '#e0e0e0',
})

results_dir = '/scratch/sesma/amica_python_validation_outputs'
json_files = glob.glob(os.path.join(results_dir, 'ds004505_sub-*.json'))

data = []
for f in json_files:
    with open(f) as f_in:
        try:
            d = json.load(f_in)
            data.append({
                'subject': d['subject'],
                'backend': d['backend'],
                'device': d['device'],
                'config': f"{d['backend'].upper()} {d['device'].upper()}",
                'runtime_s': d['runtime_s'],
                'n_samples': d['n_samples']
            })
        except Exception as e:
            print(f'Error reading {f}: {e}')

df = pd.DataFrame(data)

# Figure 1: Runtime Comparison for common subjects
common_subjects = df[df['config'] == 'NUMPY CPU']['subject'].tolist()
sub_df = df[df['subject'].isin(common_subjects)].copy()

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=sub_df, x='subject', y='runtime_s', hue='config', ax=ax, palette='muted')
ax.set_yscale('log')
ax.set_ylabel('Runtime (seconds, log scale)')
ax.set_xlabel('Subject ID')
ax.set_title('AMICA Runtime Comparison (ds004505 Table Tennis Dataset)')
plt.tight_layout()
fig.savefig(os.path.join(results_dir, 'runtime_comparison_log.png'), dpi=300)
plt.close(fig)

# Figure 2: JAX GPU Runtimes across all 25 subjects
gpu_df = df[df['config'] == 'JAX GPU'].sort_values('subject')
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(data=gpu_df, x='subject', y='runtime_s', ax=ax, color='#006D77')
ax.set_ylabel('Runtime (seconds)')
ax.set_xlabel('Subject ID')
ax.set_title('JAX GPU Runtime across all 25 Subjects')
plt.xticks(rotation=45)
plt.tight_layout()
fig.savefig(os.path.join(results_dir, 'jax_gpu_runtimes_all_subjects.png'), dpi=300)
plt.close(fig)

print('Plots generated successfully!')
