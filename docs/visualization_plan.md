# AMICA-Python ds004505 Visualization Plan

This document outlines the benchmark figures to generate for the AMICA-Python validation on the ds004505 dual-layer EEG dataset, based on the reference study (Studnicki et al., Sensors 2022).

## What to copy from the ds004505 paper

| Paper figure style | Use for your benchmark? | Why |
| :--- | :--- | :--- |
| **Figure 1: experimental/sensor setup** | Yes, intro only | Show why ds004505 is special: dual-layer scalp/noise EEG during real movement. |
| **Figure 2: artifact-characterization workflow** | Yes | Perfect visual template: time series → epochs → PSD → ERSP → correlations. |
| **Figure 4: PSD by sensor/condition** | Yes | Essential for showing movement artifacts across scalp, noise, IMU, EMG. |
| **Figure 5: ERSP around swing/hit cycle** | Yes | Best visual for event-locked mobile artifact dynamics. |
| **Figure 6: scalp–noise/IMU/EMG correlation topomaps** | Yes | Very strong for showing why noise electrodes matter more than IMUs. |
| **Figure 7: dipolarity bar plot** | Yes, but improve | Their metric is number of ICs with residual variance <15%. Good benchmark target. |
| **Figure 8: ICLabel class bars** | Yes, but improve | Use normalized stacked bars instead of many small bars. |

## Figure List & Design

### 1. Dataset + benchmark design figure (`fig01_ds004505_benchmark_workflow.svg`)
A clean schematic:
```
ds004505 raw BIDS
  ↓
Minimal preprocessing
  ↓
Optional cleaning branch:
  raw / time-reject / 3 Hz HP / iCanClean / AMICA sample rejection
  ↓
ICA methods:
  AMICA-Python JAX-GPU
  AMICA-Python NumPy-CPU
  Picard
  FastICA
  Infomax
  ↓
Metrics:
  runtime, memory, LL, MIR, ICLabel, dipolarity, PSD preservation
```

### 2. PSD fingerprint figure (`fig02_sensor_artifact_psd_by_condition.svg`)
Make one figure with rows: `scalp EEG`, `noise EEG`, `neck EMG`, `body IMU`, `head IMU, if available`.
Columns: `standing baseline`, `stationary hitting`, `moving hitting`, `cooperative`, `competitive`.
For AMICA benchmark, show:
- before ICA
- after rejecting artifact ICs
- after iCanClean + AMICA
- after 3 Hz HP + AMICA

### 3. ERSP / swing-cycle artifact figure (`fig03_hit_locked_ersp_scalp_noise_emg_ic.svg`)
Rows: `scalp`, `noise`, `EMG`, `AMICA artifact IC`, `AMICA brain IC`
Columns: `stationary`, `moving`, `cooperative`, `competitive`
Goal: Show that AMICA preserves neural alpha/beta dynamics while artifact ICs follow hit-locked noise patterns.

### 4. Scalp–noise correlation topomap (`fig04_scalp_noise_imu_emg_correlation_topomaps.svg`)
Topographic maps showing correlation of scalp channels with: matched noise electrode, body IMU, neck EMG.
Versions to show:
- before cleaning
- after iCanClean
- after AMICA artifact IC removal
- after 3 Hz HP

### 5. Pipeline × method benchmark matrix (`fig05_pipeline_method_quality_matrix.svg`)
Rows: Minimal pipelines (`Minimal`, `+ Time Reject`, `+ 3 Hz + Time Reject`, `+ iCanClean`, `+ iCanClean + AMICA rejection`).
Columns: ICA Methods (`AMICA JAX-GPU`, `AMICA NumPy-CPU`, `Picard`, `FastICA`, `Infomax`).
Color metrics: `Brain IC count`, `Dipolar IC count`, `MIR`, `Runtime`, `Peak memory`.

### 6. Dipolarity benchmark figure (`fig06_dipolarity_rv_thresholds_by_method.svg`)
x-axis: method / pipeline
y-axis: % ICs with RV < 15% (also report RV < 10% and < 5%)
Include individual subject paired points and mean ± SEM.

### 7. ICLabel composition figure (`fig07_iclabel_composition_by_method_pipeline.svg`)
Stacked normalized bars for: `Brain`, `Muscle`, `Eye`, `Heart`, `Line noise`, `Channel noise`, `Other`.
Rows/facets by pipeline/method. Treat as proxy QC metric.

### 8. MIR vs dipolarity scatterplot (`fig08_mir_vs_dipolarity_delorme_style.svg`)
x-axis: mutual information reduction (MIR)
y-axis: % dipolar ICs
color: ICA method | marker: preprocessing pipeline | size: runtime or number of brain ICs.

### 9. Runtime-quality Pareto plot (`fig09_runtime_quality_pareto_cpu_gpu.svg`)
x-axis: runtime (log scale)
y-axis: quality metric (MIR, brain IC %, or dipolar IC %)
point size: peak memory | color: method/backend
Answers: Is AMICA GPU faster AND does it preserve/improve quality?

### 10. Low-frequency preservation plot (`fig10_low_frequency_preservation_1hz_3hz_ican_amica.svg`)
x-axis: frequency band (delta, theta, alpha, beta)
y-axis: retained neural power or event-locked power change
Conditions: `1 Hz HP`, `3 Hz HP`, `iCanClean`, `AMICA rejection`.
Answers: Are we improving ICA quality by destroying low-frequency neural information?

## MNE-native figures to generate automatically

| MNE output | Use |
| :--- | :--- |
| `ICA.plot_components()` | Component topomap grids. |
| `ICA.plot_sources()` | Browse component time courses. |
| `ICA.plot_properties()` | Component-level QC cards: topomap, time course, PSD, variance. |
| `mne.Report.add_ica()` | Build one HTML report per subject/method/backend. |
| `mne_icalabel.label_components()` | Component class labels/probabilities. |

## Final Figure List

```
fig01_ds004505_benchmark_workflow.svg
fig02_sensor_artifact_psd_by_condition.svg
fig03_hit_locked_ersp_scalp_noise_emg_ic.svg
fig04_scalp_noise_imu_emg_correlation_topomaps.svg
fig05_pipeline_method_quality_matrix.svg
fig06_dipolarity_rv_thresholds_by_method.svg
fig07_iclabel_composition_by_method_pipeline.svg
fig08_mir_vs_dipolarity_delorme_style.svg
fig09_runtime_quality_pareto_cpu_gpu.svg
fig10_low_frequency_preservation_1hz_3hz_ican_amica.svg
supp01_noise_channel_count_ablation_1_5_9_32_64_120.svg
supp02_seed_stability_wcorr_heatmap.svg
supp03_mne_component_qc_report_examples.html
supp04_slurm_failure_runtime_memory_matrix.svg
```
