# EMG-Controlled Assistive Robotic Arm — MATLAB + Simulink + 3D

**Included:** original UCI EMG Data for Gestures ZIP (36 subjects / 72 recordings), modular MATLAB signal processing, 64 time/frequency features, subject-disjoint train/val/test evaluation, activation baseline, bagged-tree multiclass baseline, temporal CNN experiment, causal streaming replay, MATLAB joint control simulation, auto-generated editable Simulink `.slx` model, and a 3D animated arm GIF.

## Quick start (MATLAB desktop)

1. Unzip this archive, open MATLAB, **Current Folder** = extracted `EMG_Robotic_Arm_MATLAB_Final`.
2. Run `run_tests` (component checks).
3. Run `run_signal_demo` to visualize real dataset EMG and filtering.
4. Run `run_control_demo` to see desired/actual position tracking.
5. Run `run_3d_demo` to generate `results/09_arm_animation.gif` and snapshot.
6. Run `build_simulink_model` (requires Simulink) to CREATE and simulate editable `models/EMG_Arm_PID_Demo.slx`.
7. Run `run_realtime_replay` to generate causal streaming-replay plot.
8. Run `run_all('fast')` for end-to-end training/plots. This is an indicative fast experiment, not a definitive benchmark.
9. Run `run_all('full')` for the full 72-recording dataset, longer CNN training and deeper analysis (can take significantly longer than 30 minutes).

### MATLAB requirements

MATLAB (recent release recommended), Signal Processing Toolbox for `butter/filtfilt/freqz/pwelch`, Statistics and Machine Learning Toolbox for bagged-tree models, Deep Learning Toolbox for CNN training, and Simulink for editable block diagram. Base MATLAB provides control simulation and 3D animation. `exportgraphics`, `tiledlayout` and `confusionchart` require a modern MATLAB release.

### Critical scientific qualifications

* Dataset MYO armband nominally samples about 200 Hz; original timestamps are nonuniform/repeated. This demo analyzes the sequence at a **nominal** 200 Hz, rather than claiming verified time-domain resampling. The dataset's empirical interval statistics are recorded in `D.sampleStats`. **Timing and band limits must be reevaluated on real hardware.**
* The 5–40 Hz Butterworth band is a **conservative demonstration choice**, not a universal sEMG passband. Do not apply 20–450 Hz filtering to a 200 Hz device.
* **No data leakage**: windows stay within homogeneous labels/recordings, train/test partitioned by subject; normalization fitted on training subjects only. Random Forest and CNN may differ if you modify their configuration.
* The temporal CNN is implemented with temporal 2D kernels for compatibility with older MATLAB `trainNetwork` APIs; it is a temporal CNN architecture, but not the newer dedicated 1D convolution block.
* The **3D arm is a stylized kinematic visualization**; only the shoulder yaw follows the simulated joint. Other degrees of freedom are illustrative; no CAD geometry or measured robot parameters are claimed.
* The PI arm plant (`J=0.08`, `B=0.55`, `Kp=2`, `Ki=0.5`) is assumed for demonstration. It is **not identified from a physical robot**, and the 3D demonstration is not a Simscape Multibody physics model.
* The Simulink model uses a gesture-derived reference **scripted for demo purposes**; actual neural predictions are not yet used to command the controller. Streaming replay is also a binary illustrative example.
* Trained model metrics are **not provided in advance**: run the scripts in MATLAB and report actual measured results. MATLAB was unavailable in the authoring environment, so full scripts and generated `.slx` have **not been executed/verified in MATLAB**.
* `run_all` automatically calls the 3D GIF generation; this may slow the quick run. Use separate commands to show only the required subsystem.

## Ownership and integration

- **Signal team** owns dataset reader, `+emg/filterSignal.m`, `+emg/features.m`, `+emg/prepare.m`, filtering and window quality.
- **CNN team** owns `+emg/trainCNN.m`, architecture, training, metrics and inference thresholding.
- **Control team** owns `run_control_demo.m`, `build_simulink_model.m`, the actuator model and tuning.
- Future interface: `(gesture ID, confidence, timestamp) → safe gesture-to-joint reference → feedback controller → joint angle`.

## Expected figures

`results/01_raw_filtered.png`, `02_classes.png`, `03_rms_distributions.png`, `04_filter_response.png`, `05_psd.png`, `06_rf_confusion.png`, `07_cnn_confusion.png`, `08_control_tracking.png`, `09_arm_animation.gif`, `10_arm_3d_snapshot.png`, `11_simulink_tracking.png`, and `12_streaming_replay.png` (generated when their respective scripts run).

Read `docs/EXPERIMENTS.md` for discussion points and the review demonstration sequence.

## NEW: Interactive trained-data-driven 3D arm (dataset replay)

1. Run `run_all('fast')` to create `results/models.mat` and `results/processed_emg.mat`.
2. Run `run_interactive_emg_arm` to open an interactive UI.
3. Choose **CNN** or **Random Forest**, Play/Pause, Next window, Reset, seek through the recording, and change replay speed.
4. The selected classifier predicts each held-out EMG window. Its prediction maps to a desired joint angle, which drives a simple simulated PD joint and a 3D arm display. The ground-truth label is shown only for comparison, never used as the motion command.
5. For the signal-processing-only demonstration, run `run_signal_processing_only`. It produces raw-vs-filtered, PSD, filter frequency response and feature plots.

**Limitations:** The 3D viewer uses preprocessed held-out dataset windows and does not acquire real EMG in real time. Its controller is a simplified illustrative PD simulation, not the separately generated Simulink PID model. Filtering is offline zero-phase in this replay, not causal streaming. MATLAB execution and model compatibility have not been verified in this environment. The viewer requires MATLAB UI components and the models produced by `run_all`.
