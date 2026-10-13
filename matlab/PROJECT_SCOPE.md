# Deliverables and responsibility boundaries

Signal processing: filtering, plots, windowing, feature extraction, train-only normalization, data packaging.
Gesture recognition: Random Forest and CNN; evaluate subject-disjoint classification.
Control: PID and simulated joint plant, reference tracking plots.
Visualization: programmatically generated Simulink block diagram and 3D animation.

**Honesty note:** This package demonstrates all blocks but does not yet feed measured EMG classifier predictions into the PID plant. A fully integrated real-time EMG-to-robot demonstration requires mapping replay predictions to a reference signal, safety gating, synchronized timestamps, and physical sensor/motor validation.
