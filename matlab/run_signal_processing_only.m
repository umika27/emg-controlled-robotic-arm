function run_signal_processing_only
% Inspect actual dataset signal processing without training a classifier.
c=config('fast');addpath(c.root);c.maxFiles=8;
D=emg.prepare(c);emg.plots(D,c);
fprintf('Saved EMG processing plots to %s\n',c.out);
fprintf('Windows: %d | samples/window: %d | channels: %d\n',numel(D.Y),size(D.X,1),size(D.X,2));
end
