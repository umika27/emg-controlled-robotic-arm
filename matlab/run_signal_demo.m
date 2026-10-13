function run_signal_demo
c=config('fast');D=emg.prepare(c);emg.plots(D,c);
save(fullfile(c.out,'signal_demo.mat'),'D','c','-v7.3');disp('Generated five signal-processing plots.');
end
