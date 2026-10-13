function run_all(mode)
% RUN_ALL('fast') for rapid demonstration; RUN_ALL('full') for all 72 files.
if nargin<1,mode='fast';end
close all;clc;c=config(mode);rng(c.seed);addpath(c.root);
assert(exist('butter','file')==2,'Signal Processing Toolbox required.');
tic;D=emg.prepare(c);emg.plots(D,c);B=emg.baselines(D,c);C=emg.trainCNN(D,c);
save(fullfile(c.out,'processed_emg.mat'),'D','c','-v7.3');
try,save(fullfile(c.out,'models.mat'),'B','C','-v7.3');catch,warning('Unable to save full model objects');end
T=table(B.binaryBA,B.rfAccuracy,B.rfMacroF1,C.accuracy,C.macroF1,'VariableNames',{'ActivationBalancedAccuracy','RFAccuracy','RFMacroF1','CNNAccuracy','CNNMacroF1'});
writetable(T,fullfile(c.out,'metrics.csv'));disp(T);
run_control_demo;run_3d_demo;
fprintf('Finished in %.1f s. All results in %s\n',toc,c.out);
end
