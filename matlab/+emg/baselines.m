function M=baselines(D,c)
y=D.Y;tr=D.train;te=D.test;
% Binary activation threshold calibrated on training only.
energy=mean(D.F(:,1:8:end),2);candidates=quantile(energy(tr),linspace(.05,.95,51));best=-inf;threshold=nan;
for t=candidates
 p=energy(tr)>t;gt=y(tr)~=1;ba=(mean(p(gt))+mean(~p(~gt)))/2;
 if isfinite(ba)&&ba>best,best=ba;threshold=t;end
end
p=energy(te)>threshold;gt=y(te)~=1;
M.binaryBA=(mean(p(gt))+mean(~p(~gt)))/2;M.activationThreshold=threshold;
M.rf=[];M.rfAccuracy=nan;M.rfMacroF1=nan;
if exist('fitcensemble','file')==2
 M.rf=fitcensemble(D.Fn(tr,:),y(tr),'Method','Bag','NumLearningCycles',60,'Learners',templateTree('MinLeafSize',4));
 pred=predict(M.rf,D.Fn(te,:));M.rfAccuracy=mean(pred==y(te));M.rfMacroF1=emg.macroF1(y(te),pred,c.labels);
 fig=figure('Visible','off');confusionchart(y(te),pred);title('RF: held-out subjects');saveas(fig,fullfile(c.out,'06_rf_confusion.png'));close(fig);
end
fprintf('Binary balanced accuracy %.3f | RF accuracy %.3f | RF macro F1 %.3f\n',M.binaryBA,M.rfAccuracy,M.rfMacroF1);
end
