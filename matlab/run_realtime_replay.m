function run_realtime_replay
% Causal streaming with persistent SOS filtering state, rolling buffers,
% binary RMS activation threshold as a live-style example.
c=config('fast');files=emg.discover(c);rec=emg.loadRecording(files(1));
fs=c.fs;n=round(c.windowSec*fs);hop=round(c.hopSec*fs);
[b,a]=butter(4,c.passband/(fs/2),'bandpass');
zi=zeros(max(length(a),length(b))-1,8);buf=zeros(0,8);
idx=1;times=[];pred=[];rmsValues=[];chunk=12;
% Threshold is illustrative on replay only; real deployment MUST calibrate
% from training users or calibration session (not from held-out samples).
calib=emg.filterSignal(rec.x(1:min(3000,end),:),c);
base=sqrt(mean(calib.^2,2));threshold=quantile(base,.60);
while idx<=size(rec.x,1)
 last=min(size(rec.x,1),idx+chunk-1);
 [y,zi]=filter(b,a,rec.x(idx:last,:),zi);buf=[buf;y]; %#ok<AGROW>
 while size(buf,1)>=n
  seg=buf(1:n,:);v=sqrt(mean(seg.^2,'all'));
  rmsValues(end+1,1)=v;pred(end+1,1)=v>threshold;times(end+1,1)=last/fs; %#ok<AGROW>
  buf(1:min(hop,size(buf,1)),:)=[];
 end
 idx=last+1;
end
fig=figure('Visible','off');stairs(times,pred);ylim([-.1 1.1]);xlabel('Approximate time (s)');ylabel('Activation');
title('Streaming replay: illustrative RMS threshold, not multiclass CNN');grid on;
exportgraphics(fig,fullfile(c.out,'12_streaming_replay.png'));close(fig);
save(fullfile(c.out,'streaming_replay.mat'),'times','pred','rmsValues','threshold');
fprintf('Streaming replay finished (%d output windows).\n',numel(pred));
end
