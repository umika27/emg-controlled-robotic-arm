function run_interactive_emg_arm
% INTERACTIVE EMG-DRIVEN ARM: uses trained CNN/RF predictions on held-out EMG.
% Dataset replay is offline-filtered (NOT a live acquisition or causal validation).
% Controls: Play/Pause, Next, Reset, classifier choice, replay speed, manual seek.
c=config('fast'); addpath(c.root);
modelFile=fullfile(c.out,'models.mat'); dataFile=fullfile(c.out,'processed_emg.mat');
if ~isfile(modelFile)||~isfile(dataFile)
 error(['Models/data not found. Run run_all(''fast'') first. ', ...
 'This creates trained MATLAB models and preprocessed dataset.']);
end
m=load(modelFile,'B','C');d=load(dataFile,'D');D=d.D;
idx=find(D.test); assert(~isempty(idx),'No held-out test windows.');
% Restrict to a single held-out recording for coherent time-ordered playback.
recordings=unique(D.recording(idx),'stable');
counts=arrayfun(@(r)nnz(D.test & D.recording==r),recordings);
[~,j]=max(counts);idx=idx(D.recording(idx)==recordings(j));
assert(~isempty(idx),'No windows for selected recording');
if numel(idx)>400,idx=idx(1:400);end
N=numel(idx); labels=c.names; targetAngles=deg2rad([0 25 -25 50]);
% Predictions are computed from the actual held-out windows, never true labels.
rfAvailable=isfield(m.B,'rf') && ~isempty(m.B.rf);
cnnAvailable=isfield(m.C,'net') && ~isempty(m.C.net);
assert(rfAvailable||cnnAvailable,'No trained classifier found in models.mat.');
prf=[];pcnn=[];
if rfAvailable
 prf=double(predict(m.B.rf,D.Fn(idx,:)));
end
if cnnAvailable
 X=reshape(D.Xn(:,:,idx),size(D.Xn,1),8,1,[]);
 cp=classify(m.C.net,X,'ExecutionEnvironment','cpu');
 pcnn=nan(N,1);
 for z=1:N,pcnn(z)=find(strcmp(cellstr(cp(z)),labels),1);end
end
methods={};if cnnAvailable,methods{end+1}='CNN';end
if rfAvailable,methods{end+1}='Random Forest';end
f=uifigure('Name','EMG → trained classifier → PID → 3D arm','Position',[100 90 1170 730]);
ax=uiaxes(f,'Position',[20 175 720 520]);grid(ax,'on');axis(ax,'equal');view(ax,35,23);
xlim(ax,[-1.3 1.3]);ylim(ax,[-1.3 1.3]);zlim(ax,[0 1.5]);
xlabel(ax,'X (m)');ylabel(ax,'Y (m)');zlabel(ax,'Z (m)');hold(ax,'on');
plot3(ax,[0 0],[0 0],[0 .15],'k-','LineWidth',8);
h1=plot3(ax,nan,nan,nan,'-o','LineWidth',10,'MarkerSize',12,'Color',[.08 .4 .8]);
h2=plot3(ax,nan,nan,nan,'-o','LineWidth',8,'MarkerSize',11,'Color',[.95 .42 .12]);
href=plot3(ax,nan,nan,nan,'p','MarkerSize',12,'MarkerFaceColor',[.2 .75 .25]);
wave=uiaxes(f,'Position',[745 395 400 260]);xlabel(wave,'Sample index');ylabel(wave,'Normalized EMG');title(wave,'Current EMG window (channel 1)');grid(wave,'on');
status=uilabel(f,'Position',[745 345 420 35],'FontSize',15,'FontWeight','bold','Text','Ready');
truth=uilabel(f,'Position',[745 305 410 35],'Text','Ground truth: —');
angles=uilabel(f,'Position',[745 265 410 35],'Text','Target: — | Actual: —');
method=uidropdown(f,'Position',[745 215 200 30],'Items',methods,'Value',methods{1});
uilabel(f,'Position',[960 216 150 25],'Text','Classifier');
play= uibutton(f,'push','Position',[25 110 120 35],'Text','Play','ButtonPushedFcn',@toggle);
next= uibutton(f,'push','Position',[160 110 120 35],'Text','Next window','ButtonPushedFcn',@nextStep);
reset= uibutton(f,'push','Position',[295 110 120 35],'Text','Reset','ButtonPushedFcn',@resetDemo);
uilabel(f,'Position',[450 116 75 24],'Text','Speed');
speed=uidropdown(f,'Position',[520 110 105 30],'Items',{'0.5x','1x','2x','5x'},'Value','1x');
seek=uislider(f,'Position',[40 65 1060 3],'Limits',[1 max(N,2)],'Value',1,'ValueChangedFcn',@seekTo);
uilabel(f,'Position',[20 18 1120 35], 'Text', ...
 'DATASET REPLAY: predictions from trained model on held-out EMG windows. No live sensor or physical robot connected.');
k=1;theta=0;omega=0;isPlaying=false;closed=false;
f.CloseRequestFcn=@onClose; updateView();
while isvalid(f) && ~closed
 if isPlaying
  nextStep();
  if isvalid(f),pause(0.10/str2double(erase(speed.Value,'x')));end
 else
  pause(.05);
 end
 drawnow limitrate;
end
 function toggle(~,~)
  isPlaying=~isPlaying;
  if isPlaying,play.Text='Pause';else,play.Text='Play';end
 end
 function nextStep(~,~)
  if k<N,k=k+1;else,isPlaying=false;play.Text='Play';end
  updateView();
 end
 function resetDemo(~,~)
  isPlaying=false;play.Text='Play';k=1;theta=0;omega=0;updateView();
 end
 function seekTo(src,~)
  k=max(1,min(N,round(src.Value)));theta=0;omega=0;updateView();
 end
 function updateView()
  if ~isvalid(f),return;end
  if strcmp(method.Value,'CNN'),p=pcnn(k);else,p=prf(k);end
  if ~isfinite(p)||p<1||p>numel(targetAngles)
   desired=theta;gesture='Unknown (hold)';
  else
   desired=targetAngles(p);gesture=labels{p};
  end
  % Illustrative stable PD joint dynamics, integrated for one window interval.
  dt=c.hopSec;h=.005;
  for ii=1:max(1,round(dt/h))
   err=desired-theta;
   torque=6*err-1.6*omega;
   acc=(torque-0.15*omega)/0.2;
   omega=omega+h*acc;theta=theta+h*omega;
   theta=max(-pi/2,min(pi/2,theta));
  end
  elbow=pi/3;pitch=pi/5;base=[0 0 .15];L1=.65;L2=.5;
  p1=base+L1*[cos(theta)*cos(pitch),sin(theta)*cos(pitch),sin(pitch)];
  p2=p1+L2*[cos(theta)*cos(pitch+elbow),sin(theta)*cos(pitch+elbow),sin(pitch+elbow)];
  set(h1,'XData',[base(1) p1(1)],'YData',[base(2) p1(2)],'ZData',[base(3) p1(3)]);
  set(h2,'XData',[p1(1) p2(1)],'YData',[p1(2) p2(2)],'ZData',[p1(3) p2(3)]);
  q1=base+L1*[cos(desired)*cos(pitch),sin(desired)*cos(pitch),sin(pitch)];
  q2=q1+L2*[cos(desired)*cos(pitch+elbow),sin(desired)*cos(pitch+elbow),sin(pitch+elbow)];
  set(href,'XData',q2(1),'YData',q2(2),'ZData',q2(3));
  plot(wave,double(D.Xn(:,1,idx(k))),'Color',[.15 .35 .75]);grid(wave,'on');
  status.Text=sprintf('%s prediction: %s | window %d/%d',method.Value,gesture,k,N);
  trueLabel=D.Y(idx(k));truth.Text=sprintf('Ground truth: %s',labels{trueLabel});
  angles.Text=sprintf('Target: %.1f deg | Actual: %.1f deg',rad2deg(desired),rad2deg(theta));
  seek.Value=k;
 end
 function onClose(~,~)
  closed=true;isPlaying=false;delete(f);
 end
end
