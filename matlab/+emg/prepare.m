function D=prepare(c)
files=emg.discover(c);win=round(c.windowSec*c.fs);hop=round(c.hopSec*c.fs);
X=zeros(win,8,0,'single');F=zeros(0,64);Y=zeros(0,1);S=zeros(0,1);R=zeros(0,1);
D.demo=[];D.sampleStats=[];
for k=1:numel(files)
 rec=emg.loadRecording(files(k));d=diff(rec.t);d=d(d>0);
 D.sampleStats(end+1,:)=[rec.subject median(d) mean(diff(rec.t)==0)]; %#ok<AGROW>
 if isempty(D.demo),D.demo=rec;end
 if size(rec.x,1)<win+50,continue;end
 xf=emg.filterSignal(rec.x,c);
 if k==1,D.demo.filtered=xf;end
 for st=1:hop:(size(xf,1)-win+1)
  ids=st:st+win-1;lab=rec.y(ids);
  if ~all(lab==lab(1)) || ~ismember(lab(1),c.labels),continue;end
  piece=xf(ids,:);fv=emg.features(piece,c.fs);
  X(:,:,end+1)=single(piece);F(end+1,:)=fv;Y(end+1)=lab(1);
  S(end+1)=rec.subject;R(end+1)=k;
 end
 fprintf('Loaded %2d/%2d: %s\n',k,numel(files),rec.name);
end
assert(~isempty(Y),'No usable gesture windows found');
% Subject-based split, deterministic and independent of directory ordering.
subjects=unique(S); rng(c.seed); subjects=subjects(randperm(numel(subjects)));
a=max(1,floor(.65*numel(subjects)));b=max(a+1,floor(.8*numel(subjects)));
if numel(subjects)<3,error('Need at least three subjects for leakage-free split. Increase maxFiles.');end
b=min(b,numel(subjects)-1);
tr=ismember(S,subjects(1:a));va=ismember(S,subjects(a+1:b));te=ismember(S,subjects(b+1:end));
assert(any(tr)&&any(va)&&any(te),'Invalid subject partitions');
% Limit training volume without contaminating partitions.
if isfinite(c.maxWindows)&&numel(Y)>c.maxWindows
idxTr = find(tr, ceil(c.maxWindows*.65), 'first');
idxVa = find(va, ceil(c.maxWindows*.15), 'first');
idxTe = find(te, ceil(c.maxWindows*.20), 'first');

ix = unique([idxTr(:); idxVa(:); idxTe(:)]); X=X(:,:,ix);F=F(ix,:);Y=Y(ix);S=S(ix);R=R(ix);tr=tr(ix);va=va(ix);te=te(ix);
end
D.X=X;D.F=F;D.Y=Y;D.subject=S;D.recording=R;D.train=tr;D.val=va;D.test=te;D.fs=c.fs;
D.featureNames=emg.featureNames();
D.mu=mean(F(tr,:),1);D.sd=std(F(tr,:),0,1);D.sd(D.sd<eps)=1;
D.Fn=(F-D.mu)./D.sd;
A=double(X(:,:,tr));D.cmu=reshape(mean(A,[1 3]),1,8);D.csd=reshape(std(reshape(permute(A,[1 3 2]),[],8),0,1),1,8);
D.csd(D.csd<eps)=1;
D.Xn=single((double(X)-reshape(D.cmu,1,8,1))./reshape(D.csd,1,8,1));
assert(~any(ismember(unique(S(tr)),unique(S(te)))),'Subject leakage');
fprintf('Windows: %d | train %d | val %d | test %d | subjects %d\n',numel(Y),nnz(tr),nnz(va),nnz(te),numel(unique(S)));
end
