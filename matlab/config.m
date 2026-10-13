function c=config(mode)
if nargin<1,mode='fast';end
c.root=fileparts(mfilename('fullpath'));c.mode=mode;c.fs=200;c.passband=[5 40];
c.windowSec=.2;c.hopSec=.1;c.labels=[1 2 3 4];
c.names={'Rest','Fist','Wrist flexion','Wrist extension'};
c.seed=42;c.maxFiles=8;c.maxWindows=3500;c.epochs=3;c.makeVideo=false;
if strcmpi(mode,'full'),c.maxFiles=inf;c.maxWindows=inf;c.epochs=12;end
c.out=fullfile(c.root,'results');if ~isfolder(c.out),mkdir(c.out);end
end
