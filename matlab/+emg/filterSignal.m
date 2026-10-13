function [y,b,a]=filterSignal(x,c,stream)
% Stable MATLAB Butterworth band-pass. Streaming uses a causal filter;
% offline filtering uses zero-phase forward/backward filtering.
if nargin<3,stream=false;end
assert(c.passband(2)<c.fs/2 && c.passband(1)>0,'Bandpass violates Nyquist');
[b,a]=butter(4,c.passband/(c.fs/2),'bandpass');
if stream
 y=filter(b,a,double(x));
else
 if size(x,1)<3*max(length(a),length(b)),error('Recording too short for offline filtering');end
 y=filtfilt(b,a,double(x));
end
end
