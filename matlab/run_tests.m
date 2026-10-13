function run_tests
c=config('fast');assert(c.fs==200);assert(c.passband(2)<c.fs/2);
x=randn(500,8);[y,~,~]=emg.filterSignal(x,c);assert(isequal(size(y),size(x)));
[f,n]=emg.features(x(1:40,:),200);assert(numel(f)==64&&numel(n)==64);
assert(abs(f(1)-sqrt(mean(x(1:40,1).^2)))<1e-9);
assert(abs(emg.macroF1([1;2;3],[1;2;3],[1 2 3])-1)<1e-9);
% Verify parsing a real, original dataset file.
files=emg.discover(c);rec=emg.loadRecording(files(1));assert(size(rec.x,2)==8);
assert(numel(rec.y)==size(rec.x,1));
% Verify Nyquist protection.
bad=c;bad.passband=[5 120];failed=false;
try,emg.filterSignal(x,bad);catch,failed=true;end
assert(failed,'Expected illegal filter cutoff to fail');
disp('PASS: MATLAB component smoke tests. Full experimental validation requires run_all.');
end
