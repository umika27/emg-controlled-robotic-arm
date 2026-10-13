function tests=test_emg_math
tests=functiontests(localfunctions);
end
function testFeatures(t)
x=ones(40,8,2,'single');F=emg_features(x,200);
verifyEqual(t,size(F),[2 56]);verifyEqual(t,F(1,1),1,'AbsTol',1e-8);
end
function testNyquist(t)
didFail=false;try,filter_emg(randn(100,8),200,5,120,4,false);catch,didFail=true;end
verifyTrue(t,didFail);
end
function testNormalize(t)
W=single(randn(40,8,10));[Z,s]=normalize_windows(W,[true(5,1);false(5,1)]);
verifyEqual(t,size(Z),size(W));verifyTrue(t,all(isfinite(Z(:))));verifyEqual(t,numel(s.mean),8);
end
