function score=macroF1(y,p,labels)
f=zeros(numel(labels),1);
for i=1:numel(labels)
 c=labels(i);tp=nnz(y==c&p==c);fp=nnz(y~=c&p==c);fn=nnz(y==c&p~=c);
 f(i)=2*tp/max(2*tp+fp+fn,eps);
end
score=mean(f);
end
