function [v,names]=features(x,fs)
% x: samples-by-channels
n=size(x,1);nc=size(x,2);v=zeros(1,nc*8);names=cell(1,nc*8);
for ch=1:nc
 a=x(:,ch);d=diff(a);thr=max(std(a)*.05,eps);
 rmsv=sqrt(mean(a.^2));mav=mean(abs(a));vv=var(a,1);wl=sum(abs(d));
 zc=sum(a(1:end-1).*a(2:end)<0 & abs(d)>thr);
 ssc=sum(d(1:end-1).*d(2:end)<0 & abs(d(1:end-1)-d(2:end))>thr);
 p=abs(fft(a)).^2/n;nf=floor(n/2)+1;pow=p(1:nf);f=(0:nf-1)'*fs/n;
 mp=sum(f.*pow)/(sum(pow)+eps);cs=cumsum(pow);idx=find(cs>=cs(end)/2,1);if isempty(idx),idx=1;end
 medf=f(idx);vvv=[rmsv mav vv wl zc ssc mp medf];
 ids=(ch-1)*8+(1:8);v(ids)=vvv;
 base={'RMS','MAV','VAR','WL','ZC','SSC','MNF','MDF'};
 for k=1:8,names{ids(k)}=sprintf('ch%d_%s',ch,base{k});end
end
end
