function names=featureNames()
names=cell(1,64);base={'RMS','MAV','VAR','WL','ZC','SSC','MNF','MDF'};
for ch=1:8,for k=1:8,names{(ch-1)*8+k}=sprintf('ch%d_%s',ch,base{k});end,end
end
