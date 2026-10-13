function rec=loadRecording(f)
p=fullfile(f.folder,f.name);T=readtable(p,'FileType','text','Delimiter','\t');
assert(width(T)>=10,'Expected 10 columns in %s',p);
rec.t=double(T{:,1})/1000;rec.x=double(T{:,2:9});rec.y=double(T{:,10});
[~,sn]=fileparts(f.folder);rec.subject=str2double(sn);rec.name=f.name;
assert(size(rec.x,2)==8 && all(isfinite(rec.x(:))),'Malformed channels');
assert(all(isfinite(rec.t)) && all(diff(rec.t)>=0),'Nonmonotonic timestamps');
end
