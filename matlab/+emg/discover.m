function files=discover(c)
folder=fullfile(c.root,'dataset','EMG_data_for_gestures-master');
if ~isfolder(folder)
 z=fullfile(c.root,'dataset','emg+data+for+gestures.zip');
 assert(isfile(z),'Dataset ZIP missing: %s',z);unzip(z,fullfile(c.root,'dataset'));
end
files=dir(fullfile(folder,'*','*.txt'));
assert(~isempty(files),'No EMG TXT recordings found.');
[~,ix]=sort(strcat(string({files.folder})',filesep,string({files.name})'));files=files(ix);
if isfinite(c.maxFiles), files=files(1:min(numel(files),c.maxFiles));end
end
