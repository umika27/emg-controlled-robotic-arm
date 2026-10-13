function M=trainCNN(D,c)
M.net=[];M.accuracy=nan;M.macroF1=nan;
if exist('trainNetwork','file')~=2
 warning('Deep Learning Toolbox not installed; CNN skipped.');return;
end
n=size(D.Xn,1);X=reshape(D.Xn,n,8,1,[]);
% Temporal kernels (5-by-1) learn within-channel time structure; later
% 1-by-8 fusion learns across channels. Implemented with image CNN layers.
layers=[imageInputLayer([n 8 1],'Normalization','none')
 convolution2dLayer([5 1],16,'Padding','same');batchNormalizationLayer;reluLayer
 convolution2dLayer([1 8],16,'Padding','same');reluLayer
 maxPooling2dLayer([2 1],'Stride',[2 1]);convolution2dLayer([5 1],32,'Padding','same');reluLayer
 globalAveragePooling2dLayer;fullyConnectedLayer(numel(c.labels));softmaxLayer;classificationLayer];
Y=categorical(D.Y,c.labels,c.names);
opts=trainingOptions('adam','MaxEpochs',c.epochs,'MiniBatchSize',64,'InitialLearnRate',.001, ...
 'ValidationData',{X(:,:,:,D.val),Y(D.val)},'Shuffle','every-epoch','Plots','none','Verbose',false,'ExecutionEnvironment','cpu');
M.net=trainNetwork(X(:,:,:,D.train),Y(D.train),layers,opts);
p=classify(M.net,X(:,:,:,D.test),'ExecutionEnvironment','cpu');actual=Y(D.test);
M.accuracy=mean(p==actual);M.macroF1=emg.macroF1(double(actual),double(p),1:numel(c.labels));
fig=figure('Visible','off');confusionchart(actual,p);title('Temporal CNN: held-out subjects');saveas(fig,fullfile(c.out,'07_cnn_confusion.png'));close(fig);
fprintf('CNN accuracy %.3f | macro F1 %.3f\n',M.accuracy,M.macroF1);
end
