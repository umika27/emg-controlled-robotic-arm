function plots(D,c)
set(0,'DefaultFigureVisible','off');cleanup=onCleanup(@()set(0,'DefaultFigureVisible','on'));
a=D.demo;n=min(size(a.x,1),1200);t=(0:n-1)/c.fs;
f=figure; tiledlayout(3,1);nexttile;plot(t,a.x(1:n,1));ylabel('Dataset amplitude');title('EMG channel 1: raw');grid on;
nexttile;plot(t,a.filtered(1:n,1));ylabel('Dataset amplitude');title('5–40 Hz filtered, offline');grid on;
nexttile;stairs(t,a.y(1:n));xlabel('Approximate time (s)');ylabel('Gesture label');grid on;
exportgraphics(f,fullfile(c.out,'01_raw_filtered.png'));close(f);
f=figure;histogram(D.Y,'BinMethod','integers');title('Gesture-window distribution');xlabel('Class');ylabel('Windows');grid on;
exportgraphics(f,fullfile(c.out,'02_classes.png'));close(f);
f=figure;hold on;for lab=c.labels
 vals=D.F(D.Y==lab,1);if ~isempty(vals), histogram(vals,35,'DisplayStyle','stairs','Normalization','probability');end
end;hold off;legend(c.names);xlabel('Channel 1 RMS');ylabel('Relative count');title('RMS feature distribution');grid on;
exportgraphics(f,fullfile(c.out,'03_rms_distributions.png'));close(f);
[b,a]=butter(4,c.passband/(c.fs/2),'bandpass');[h,w]=freqz(b,a,1024,c.fs);
f=figure;plot(w,20*log10(abs(h)+eps));xlabel('Frequency (Hz)');ylabel('Magnitude (dB)');title('Band-pass filter response');grid on;
exportgraphics(f,fullfile(c.out,'04_filter_response.png'));close(f);
f=figure;[pxx,w]=pwelch(aSafe(D.demo.x(:,1)),[],[],[],c.fs);[pyy,~]=pwelch(aSafe(D.demo.filtered(:,1)),[],[],[],c.fs);
plot(w,10*log10(pxx+eps),w,10*log10(pyy+eps));xlabel('Frequency (Hz)');ylabel('PSD (dB/unit Hz)');legend('Raw','Filtered');title('Power spectral density');grid on;
exportgraphics(f,fullfile(c.out,'05_psd.png'));close(f);
end
function a=aSafe(a),a=double(a);end
