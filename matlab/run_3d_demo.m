function run_3d_demo
% Animated 3D stick-robot demo; uses MATLAB graphics (no Simscape required).
c=config('fast');path=fullfile(c.out,'control_trajectory.mat');
if ~isfile(path),run_control_demo;end
A=load(path);ang=A.theta;t=A.t;
% Shoulder yaw maps to the simulated one-DOF position; elbow is a scripted
% illustrative motion, NOT learned or controlled by EMG.
base=[0 0 .15];L1=.65;L2=.50;
f=figure('Name','EMG-Controlled Robot Arm — 3D Demonstration','Color','w','Position',[120 80 900 600]);
ax=axes(f);hold(ax,'on');grid(ax,'on');axis(ax,'equal');view(ax,35,23);
xlim(ax,[-1.3 1.3]);ylim(ax,[-1.3 1.3]);zlim(ax,[0 1.5]);xlabel('X (m)');ylabel('Y (m)');zlabel('Z (m)');
plot3(ax,[0 0],[0 0],[0 .15],'k-','LineWidth',7);
h1=plot3(ax,nan,nan,nan,'-o','LineWidth',9,'MarkerSize',12,'Color',[.09 .40 .75]);
h2=plot3(ax,nan,nan,nan,'-o','LineWidth',7,'MarkerSize',11,'Color',[.95 .43 .1]);
hTarget=plot3(ax,nan,nan,nan,'p','MarkerSize',11,'MarkerFaceColor','g');
ht=title(ax,'Simulated 3D Arm');
% Save representative stills and an animated GIF for review presentations.
gif=fullfile(c.out,'09_arm_animation.gif');idx=unique(round(linspace(1,numel(t),85)));
for j=1:numel(idx)
 k=idx(j);yaw=ang(k);pitch=pi/5+.12*sin(.5*t(k));elbow=pi/3;
 p1=base+L1*[cos(yaw)*cos(pitch),sin(yaw)*cos(pitch),sin(pitch)];
 p2=p1+L2*[cos(yaw)*cos(pitch+elbow),sin(yaw)*cos(pitch+elbow),sin(pitch+elbow)];
 set(h1,'XData',[base(1) p1(1)],'YData',[base(2) p1(2)],'ZData',[base(3) p1(3)]);
 set(h2,'XData',[p1(1) p2(1)],'YData',[p1(2) p2(2)],'ZData',[p1(3) p2(3)]);
 target=A.ref(k);q1=base+L1*[cos(target)*cos(pitch),sin(target)*cos(pitch),sin(pitch)];
 q2=q1+L2*[cos(target)*cos(pitch+elbow),sin(target)*cos(pitch+elbow),sin(pitch+elbow)];
 set(hTarget,'XData',q2(1),'YData',q2(2),'ZData',q2(3));
 set(ht,'String',sprintf('3D arm demo | t = %.1f s | target = %.1f° | actual = %.1f°',t(k),rad2deg(A.ref(k)),rad2deg(yaw)));
 drawnow;
 fr=getframe(f);[I,map]=rgb2ind(frame2im(fr),256);
 if j==1,imwrite(I,map,gif,'gif','LoopCount',inf,'DelayTime',.1);
 else,imwrite(I,map,gif,'gif','WriteMode','append','DelayTime',.1);end
end
exportgraphics(f,fullfile(c.out,'10_arm_3d_snapshot.png'));close(f);
fprintf('Saved 3D arm GIF: %s\n',gif);
end
