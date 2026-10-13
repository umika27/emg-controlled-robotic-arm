function run_control_demo
% Simulated reference from decoded gesture commands; no physical robot.
c=config('fast');dt=.01;t=(0:dt:14)';
% Four illustrative gestures: rest, flexion, extension, fist/grip.
gesture=zeros(size(t));gesture(t>=2&t<5)=3;gesture(t>=5&t<8)=4;gesture(t>=8&t<11)=2;gesture(t>=11)=1;
ref=zeros(size(t));ref(gesture==3)=pi/5;ref(gesture==4)=-pi/6;ref(gesture==2)=pi/8;
% Plant: J theta''+B theta'=torque, PI control. Semi-implicit Euler.
J=.08;B=.55;Kp=2;Ki=.5;theta=zeros(size(t));vel=zeros(size(t));u=zeros(size(t));intE=0;
for k=2:numel(t)
 e=ref(k-1)-theta(k-1);intE=max(-2,min(2,intE+e*dt));
 torque=max(-3,min(3,Kp*e+Ki*intE));u(k)=torque;
 vel(k)=vel(k-1)+dt*(torque-B*vel(k-1))/J;
 theta(k)=theta(k-1)+dt*vel(k);
end
f=figure('Visible','off');tiledlayout(3,1);
nexttile;plot(t,ref,'--',t,theta,'LineWidth',1.3);ylabel('Angle (rad)');legend('Desired','Actual');title('Simulated closed-loop joint control');grid on;
nexttile;plot(t,ref-theta);ylabel('Tracking error (rad)');grid on;
nexttile;stairs(t,gesture);xlabel('Time (s)');ylabel('Gesture class');grid on;
exportgraphics(f,fullfile(c.out,'08_control_tracking.png'));close(f);
save(fullfile(c.out,'control_trajectory.mat'),'t','ref','theta','u','gesture','J','B','Kp','Ki');
rmse=sqrt(mean((ref-theta).^2));fprintf('Illustrative closed-loop tracking RMSE: %.3f rad (simulation only)\n',rmse);
end
