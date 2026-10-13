function build_simulink_model
% Generates an editable Simulink closed-loop arm model programmatically.
% Requires Simulink. The plant is an illustrative single-DOF arm, NOT a
% physically identified robotic actuator.
c=config('fast');model='EMG_Arm_PID_Demo';
if bdIsLoaded(model),close_system(model,0);end
if isfile(fullfile(c.root,'models',[model '.slx'])),delete(fullfile(c.root,'models',[model '.slx']));end
new_system(model);open_system(model);
% Input reference from workspace is explicitly intended to be replaced by
% EMG classifier gesture-to-target output during hardware integration.
t=(0:.01:14)';ref=zeros(size(t));ref(t>=2&t<5)=pi/5;ref(t>=5&t<8)=-pi/6;ref(t>=8&t<11)=pi/8;
assignin('base','emg_ref_signal',timeseries(ref,t));
add_block('simulink/Sources/From Workspace',[model '/Gesture mapped angle'],'Position',[35 96 165 134]);
set_param([model '/Gesture mapped angle'],'VariableName','emg_ref_signal');
add_block('simulink/Math Operations/Sum',[model '/Error'],'Inputs','+-','Position',[215 100 240 130]);
add_block('simulink/Math Operations/Gain',[model '/Kp'],'Gain','2','Position',[295 35 355 70]);
add_block('simulink/Continuous/Integrator',[model '/Integral error'],'Position',[295 165 330 200]);
add_block('simulink/Math Operations/Gain',[model '/Ki'],'Gain','0.5','Position',[365 165 425 200]);
add_block('simulink/Math Operations/Sum',[model '/PI torque'],'Inputs','++','Position',[470 98 495 128]);
add_block('simulink/Continuous/Transfer Fcn',[model '/Arm plant'],'Numerator','[1]','Denominator','[0.08 0.55 0]','Position',[565 95 695 130]);
add_block('simulink/Signal Routing/Mux',[model '/Reference and actual'],'Inputs','2','Position',[750 45 755 118]);
add_block('simulink/Sinks/Scope',[model '/Scope'],'Position',[800 67 830 101]);
add_block('simulink/Sinks/To Workspace',[model '/theta_output'],'Position',[755 155 865 185]);
set_param([model '/theta_output'],'VariableName','theta_sim','SaveFormat','Timeseries');
add_line(model,'Gesture mapped angle/1','Error/1','autorouting','on');
add_line(model,'Error/1','Kp/1','autorouting','on');
add_line(model,'Error/1','Integral error/1','autorouting','on');
add_line(model,'Kp/1','PI torque/1','autorouting','on');
add_line(model,'Integral error/1','Ki/1','autorouting','on');
add_line(model,'Ki/1','PI torque/2','autorouting','on');
add_line(model,'PI torque/1','Arm plant/1','autorouting','on');
add_line(model,'Arm plant/1','Error/2','autorouting','on');
add_line(model,'Gesture mapped angle/1','Reference and actual/1','autorouting','on');
add_line(model,'Arm plant/1','Reference and actual/2','autorouting','on');
add_line(model,'Reference and actual/1','Scope/1','autorouting','on');
add_line(model,'Arm plant/1','theta_output/1','autorouting','on');
set_param(model,'StopTime','14','Solver','ode45','ReturnWorkspaceOutputs','on');
save_system(model,fullfile(c.root,'models',[model '.slx']));
try
 simOut=sim(model);simtheta=simOut.get('theta_sim');
 f=figure('Visible','off');plot(t,ref,'--',simtheta.Time,simtheta.Data,'LineWidth',1.3);
 legend('Desired joint position','Simulink actual position');xlabel('Time (s)');ylabel('Joint angle (rad)');
 title('Simulink PI closed-loop tracking');grid on;
 exportgraphics(f,fullfile(c.out,'11_simulink_tracking.png'));close(f);
catch ME
 warning('Model built and saved, but simulation failed: %s',ME.message);
end
close_system(model,0);disp('Saved models/EMG_Arm_PID_Demo.slx');
end
