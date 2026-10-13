%% PID simulation of a DC gear motor + arm joint (PWM -> angle)
clear; clc; close all;

%% 1. PLANT: theta(s)/U(s) = K / (s*(tau*s + 1))
RPM_MAX = 100;                    % datasheet speed at full PWM
K   = RPM_MAX * 360 / 60 / 255;   % DC gain, (deg/s) per PWM unit
tau = 0.1;                        % time constant, s (guess)

%% 2. PID GAINS (change these!)
Kp = 10.0;
Ki = 0.0;
Kd = 0.5;

%% 3. SETTINGS
target  = 45;      % deg
dt      = 0.01;    % s
T_end   = 3;       % s
PWM_MAX = 255;
I_MAX   = 255;     % anti-windup limit on the Ki*integral term

%% 4. SIMULATION LOOP (discrete, with saturation)
n = round(T_end / dt);
t     = (0:n-1)' * dt;
angle = zeros(n,1);
speed = zeros(n,1);
pwm   = zeros(n,1);
integral   = 0;
last_error = target;   % avoids a fake derivative spike at step 1

for k = 1:n-1
    error = target - angle(k);

    integral = integral + error * dt;
    if Ki ~= 0
        integral = max(min(integral, I_MAX/Ki), -I_MAX/Ki);   % clamp
    end

    derivative = (error - last_error) / dt;
    last_error = error;

    u = Kp*error + Ki*integral + Kd*derivative;
    u = max(min(u, PWM_MAX), -PWM_MAX);        % saturation
    pwm(k) = u;

    speed(k+1) = speed(k) + (dt/tau) * (K*u - speed(k));   % first-order motor
    angle(k+1) = angle(k) + speed(k) * dt;                 % integrator
end

%% 5. METRICS
overshoot = max(0, (max(angle) - target) / target * 100);
t10 = t(find(angle >= 0.1*target, 1));
t90 = t(find(angle >= 0.9*target, 1));
outside = find(abs(angle - target) > 0.02*target);
if isempty(outside), settling = 0; else, settling = t(min(outside(end)+1, n)); end

fprintf('K = %.2f (deg/s)/PWM, tau = %.2f s\n', K, tau);
fprintf('Rise time (10-90%%): %.3f s\n', t90 - t10);
fprintf('Overshoot:          %.1f %%\n', overshoot);
fprintf('Settling time (2%%): %.3f s\n', settling);
fprintf('Steady-state error: %.3f deg\n', target - angle(end));

%% 6. PLOTS
figure;
subplot(2,1,1);
plot(t, angle, 'LineWidth', 1.5); hold on;
yline(target, 'r--', 'target');
ylabel('Angle (deg)'); grid on;
title(sprintf('Kp=%g, Ki=%g, Kd=%g', Kp, Ki, Kd));
subplot(2,1,2);
plot(t(1:end-1), pwm(1:end-1), 'g', 'LineWidth', 1.5);
ylabel('PWM'); xlabel('Time (s)'); grid on;
saveas(gcf, 'step_response_matlab.png');

%% 7. OPTIONAL: Control System Toolbox check (linear model, no saturation)
% Uncomment if you have the toolbox. It should match the loop above
% when the PWM does not saturate.
%
% G  = tf(K, [tau 1 0]);              % K / (tau*s^2 + s)
% C  = pid(Kp, Ki, Kd);
% T  = feedback(C*G, 1);              % closed loop
% figure; step(target*T, T_end); grid on;
% stepinfo(T)                         % rise time, overshoot, settling time
%
% [Cauto, info] = pidtune(G, 'PID');  % MATLAB suggests gains
% disp(Cauto)