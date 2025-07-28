
%% Evaluate the Solution
% Plot the training loss history.
figure
plot(lossHistory)
set(gca, 'YScale', 'log')
xlabel('Epochs')
ylabel('Loss (log scale)')
title('Loss History')

%% Simulate the System with the Learned Controller
% Define an initial state and simulate the differential drive robot's
% trajectory using the optimal controller derived from the trained PINN.

x_init = [0.5; 0.5; pi]; % Initial state [x, y, theta]
tspan = linspace(0, 5, 5001);
dt = tspan(2) - tspan(1);

states = zeros(length(tspan), 3);
controls = zeros(length(tspan)-1, 2);
states(1, :) = x_init';
x_current = dlarray(x_init);

tic;
% Simulation loop using Euler integration
for i = 1:length(tspan)-1
    t_current = dlarray(tspan(i));
    
    % Get optimal control input from the network
    u_opt = optimalControl(net, t_current, x_current, r);
    controls(i, :) = extractdata(u_opt)';
    
    % System dynamics
    x_dot = [u_opt(1) * cos(x_current(3)); ...
             u_opt(1) * sin(x_current(3)); ...
             u_opt(2)];
    
    % Update state
    x_current = x_current + x_dot * dt;
    states(i+1, :) = extractdata(x_current)';
end
computationTime = toc;
fprintf('Trajectory computation time: %.4f seconds\n', computationTime);

%% Plot Trajectories
% Plot the state and control trajectories over time.
figure('Position', [100, 100, 800, 600])

% Plot states
subplot(2, 1, 1)
plot(tspan, states(:,1), 'LineWidth', 2)
hold on
plot(tspan, states(:,2), 'LineWidth', 2)
plot(tspan, states(:,3), 'LineWidth', 2)
hold off
title('System States with Time')
xlabel('Time (s)')
ylabel('State Value')
legend('x', 'y', '\theta')
grid on

% Plot controls
subplot(2, 1, 2)
plot(tspan(1:end-1), controls(:,1), 'LineWidth', 2)
hold on
plot(tspan(1:end-1), controls(:,2), 'LineWidth', 2)
hold off
title('Control Inputs with Time')
xlabel('Time (s)')
ylabel('Control Value')
legend('Velocity (v)', 'Yaw Rate (\omega)')
grid on

% Plot XY trajectory
figure
plot(states(:,1), states(:,2), 'LineWidth', 2)
title('XY Trajectory')
xlabel('X')
ylabel('Y')
axis equal
grid on

%% Supporting Functions
% Model Loss Function
function [loss, gradients] = modelLoss(net, T, X, Y, Th, T0, X0, Y0, Th0, U0, q, r, m)
    % PDE Residual Loss
    % Concatenate inputs for the forward pass
    % XT_coll = [T; X; Y; Th];
    XT_coll = cat(1, T, X, Y, Th);
    % size(XT_coll)
    V = forward(net, XT_coll);
    
    % Calculate first-order derivatives using dlgradient
    gradientsV = dlgradient(sum(V, 'all'), {T, X, Y, Th}, 'EnableHigherDerivatives', true);
    
    [V_T, V_X, V_Y, V_Th] = gradientsV{:};

    % HJB PDE components
    term_qx = q * (X.^2 + Y.^2 + Th.^2);
    g_gradV_term1 = V_X.*cos(Th) + V_Y.*sin(Th);
    g_gradV_term2 = V_Th;
    hamiltonian_term = (1/(4*r)) * (g_gradV_term1.^2 + g_gradV_term2.^2);
    
    % PDE residual
    f = -V_T - term_qx + hamiltonian_term;
    mseF = l2loss(f, zeros(size(f), 'like', f));
    
    % Terminal Condition Loss
    % XT0_bnd = [T0; X0; Y0; Th0];
    XT0_bnd = cat(1, T0, X0, Y0, Th0);
    V0_pred = forward(net, XT0_bnd);
    mseU = l2loss(V0_pred, U0);
    
    % Total Loss
    loss = mseF + mseU
    
    % Calculate gradients with respect to learnable parameters
    gradients = dlgradient(loss, net.Learnables)
end

% Optimal Control Function
function u = optimalControl(net, t, state, r)
    % state = [x, y, theta]'
    x = state(1);
    y = state(2);
    th = state(3);
    
    % Get gradients of V with respect to spatial states
    [V_X, V_Y, V_Th] = dlfeval(@stateGradients, net, t, x, y, th);
    
    % Optimal control law u = -0.5 * R^-1 * g(x)' * grad_V
    % R = r*I, so R^-1 = (1/r)*I
    % g(x)'*grad_V = [cos(th) sin(th) 0; 0 0 1] * [V_X; V_Y; V_Th]
    u1 = -0.5 * (1/r) * (V_X*cos(th) + V_Y*sin(th)); % Velocity v
    u2 = -0.5 * (1/r) * V_Th;                         % Yaw rate omega
    
    % Input saturation
    u_sat = 5;
    u1_sat = tanh(u1 / u_sat) * u_sat;
    u2_sat = tanh(u2 / u_sat) * u_sat;

    u = [u1_sat; u2_sat];
% end    u = [u1_sat; u2_sat];
end

function [V_X, V_Y, V_Th] = stateGradients(net, t, x, y, th)
    input = cat(1, t, x, y, th);
    input = dlarray(input, 'CB');
    V = forward(net, input);
    gradientsV = dlgradient(sum(V, 'all'), {x, y, th});
    [V_X, V_Y, V_Th] = gradientsV{:};
end