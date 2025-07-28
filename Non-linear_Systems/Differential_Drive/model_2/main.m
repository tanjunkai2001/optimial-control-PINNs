%% Solve Hamilton-Jacobi-Bellman PDE for Differential Drive Robot with PINNs
% This script shows how to train a physics informed neural network (PINN) 
% to solve the Hamilton-Jacobi-Bellman (HJB) partial differential equation
% associated with the optimal control of a differential drive robot.
%
% The script trains a multilayer perceptron neural network that takes time 
% and state (t, x, y, theta) as input and returns the value function V(t,x).
% The training process minimizes a loss function composed of the HJB PDE 
% residual and the terminal cost condition.

%% Problem Formulation
% *Dynamics*:
%   dx/dt = v*cos(theta)
%   dy/dt = v*sin(theta)
%   dtheta/dt = omega
%   Which can be written as: x_dot = f(x) + g(x)u, where u = [v, omega]'
%
% *Optimal Control Problem* (finite horizon with terminal cost):
%   min_u L = x(t_f)'*M*x(t_f) + integral from 0 to t_f of (x'*Q*x + u'*R*u) dt
%   where t_f = 1, Q=qI, R=rI, M=mI
%
% *Resulting HJB PDE to solve for the Value Function V(t,x)*:
%   -V_t - x'*Q*x + (1/4)*(grad_x V)'*g(x)*R^-1*g(x)'*(grad_x V) = 0
%
% *Boundary (Terminal) Condition*: 
%   V(t=1, x) = x'*M*x

%% Generate Training Data
% Training data consists of collocation points inside the domain (t,x,y,theta)
% to enforce the PDE and points on the boundary (t=1) to enforce the
% terminal condition.

% Define cost function parameters
q = 6;  % Q = q*I
r = 0.5;% R = r*I
m = 3;  % M = m*I

% Generate collocation points using a grid.
nPoints = 25; % Number of points in each dimension
t = linspace(0, 1, nPoints);
x = linspace(-1, 1, nPoints);
y = linspace(-1, 1, nPoints);
th = linspace(-pi, pi, nPoints);

[T, X, Y, Th] = ndgrid(t, x, y, th);

% Collocation points for PDE residual
dataT = T(:);
dataX = X(:);
dataY = Y(:);
dataTh = Th(:);


% Boundary points for terminal condition (at t=1)
idx = T == 1;
T0 = T(idx);
X0 = X(idx);
Y0 = Y(idx);
Th0 = Th(idx);
U0 = m * (X0.^2 + Y0.^2 + Th0.^2);

%% Define Neural Network Architecture
% Define a multilayer perceptron neural network architecture with 4 layers 
% (3 hidden) with 10 hidden neurons each. The input is 4-dimensional 
% (t, x, y, theta) and the output is 1-dimensional (V).

numLayers = 4;
numNeurons = 10;

layers = featureInputLayer(4, 'Name', 'input');

for i = 1:numLayers-1
    layers = [
        layers
        fullyConnectedLayer(numNeurons, 'Name', 'fc'+string(i))
        tanhLayer('Name', 'tanh'+string(i))]; % Using tanh as a smooth alternative to SiLU
end

layers = [
    layers
    fullyConnectedLayer(1, 'Name', 'output')];

net = dlnetwork(layers);

% Convert network to double for better accuracy with L-BFGS.
net = dlupdate(@double,net);

%% Specify Training Options
% Train for 5000 iterations using L-BFGS.
maxIterations = 5000;
gradientTolerance = 1e-5;
stepTolerance = 1e-5;
solverState = lbfgsState;

%% Train Neural Network
% Convert training data to dlarray objects with batch dimension 'B' and 
% channel dimension 'C'.
dataT = dlarray(dataT, "BC");
dataX = dlarray(dataX, "BC");
dataY = dlarray(dataY, "BC");
dataTh = dlarray(dataTh, "BC");

T0 = dlarray(T0, "CB");
X0 = dlarray(X0, "CB");
Y0 = dlarray(Y0, "CB");
Th0 = dlarray(Th0, "CB");
U0 = dlarray(U0, "CB");

% Accelerate the loss function using dlaccelerate.
accfun = dlaccelerate(@modelLoss);

% Create a function handle for the L-BFGS update.
lossFcn = @(net) dlfeval(accfun, net, dataT, dataX, dataY, dataTh, T0, X0, Y0, Th0, U0, q, r, m);

% Initialize the TrainingProgressMonitor object.
% monitor = trainingProgressMonitor(...
%     Metrics="TrainingLoss",...
%     Info=["Iteration" "GradientsNorm" "StepNorm"],...
%     XLabel="Iteration");

% Train the network.
iteration = 0;
lossHistory = [];
while iteration < maxIterations && ~monitor.Stop
    iteration = iteration + 1
    [net, solverState] = lbfgsupdate(net, lossFcn, solverState);
    
    lossVal = extractdata(solverState.Loss);
    lossHistory = [lossHistory; lossVal];

    updateInfo(monitor,...
        Iteration=iteration,...
        GradientsNorm=solverState.GradientsNorm,...
        StepNorm=solverState.StepNorm);
    
    recordMetrics(monitor,iteration,TrainingLoss=lossVal);
    monitor.Progress = 100 * iteration/maxIterations;
    
    if solverState.GradientsNorm < gradientTolerance ||...
            solverState.StepNorm < stepTolerance ||...
            solverState.LineSearchStatus == "failed"
        break
    end
end

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

x_init = [1.0; 0; pi]; % Initial state [x, y, theta]
tspan = linspace(0, 1, 101);
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
    XT_coll = [T; X; Y; Th];
    % XT_coll = cat(1, T, X, Y, Th);
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
    XT0_bnd = [T0; X0; Y0; Th0];
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
    [V_X, V_Y, V_Th] = dlgradient(forward(net, [t;x;y;th]), {x, y, th});
    
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
end