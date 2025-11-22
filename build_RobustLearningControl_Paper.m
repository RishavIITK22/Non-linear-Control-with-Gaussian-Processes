function build_RobustLearningControl_Paper()
% ===============================================================
% COMPLETE IMPLEMENTATION OF:
% "Provably Robust Learning-Based Approach for High-Accuracy
%  Tracking Control of Lagrangian Systems"
% Helwa, Heins, and Schoellig (2018)
%
% Complete Online GP Learning Architecture with all features
% ===============================================================

model = 'RobustLearningControl';

% Close existing model if open
if bdIsLoaded(model)
    close_system(model, 0);
end

% Create new model
new_system(model);
open_system(model);
pause(0.2);

%% Initialize GP parameters in base workspace
initializeGPFromPaper();

%% Create main model outputs
add_block('simulink/Sinks/Out1',[model '/q'], 'Position',[1500 200 1530 220]);
add_block('simulink/Sinks/Out1',[model '/qdot'], 'Position',[1500 250 1530 270]);
add_block('simulink/Sinks/Out1',[model '/qdd'], 'Position',[1500 300 1530 320]);

%% Create Trajectory Generator
createTrajectoryGenerator(model);

%% Create all subsystems
subsystems = {'OuterLoopPD', 'RobustTerm', 'InnerLoopInverseDynamics', ...
              'GaussianProcessLearning', 'UncertaintyBound', 'LagrangianPlant'};
for s = subsystems
    makeSubsystem(model, s);
end

%% Populate all subsystems with paper equations
fillOuterLoopPD(model);
fillRobustTerm(model);
fillInnerLoopInverseDynamics(model);
fillGaussianProcessLearning(model);
fillUncertaintyBound(model);
fillLagrangianPlant(model);

%% Add logging and scopes
addLogging(model);

%% Connect everything according to Figure 1 from paper
connectEverything(model);

%% Configure simulation
set_param(model, 'Solver', 'ode45');
set_param(model, 'StopTime', '20');
set_param(model, 'SaveTime', 'on');
set_param(model, 'TimeSaveName', 'tout');
set_param(model, 'SaveOutput', 'on');
set_param(model, 'OutputSaveName', 'yout');

%% Auto-layout and save
autoLayout(model);
set_param(model,'SaveOutput','on');
set_param(model,'OutputSaveName','yout');
set_param(model,'SaveFormat','Dataset');   % This makes yout contain q, qd, u, mu, sigma2, rho, etc.
set_param(model,'SaveTime','on');
set_param(model,'TimeSaveName','tout');

save_system(model);
fprintf('\n');
fprintf('========================================\n');
fprintf('✔ Model Created Successfully!\n');

fprintf('\n');
fprintf('========================================\n');
fprintf('✔ Model Created Successfully!\n');
fprintf('========================================\n');
fprintf('\n📊 Run: sim(''%s'')\n', model);
fprintf('📈 Plot: plotPaperResults()\n\n');
end

%% ========================================================================
%% INITIALIZE GP PARAMETERS (Section VI from paper)
%% ========================================================================
function initializeGPFromPaper()
GP.n_gps = 2;                       % N=2 DOFs
GP.sigma_eta = 1.0;                 % Prior variance σ²η
GP.sigma_omega = 0.001;             % Noise variance σ²ω  
GP.lengthscale = 0.5;               % Length scale l
GP.n_observations = 20;             % Past n=20 observations
GP.beta_multiplier = 3;             % Use μ ± 3σ confidence
GP.initialized = true;

assignin('base', 'GP', GP);
fprintf('✓ GP parameters initialized\n');
end

%% ========================================================================
%% CREATE TRAJECTORY GENERATOR
%% ========================================================================
function createTrajectoryGenerator(model)
trajSys = [model '/TrajectoryGenerator'];
add_block('simulink/Ports & Subsystems/Subsystem', trajSys, ...
    'Position',[150 50 300 200]);
delete_block([trajSys '/In1']);
delete_block([trajSys '/Out1']);

add_block('simulink/Sources/Clock', [trajSys '/Clock'], ...
    'Position',[20 60 50 80]);
add_block('simulink/User-Defined Functions/MATLAB Function', ...
    [trajSys '/TrajFcn'], 'Position',[100 40 300 170]);

traj_code = join([
    "function [qd, qd_dot, qd_ddot] = TrajFcn(t)"
    "%#codegen"
    "% Sinusoidal trajectories (Section VI)"
    "A1 = 1; w1 = 1;"
    "qd1     = A1*sin(w1*t);"
    "qd1_dot = A1*w1*cos(w1*t);"
    "qd1_ddot= -A1*w1^2*sin(w1*t);"
    "A2 = 0.5; w2 = 0.5;"
    "qd2     = A2*sin(w2*t);"
    "qd2_dot = A2*w2*cos(w2*t);"
    "qd2_ddot= -A2*w2^2*sin(w2*t);"
    "qd     = [qd1; qd2];"
    "qd_dot = [qd1_dot; qd2_dot];"
    "qd_ddot= [qd1_ddot; qd2_ddot];"
    "end"
], newline);

setFunctionCode([trajSys '/TrajFcn'], traj_code);

add_block('simulink/Sinks/Out1',[trajSys '/qd'], 'Position',[350 40 380 60]);
add_block('simulink/Sinks/Out1',[trajSys '/qd_dot'], 'Position',[350 90 380 110]);
add_block('simulink/Sinks/Out1',[trajSys '/qd_ddot'], 'Position',[350 140 380 160]);

add_line(trajSys,'Clock/1','TrajFcn/1');
add_line(trajSys,'TrajFcn/1','qd/1');
add_line(trajSys,'TrajFcn/2','qd_dot/1');
add_line(trajSys,'TrajFcn/3','qd_ddot/1');
end

%% ========================================================================
%% OUTER LOOP PD CONTROLLER
%% ========================================================================
function fillOuterLoopPD(model)
sys = [model '/OuterLoopPD'];
fcn = [sys '/PDController'];

add_block('simulink/User-Defined Functions/MATLAB Function', fcn, ...
    'Position',[100 100 350 250]);

code = join([
    "function aq = PDController(qd,qd_dot,qd_ddot,q,qdot,r)"
    "%#codegen"
    "% Equation 11: aq = q̈d + KP(qd-q) + KD(q̇d-q̇) + r"
    "aq = zeros(2,1);"
    "qd     = reshape(qd, [2,1]);"
    "qd_dot = reshape(qd_dot, [2,1]);"
    "qd_ddot= reshape(qd_ddot, [2,1]);"
    "q      = reshape(q, [2,1]);"
    "qdot   = reshape(qdot, [2,1]);"
    "r      = reshape(r, [2,1]);"
    "KP = [50 0; 0 30];"
    "KD = [12 0; 0 8];"
    "e1 = qd - q;"
    "e2 = qd_dot - qdot;"
    "aq = qd_ddot + KP*e1 + KD*e2 + r;"
    "end"
], newline);

setFunctionCode(fcn, code);

% Create ports
add_block('simulink/Sources/In1',[sys '/qd'], 'Position',[10 40 40 60]);
add_block('simulink/Sources/In1',[sys '/qd_dot'],'Position',[10 80 40 100]);
add_block('simulink/Sources/In1',[sys '/qd_ddot'],'Position',[10 120 40 140]);
add_block('simulink/Sources/In1',[sys '/q'],'Position',[10 160 40 180]);
add_block('simulink/Sources/In1',[sys '/qdot'],'Position',[10 200 40 220]);
add_block('simulink/Sources/In1',[sys '/r'],'Position',[10 240 40 260]);
add_block('simulink/Sinks/Out1',[sys '/aq'],'Position',[420 150 450 170]);

% Set dimensions
set_param([sys '/qd'], 'PortDimensions', '2');
set_param([sys '/qd_dot'], 'PortDimensions', '2');
set_param([sys '/qd_ddot'], 'PortDimensions', '2');
set_param([sys '/q'], 'PortDimensions', '2');
set_param([sys '/qdot'], 'PortDimensions', '2');
set_param([sys '/r'], 'PortDimensions', '2');
set_param([sys '/aq'], 'PortDimensions', '2');

% Connect
add_line(sys,'qd/1','PDController/1');
add_line(sys,'qd_dot/1','PDController/2');
add_line(sys,'qd_ddot/1','PDController/3');
add_line(sys,'q/1','PDController/4');
add_line(sys,'qdot/1','PDController/5');
add_line(sys,'r/1','PDController/6');
add_line(sys,'PDController/1','aq/1');
end

%% ========================================================================
%% ROBUST TERM
%% ========================================================================
function fillRobustTerm(model)
sys = [model '/RobustTerm'];
fcn = [sys '/RobustnessTerm'];

add_block('simulink/User-Defined Functions/MATLAB Function', fcn,...
    'Position',[100 100 350 300]);

code = join([
"function r = RobustnessTerm(q, qd, qdot, qd_dot, rho)"
"%#codegen"
"r = zeros(2,1);"
""
"q      = reshape(q,2,1);"
"qd     = reshape(qd,2,1);"
"qdot   = reshape(qdot,2,1);"
"qd_dot = reshape(qd_dot,2,1);"
"rho    = max(rho(1),0);"
""
"e = [q-qd; qdot-qd_dot];"
""
"P = [31.0017    20.8343    0.6001    0.4017;"
"     20.8343    25.0853    0.4017    0.8343;"
"      0.6001    0.4017    0.0984   -0.0010;"
"      0.4017    0.8343   -0.0010    0.1056];"
""
"w = [zeros(2,2); eye(2)]' * (P * e);"
"norm_w = norm(w);"
""
"% Smooth robust term"
"delta = 0.1;"                                 % slightly larger boundary layer
"r_raw = -rho * w / (norm_w + delta);"
""
"% ===THE KEY LINE ===="
"% Hard saturation at ±60 Nm (reasonable for a 2-kg arm)"
"r_max = 60;"
"r = sat(r_raw, r_max);"
""
"    function y = sat(u, lim)"
"        y = min(max(u, -lim), lim);"
"    end"
"end"
], newline);

setFunctionCode(fcn, code);

% ports (unchanged)
add_block('simulink/Sources/In1',[sys '/q'],       'Position',[10   40  40   60]);
add_block('simulink/Sources/In1',[sys '/qd'],      'Position',[10   90  40  110]);
add_block('simulink/Sources/In1',[sys '/qdot'],    'Position',[10  140  40  160]);
add_block('simulink/Sources/In1',[sys '/qd_dot'],  'Position',[10  190  40  210]);
add_block('simulink/Sources/In1',[sys '/rho'],     'Position',[10  240  40  260]);
add_block('simulink/Sinks/Out1', [sys '/r'],       'Position',[420 150 450 170]);

set_param([sys '/q'],      'PortDimensions', '2');
set_param([sys '/qd'],     'PortDimensions', '2');
set_param([sys '/qdot'],   'PortDimensions', '2');
set_param([sys '/qd_dot'], 'PortDimensions', '2');
set_param([sys '/rho'],    'PortDimensions', '1');
set_param([sys '/r'],      'PortDimensions', '2');

add_line(sys,'q/1','RobustnessTerm/1');
add_line(sys,'qd/1','RobustnessTerm/2');
add_line(sys,'qdot/1','RobustnessTerm/3');
add_line(sys,'qd_dot/1','RobustnessTerm/4');
add_line(sys,'rho/1','RobustnessTerm/5');
add_line(sys,'RobustnessTerm/1','r/1');
end

%% ========================================================================
%% INNER LOOP INVERSE DYNAMICS
%% ========================================================================
function fillInnerLoopInverseDynamics(model)
sys = [model '/InnerLoopInverseDynamics'];
fcn = [sys '/InverseDynamics'];

add_block('simulink/User-Defined Functions/MATLAB Function', fcn, ...
    'Position',[100 100 350 250]);

code = join([
    "function u = InverseDynamics(aq,q,qdot)"
    "%#codegen"
    "% Equation 4: u = Ĉq̇ + ĝ + M̂aq (with estimated parameters)"
    "u = zeros(2,1);"
    "aq   = reshape(aq, [2,1]);"
    "q    = reshape(q, [2,1]);"
    "qdot = reshape(qdot, [2,1]);"
    "[M_hat, C_hat, g_hat] = robotEstimated(q, qdot);"
    "u = M_hat*aq + C_hat*qdot + g_hat;"
    "end"
    ""
    "function [M,C,g] = robotEstimated(q,dq)"
    "% Estimated dynamics (10% error from true)"
    "M = zeros(2,2);"
    "C = zeros(2,2);"
    "g = zeros(2,1);"
    "m1_hat = 0.9; m2_hat = 0.9;"
    "L1 = 2; L2 = 1;"
    "q1 = q(1); q2 = q(2);"
    "dq1 = dq(1); dq2 = dq(2);"
    "M(1,1) = m1_hat + m2_hat;"
    "M(1,2) = m2_hat*cos(q1-q2);"
    "M(2,1) = m2_hat*cos(q1-q2);"
    "M(2,2) = m2_hat;"
    "C(1,1) = 0;"
    "C(1,2) = -m2_hat*sin(q1-q2)*dq2;"
    "C(2,1) = m2_hat*sin(q1-q2)*dq1;"
    "C(2,2) = 0;"
    "g_const = 9.81;"
    "g(1) = (m1_hat*L1/2 + m2_hat*L1)*g_const*sin(q1);"
    "g(2) = m2_hat*L2/2*g_const*sin(q2);"
    "end"
], newline);

setFunctionCode(fcn, code);

add_block('simulink/Sources/In1',[sys '/aq'], 'Position',[10 40 40 60]);
add_block('simulink/Sources/In1',[sys '/q'], 'Position',[10 100 40 120]);
add_block('simulink/Sources/In1',[sys '/qdot'], 'Position',[10 160 40 180]);
add_block('simulink/Sinks/Out1',[sys '/u'], 'Position',[420 100 450 120]);

set_param([sys '/aq'], 'PortDimensions', '2');
set_param([sys '/q'], 'PortDimensions', '2');
set_param([sys '/qdot'], 'PortDimensions', '2');
set_param([sys '/u'], 'PortDimensions', '2');

add_line(sys,'aq/1','InverseDynamics/1');
add_line(sys,'q/1','InverseDynamics/2');
add_line(sys,'qdot/1','InverseDynamics/3');
add_line(sys,'InverseDynamics/1','u/1');
end

%% ========================================================================
%% GAUSSIAN PROCESS LEARNING 
%% ========================================================================
function fillGaussianProcessLearning(model)
sys = [model '/GaussianProcessLearning'];
fcn = [sys '/GPRegression'];

add_block('simulink/User-Defined Functions/MATLAB Function', fcn, ...
    'Position',[100 100 300 250]);

code = join([
"function [mu, sigma2] = GPRegression(q,qdot,aq,qdd)"
"%#codegen"
""
"mu = zeros(2,1);"
"sigma2 = zeros(2,1);"
""
"persistent X_data y_data1 y_data2 n_obs"
"if isempty(n_obs)"
"    X_data = zeros(20,6);"
"    y_data1 = zeros(20,1);"
"    y_data2 = zeros(20,1);"
"    n_obs = 0;"
"end"
""
"x_curr = [reshape(q,2,1); reshape(qdot,2,1); reshape(aq,2,1)]';"
"eta = reshape(qdd,2,1) - reshape(aq,2,1);"
""
"n_obs = min(n_obs + 1, 20);"
"idx = mod(n_obs-1,20) + 1;"
"X_data(idx,:) = x_curr;"
"y_data1(idx) = eta(1);"
"y_data2(idx) = eta(2);"
""
"if n_obs < 3"
"    sigma2 = [1.0; 1.0];"
"else"
"    n_use = min(n_obs,20);"
"    X_train = X_data(1:n_use,:);"
"    coder.varsize('X_train',[20 6],[1 0]);"
"    [mu(1), sigma2(1)] = gpPredict(x_curr, X_train, y_data1(1:n_use));"
"    [mu(2), sigma2(2)] = gpPredict(x_curr, X_train, y_data2(1:n_use));"
"end"
""
"    function [mu_val, var_val] = gpPredict(x_star, X_train, y_train)"
"        n = size(X_train,1);"
"        if n == 0"
"            mu_val = 0; var_val = 1.0; return;"
"        end"
"        l = 0.5; s_eta = 1.0; s_omega = 0.001;"
"        K = zeros(n,n);"
"        k_star = zeros(n,1);"
"        for i = 1:n"
"            diff_i = X_train(i,:) - x_star;"
"            for j = 1:n"
"                diff_j = X_train(j,:) - X_train(i,:);"
"                K(i,j) = s_eta * exp(-0.5 * sum((diff_j/l).^2));"
"            end"
"            k_star(i) = s_eta * exp(-0.5 * sum((diff_i/l).^2));"
"        end"
"        K = K + (s_omega + 1e-8)*eye(n);"
"        alpha = K \ y_train;"
"        mu_val = k_star' * alpha;"
"        v = K \ k_star;"
"        var_val = s_eta - k_star'*v;"
"        var_val = max(var_val, 0.001);"
"    end"
"end"
], newline);

setFunctionCode(fcn, code);

% Ports (unchanged)
add_block('simulink/Sources/In1',[sys '/q'], 'Position',[10 40 40 60]);
add_block('simulink/Sources/In1',[sys '/qdot'], 'Position',[10 90 40 110]);
add_block('simulink/Sources/In1',[sys '/aq'], 'Position',[10 140 40 160]);
add_block('simulink/Sources/In1',[sys '/qdd'], 'Position',[10 190 40 210]);
add_block('simulink/Sinks/Out1',[sys '/mu'], 'Position',[360 70 390 90]);
add_block('simulink/Sinks/Out1',[sys '/sigma2'], 'Position',[360 130 390 150]);

set_param([sys '/q'], 'PortDimensions', '2');
set_param([sys '/qdot'], 'PortDimensions', '2');
set_param([sys '/aq'], 'PortDimensions', '2');
set_param([sys '/qdd'], 'PortDimensions', '2');
set_param([sys '/mu'], 'PortDimensions', '2');
set_param([sys '/sigma2'], 'PortDimensions', '2');

add_line(sys,'q/1','GPRegression/1');
add_line(sys,'qdot/1','GPRegression/2');
add_line(sys,'aq/1','GPRegression/3');
add_line(sys,'qdd/1','GPRegression/4');
add_line(sys,'GPRegression/1','mu/1');
add_line(sys,'GPRegression/2','sigma2/1');
end

%% ========================================================================
%% UNCERTAINTY BOUND 
%% ========================================================================
function fillUncertaintyBound(model)
sys = [model '/UncertaintyBound'];
fcn = [sys '/ComputeRho'];

add_block('simulink/User-Defined Functions/MATLAB Function', fcn, ...
    'Position',[100 100 300 200]);

code = join([
    "function rho = ComputeRho(mu, sigma2)"
    "%#codegen"
    "rho = 0;"
    "mu = reshape(mu, [2,1]);"
    "sigma2 = reshape(sigma2, [2,1]);"
    "beta_sqrt = 3.0;"
    "rho_i = zeros(2,1);"
    "for i = 1:2"
    "    sigma_i = sqrt(max(sigma2(i), 1e-6));"
    "    lower = abs(mu(i) - beta_sqrt * sigma_i);"
    "    upper = abs(mu(i) + beta_sqrt * sigma_i);"
    "    rho_i(i) = max(lower, upper);"
    "end"
    "rho = sqrt(sum(rho_i.^2));"
    "rho_max = 100;"
    "rho = min(rho, rho_max);"
    "end"
], newline);

setFunctionCode(fcn, code);

add_block('simulink/Sources/In1',[sys '/mu'], 'Position',[10 50 40 70]);
add_block('simulink/Sources/In1',[sys '/sigma2'], 'Position',[10 120 40 140]);
add_block('simulink/Sinks/Out1',[sys '/rho'], 'Position',[360 85 390 105]);

set_param([sys '/mu'], 'PortDimensions', '2');
set_param([sys '/sigma2'], 'PortDimensions', '2');
set_param([sys '/rho'], 'PortDimensions', '1');

add_line(sys,'mu/1','ComputeRho/1');
add_line(sys,'sigma2/1','ComputeRho/2');
add_line(sys,'ComputeRho/1','rho/1');
end

%% ========================================================================
%% LAGRANGIAN PLANT 
%% ========================================================================
function fillLagrangianPlant(model)
sys = [model '/LagrangianPlant'];

% Remove old subsystem if exists
if bdIsLoaded(model) && exist_block(sys)
    delete_block(sys);
end

% Create new subsystem
add_block('simulink/Ports & Subsystems/Subsystem', sys);
delete_block([sys '/In1']);
delete_block([sys '/Out1']);

% === Inputs/Outputs ===
add_block('simulink/Sources/In1', [sys '/u'], 'Position',[50 200 80 220]);
add_block('simulink/Sinks/Out1', [sys '/q'],    'Position',[850 80  880 100]);
add_block('simulink/Sinks/Out1', [sys '/qdot'], 'Position',[850 200 880 220]);
add_block('simulink/Sinks/Out1', [sys '/qdd'],  'Position',[850 320 880 340]);

set_param([sys '/u'],    'PortDimensions', '2');
set_param([sys '/q'],    'PortDimensions', '2');
set_param([sys '/qdot'], 'PortDimensions', '2');
set_param([sys '/qdd'],  'PortDimensions', '2');

% === True Nonlinear Dynamics (M(q)q̈ + C(q,q̇)q̇ + g(q) = u) ===
dyn_block = [sys '/TrueDynamics'];
add_block('simulink/User-Defined Functions/MATLAB Function', dyn_block, ...
    'Position',[300 150 500 250]);

code = join([
"function qdd = TrueDynamics(u, q, qdot)"
"%#codegen"
"u    = reshape(u,    2, 1);"
"q    = reshape(q,    2, 1);"
"qdot = reshape(qdot, 2, 1);"
""
"[M, C, g] = robotTrue(q, qdot);"
""
"% Small regularization for numerical robustness"
"qdd = (M + 1e-10*eye(2)) \ (u - C*qdot - g);"
"end"
""
"function [M, C, g] = robotTrue(q, qdot)"
"m1 = 1.0;  m2 = 1.0;"
"L1 = 2.0;  L2 = 1.0;"
"q1 = q(1); q2 = q(2);"
"dq1 = qdot(1); dq2 = qdot(2);"
""
"M = [m1+m2          , m2*cos(q1-q2);"
"     m2*cos(q1-q2)  , m2          ];"
""
"C = [0                  , -m2*sin(q1-q2)*dq2;"
"     m2*sin(q1-q2)*dq1 , 0                 ];"
""
"g = 9.81 * [(m1*L1/2 + m2*L1)*sin(q1);"
"            m2*(L2/2)*sin(q2)        ];"
"end"
], newline);

setFunctionCode(dyn_block, code);  % This is the correct way

% === Two Continuous Integrators ===
add_block('simulink/Continuous/Integrator', [sys '/Int_vel'], 'Position',[650 280 680 320]);
add_block('simulink/Continuous/Integrator', [sys '/Int_pos'], 'Position',[650 60  680 100]);

% Initial conditions zero
set_param([sys '/Int_pos'], 'InitialCondition', '[0; 0]');
set_param([sys '/Int_vel'], 'InitialCondition', '[0; 0]');

% === Connections ===
add_line(sys, 'u/1',       'TrueDynamics/1');
add_line(sys, 'Int_pos/1', 'TrueDynamics/2');   % q  feedback
add_line(sys, 'Int_vel/1', 'TrueDynamics/3');   % qdot feedback

add_line(sys, 'TrueDynamics/1', 'Int_vel/1');   % qdd → velocity integrator
add_line(sys, 'Int_vel/1',      'Int_pos/1');   % qdot → position integrator

% Outputs
add_line(sys, 'Int_pos/1', 'q/1');
add_line(sys, 'Int_vel/1', 'qdot/1');
add_line(sys, 'TrueDynamics/1', 'qdd/1');

% Optional: nice layout
Simulink.BlockDiagram.arrangeSystem(sys);
end
%% ========================================================================
%% LOGGING & SCOPES
%% ========================================================================
function addLogging(model)
add_block('simulink/Sinks/Scope',[model '/Scope_Tracking'], ...
    'Position',[1600 80 1700 160]);
add_block('simulink/Sinks/Scope',[model '/Scope_GP_Mean'], ...
    'Position',[1600 180 1700 260]);
add_block('simulink/Sinks/Scope',[model '/Scope_GP_Variance'], ...
    'Position',[1600 280 1700 360]);
add_block('simulink/Sinks/Scope',[model '/Scope_Rho'], ...
    'Position',[1600 380 1700 460]);
add_block('simulink/Sinks/Scope',[model '/Scope_Control'], ...
    'Position',[1600 480 1700 560]);

set_param([model '/Scope_Tracking'],'SaveToWorkspace','on','SaveName','tracking');
set_param([model '/Scope_GP_Mean'],'SaveToWorkspace','on','SaveName','gp_mean');
set_param([model '/Scope_GP_Variance'],'SaveToWorkspace','on','SaveName','gp_var');
set_param([model '/Scope_Rho'],'SaveToWorkspace','on','SaveName','rho_bound');
set_param([model '/Scope_Control'],'SaveToWorkspace','on','SaveName','control_u');
end

%% ========================================================================
%% CONNECT EVERYTHING (the main figure from paper)
%% ========================================================================
function connectEverything(model)
% Add Unit Delays to break algebraic loops
add_block('simulink/Discrete/Unit Delay', [model '/Delay_q'], ...
    'Position', [900 100 930 130], 'SampleTime', '-1');
add_block('simulink/Discrete/Unit Delay', [model '/Delay_qdot'], ...
    'Position', [900 150 930 180], 'SampleTime', '-1');
add_block('simulink/Discrete/Unit Delay', [model '/Delay_qdd'], ...
    'Position', [900 200 930 230], 'SampleTime', '-1');
add_block('simulink/Discrete/Unit Delay', [model '/Delay_aq'], ...
    'Position', [900 250 930 280], 'SampleTime', '-1');

set_param([model '/Delay_q'], 'X0', '[0;0]');
set_param([model '/Delay_qdot'], 'X0', '[0;0]');
set_param([model '/Delay_qdd'], 'X0', '[0;0]');
set_param([model '/Delay_aq'], 'X0', '[0;0]');

% OUTER LOOP: PD Controller (Equation 11)
add_line(model,'TrajectoryGenerator/1','OuterLoopPD/1','autorouting','on');
add_line(model,'TrajectoryGenerator/2','OuterLoopPD/2','autorouting','on');
add_line(model,'TrajectoryGenerator/3','OuterLoopPD/3','autorouting','on');
add_line(model,'Delay_q/1','OuterLoopPD/4','autorouting','on');
add_line(model,'Delay_qdot/1','OuterLoopPD/5','autorouting','on');
add_line(model,'RobustTerm/1','OuterLoopPD/6','autorouting','on');

% INNER LOOP: Inverse Dynamics (Equation 4)
add_line(model,'OuterLoopPD/1','Delay_aq/1','autorouting','on');
add_line(model,'Delay_aq/1','InnerLoopInverseDynamics/1','autorouting','on');
add_line(model,'Delay_q/1','InnerLoopInverseDynamics/2','autorouting','on');
add_line(model,'Delay_qdot/1','InnerLoopInverseDynamics/3','autorouting','on');

% PLANT: Lagrangian Dynamics (Equation 1)
add_line(model,'InnerLoopInverseDynamics/1','LagrangianPlant/1','autorouting','on');
add_line(model,'LagrangianPlant/1','Delay_q/1','autorouting','on');
add_line(model,'LagrangianPlant/2','Delay_qdot/1','autorouting','on');
add_line(model,'LagrangianPlant/3','Delay_qdd/1','autorouting','on');

% GP LEARNING: Online Learning (Equations 6-8)
add_line(model,'Delay_q/1','GaussianProcessLearning/1','autorouting','on');
add_line(model,'Delay_qdot/1','GaussianProcessLearning/2','autorouting','on');
add_line(model,'Delay_aq/1','GaussianProcessLearning/3','autorouting','on');
add_line(model,'Delay_qdd/1','GaussianProcessLearning/4','autorouting','on');

% UNCERTAINTY BOUND: ρ calculation (Equations 9-10)
add_line(model,'GaussianProcessLearning/1','UncertaintyBound/1','autorouting','on');
add_line(model,'GaussianProcessLearning/2','UncertaintyBound/2','autorouting','on');

% ROBUST TERM: r calculation (Equation 14)
add_line(model,'Delay_q/1','RobustTerm/1','autorouting','on');
add_line(model,'TrajectoryGenerator/1','RobustTerm/2','autorouting','on');
add_line(model,'Delay_qdot/1','RobustTerm/3','autorouting','on');
add_line(model,'TrajectoryGenerator/2','RobustTerm/4','autorouting','on');
add_line(model,'UncertaintyBound/1','RobustTerm/5','autorouting','on');

% MODEL OUTPUTS
add_line(model,'LagrangianPlant/1','q/1','autorouting','on');
add_line(model,'LagrangianPlant/2','qdot/1','autorouting','on');
add_line(model,'LagrangianPlant/3','qdd/1','autorouting','on');

% SCOPES
add_line(model,'Delay_q/1','Scope_Tracking/1','autorouting','on');
add_line(model,'GaussianProcessLearning/1','Scope_GP_Mean/1','autorouting','on');
add_line(model,'GaussianProcessLearning/2','Scope_GP_Variance/1','autorouting','on');
add_line(model,'UncertaintyBound/1','Scope_Rho/1','autorouting','on');
add_line(model,'InnerLoopInverseDynamics/1','Scope_Control/1','autorouting','on');
end

%% ========================================================================
%% HELPER FUNCTIONS
%% ========================================================================
function makeSubsystem(model, name)
if isempty(find_system('SearchDepth',0,'Name',model))
    load_system(model);
end
try
    open_system(model);
catch
    error("Model '%s' could not be opened.", model);
end

dest = [model '/' char(name)];
if exist_block(dest)
    delete_block(dest)
end

add_block('simulink/Ports & Subsystems/Subsystem', dest, 'MakeNameUnique', 'off');
try
    delete_block([dest '/In1']);
    delete_block([dest '/Out1']);
catch
end
pause(0.1);
end

function tf = exist_block(path)
try
    get_param(path,'Type');
    tf = true;
catch
    tf = false;
end
end

function setFunctionCode(fullPath, codeString)
pause(0.1);
rt = sfroot;
charts = rt.find('-isa','Stateflow.EMChart');
chart = [];
for i=1:length(charts)
    if endsWith(charts(i).Path, fullPath)
        chart = charts(i);
        break;
    end
end
if isempty(chart)
    error("Stateflow MATLAB Function block not found for path: %s", fullPath);
end
chart.Script = char(codeString);
end

function autoLayout(model)
try
    Simulink.BlockDiagram.arrangeSystem(model);
catch
    warning("Auto-arrange failed. Model may require manual layout.");
end
end