%% *************************************************************************************************************
% Replication of Angelini G. and Fanelli L. (2019) framework
% Lakdawala (2019) Proxy-SVAR-IV Identification (Partial Identification)
% author: Luca Orlando, Luca Marchesi
%% *************************************************************************************************************
clc; clear; close all;
addpath functions;

% Global variables
global NLags TAll sig_CMD mS_opt n r g_shock M Sigma_u mDDr

%% 1. settings
BootstrapIterations = 200; 
NLags = 12;
options = optimoptions('fminunc','Display','off','MaxFunctionEvaluations',200000,...
    'OptimalityTolerance',1e-10,'StepTolerance',1e-10,'MaxIterations',200000);


%% 2. loading data 
load(fullfile('Data','all_data.mat'));

data_matrix   = Dataset1.all_data;   % T x N
all_dates_raw = Dataset1.all_dates;  % T x 1 (datenum)
all_labels    = string(Dataset1.all_data_labels);

% specification: [FFR, 1Y, log(CPI), log(IP)]

var_idx   = [3 4 1 2];
var_names = {'FFR','1Y','log(CPI)','log(IP)'};

% Dataset VAR (full sample)
DataSet = data_matrix(:, var_idx);

% Check: print labels 
fprintf('\nSelected series:\n');
for j = 1:numel(var_idx)
    fprintf('  %s  <- col %2d : %s\n', var_names{j}, var_idx(j), all_labels(var_idx(j)));
end
% --- Constructing the dataset VAR ---
DataSet = data_matrix(:, var_idx);

% --- Dimensions ---
n = size(DataSet,2);    
r = 2;                  % TP1 => 2 proxies
g_shock = 2;            % Target + Path
M = n + r;              % stacked dimension (u,z)

% Sanity check
fprintf('Loaded VAR dataset: n=%d variables, instruments r=%d, M=n+r=%d\n', n, r, M);
assert(n==4 && r==2 && g_shock==2, 'Unexpected dimensions: check var_idx or instruments.');
%% 3. filtering data
start_date = datenum('7/1/1979');
end_date   = datenum('12/1/2011');

start  = find(all_dates_raw == start_date, 1);
finish = find(all_dates_raw == end_date,   1);
assert(~isempty(start) && ~isempty(finish), 'Start/end date not found exactly in all_dates_raw.');

% VAR sample (macro)
VAR_dates  = all_dates_raw(start:finish);
AllDataSet = DataSet(start:finish, :);

% (effective sample after lags)
TAll = size(AllDataSet,1) - NLags;
assert(TAll > 0, 'TAll <= 0: check NLags or sample length.');
%% 4. LOAD INSTRUMENTS (TP1) + ALIGN END OF SAMPLE
load(fullfile('Instruments','TP1_instr.mat'));   % creates instr

proxies    = instr.Proxy;        % Tm x 2
proxydates = instr.ProxyDates;   % Tm x 1

% Keep only proxies that fall inside VAR sample (do NOT trim VAR)
keep = (proxydates >= VAR_dates(1)) & (proxydates <= VAR_dates(end));
proxies    = proxies(keep,:);
proxydates = proxydates(keep);

% Tail mismatch length: how many VAR obs at the end have no proxy coverage?
if VAR_dates(end) == proxydates(end)
    T_m_end = 0;
elseif VAR_dates(end) > proxydates(end)
    T_m_end = length(VAR_dates) - find(VAR_dates == proxydates(end), 1);
else
    error('End date for instrument sample cannot be after VAR end date');
end

fprintf('VAR sample:   %s to %s | T=%d\n', datestr(VAR_dates(1)), datestr(VAR_dates(end)), size(AllDataSet,1));
fprintf('Proxy sample: %s to %s | Tm=%d | T_m_end=%d\n', ...
    datestr(proxydates(1)), datestr(proxydates(end)), size(proxies,1), T_m_end);
%% 6. REDUCED-FORM VAR ESTIMATION (varm) -> residuals U and dates rf_dates
Mdl = varm(n, NLags);
Mdl.Constant = NaN(n,1);

[EstMdl, ~, ~, U_full] = estimate(Mdl, AllDataSet);

T = size(AllDataSet,1);

% --- Make U and rf_dates consistent ---
if size(U_full,1) == T
    % Case A: U_full is T x n with initial NaNs
    nan_rows = any(isnan(U_full), 2);
    U = U_full(~nan_rows, :);
    rf_dates = VAR_dates(~nan_rows);

elseif size(U_full,1) == (T - NLags)
    % Case B: U_full is already (T-NLags) x n (no NaNs)
    U = U_full;
    rf_dates = VAR_dates(NLags+1:end);

else
    error('Unexpected residual length: size(U_full,1)=%d (T=%d, NLags=%d)', size(U_full,1), T, NLags);
end

T_eff = size(U,1);

Sigma_u_VAR = EstMdl.Covariance;
Sigma_u_check = (U' * U) / T_eff;

fprintf('\nReduced-form VAR(%d) estimated via varm.\n', NLags);
fprintf('Residual sample: %s to %s | T_eff=%d\n', datestr(rf_dates(1)), datestr(rf_dates(end)), T_eff);
fprintf('||Sigma_u_VAR - Sigma_u_check||_F = %.3e\n', norm(Sigma_u_VAR - Sigma_u_check, 'fro'));
%% 7. ALIGN PROXIES TO RESIDUALS
[tf, loc] = ismember(rf_dates, proxydates);
idx_u = find(tf);

U_m     = U(idx_u, :);
Z_m     = proxies(loc(tf), :);
dates_m = rf_dates(idx_u);

fprintf('\nAlignment:\n');
fprintf('Matched observations (U_m, Z_m): %d\n', size(U_m,1));
fprintf('First match: %s | Last match: %s\n', datestr(dates_m(1)), datestr(dates_m(end)));

% With your data, this should end at 01-Dec-2011 and start at 01-Jan-1991
assert(dates_m(end) == proxydates(end), 'End date mismatch after alignment.');
%% PARTIAL IDENTIFICATION using (u,z) moments only

Tm = size(U_m,1);
W  = [U_m, Z_m];                 % Tm x M
Sigma_Eta_Sample = (W' * W) / Tm; % M x M 

% Partition (u = macro residuals, v = proxies)
Sigma_u  = Sigma_Eta_Sample(1:n, 1:n);             % n x n
Sigma_uv = Sigma_Eta_Sample(1:n, n+1:end);         % n x r
Sigma_vu = Sigma_Eta_Sample(n+1:end, 1:n);         % r x n
Sigma_v  = Sigma_Eta_Sample(n+1:end, n+1:end);     % r x r

%% --- Useful matrices ---
D_M  = DuplicationMatrixFunction(M);
mDD = (D_M' * D_M) \ D_M';

D_r   = DuplicationMatrixFunction(r);
mDDr  = (D_r' * D_r) \ D_r';

D_n   = DuplicationMatrixFunction(n);
mDDn  = (D_n' * D_n) \ D_n';

K_MM  = CommutationMatrixFunction(M, M);

% Omega_eta = 2 * D^+ * (Sigma ⊗ Sigma) * (D^+)'  
Omega_eta = 2 * mDD * kron(Sigma_Eta_Sample, Sigma_Eta_Sample) * mDD';

%% --- Build P_matrix automatically ---
P_matrix = buildPmatrix_vech_to_blocks(n, r);  % function defined below

% vech(Sigma_eta)
VechSigma = mDD * Sigma_Eta_Sample(:);

% lambda = [ vech(Sigma_u) ; vec(Sigma_vu) ; vech(Sigma_v) ]
lambda = P_matrix * VechSigma;

Omega_lambda = P_matrix * Omega_eta * P_matrix';

%% --- Map lambda -> zeta moments ---
% zeta stacks:
% 1) vech( Sigma_vu * Sigma_u^{-1} * Sigma_uv )  (r(r+1)/2 x 1)
% 2) vec( Sigma_vu )                              (n*r x 1)

A = Sigma_vu / Sigma_u;                          % r x n  ( = Sigma_vu * inv(Sigma_u) )
S = A * Sigma_uv;                                % r x r  ( = Sigma_vu*inv(Sigma_u)*Sigma_uv )

zeta_hat = [ mDDr * S(:) ; Sigma_vu(:) ];         % (r(r+1)/2 + n*r) x 1

% Jacobian F_lambda (general dimensions)
F_lambda = [ ...
    -mDDr * kron(A, A) * (mDDn)' ,  2*mDDr*kron(A, eye(r)) , zeros(0.5*r*(r+1), 0.5*r*(r+1)) ; ...
     zeros(n*r, 0.5*n*(n+1))      ,  eye(n*r)              , zeros(n*r,        0.5*r*(r+1)) ...
];

Omega_zeta = F_lambda * Omega_lambda * F_lambda';

% For compatibility with AF naming:
sig_CMD = zeta_hat;          % moments (vector)
mS_opt  = Omega_zeta;        % weighting matrix (cov of moments)

fprintf('\nPartial-ID moments ready.\n');
fprintf('dim(zeta_hat)=%d | dim(Omega_zeta)=%dx%d\n', numel(zeta_hat), size(Omega_zeta,1), size(Omega_zeta,2));


%% ============================================================
%  BASELINE MD (partial-ID) with Phi normalization only (g=r=2)
%  Phi is lower-triangular with positive diagonal:
%      Phi = [exp(a11)   0
%             a21     exp(a22)]
%  one extra restriction on B1.
%  Requires in workspace:
%   - Sigma_u (n x n)  from the proxy-sample moments (1991-2011)
%   - Sigma_vu (r x n) from the proxy-sample moments
%   - S (r x r) = Sigma_vu * inv(Sigma_u) * Sigma_uv   (or recompute)
%   - zeta_hat (11x1) and Omega_zeta (11x11) from your "Partial-ID moments" block
%   - options for fminunc
%   - mDDr (duplication pseudo-inverse for r, already computed)
% ============================================================

%% Starting values (stable)
% If you still have S and Sigma_vu in workspace from the partial-moments block, use them:
%   S       = Sigma_vu * (Sigma_u \ Sigma_vu');   % r x r (same as before)
% Otherwise recompute from your partitions:
S0 = Sigma_vu * (Sigma_u \ Sigma_vu');           % r x r, symmetric PSD

% Phi0 from Cholesky (unique if SPD)
Phi0 = chol(S0,'lower');                          % 2x2 lower-tri with + diag

% Back out B1' ≈ Phi^{-1} Sigma_vu  -> B1 ≈ (Phi^{-1} Sigma_vu)'
B10T = Phi0 \ Sigma_vu;                           % 2x4
B10  = B10T.';                                    % 4x2

% Pack parameters: theta = [vec(B1); a11; a21; a22]
theta0 = [B10(:); log(Phi0(1,1)); Phi0(2,1); log(Phi0(2,2))];

B10(1, 2) = 0; 
theta0 = [B10(:); log(Phi0(1,1)); Phi0(2,1); log(Phi0(2,2))];

%% Optimize 
[theta_hat, Jval, exitflag, output] = fminunc(@MinimumDistance_Partial_g2, theta0, options);

%% Unpack estimates
B1_hat = reshape(theta_hat(1:n*g_shock), n, g_shock);

a11 = theta_hat(n*g_shock + 1);
a21 = theta_hat(n*g_shock + 2);
a22 = theta_hat(n*g_shock + 3);
Phi_hat = [exp(a11) 0; a21 exp(a22)];

% Implied Sigma_vu and S
Sigma_vu_hat = Phi_hat * B1_hat';                 % r x n
S_hat = Sigma_vu_hat * (Sigma_u \ Sigma_vu_hat'); % r x r

% Implied zeta(theta)
zeta_fit = [mDDr * S_hat(:); Sigma_vu_hat(:)];

%% Report
fprintf('\n================ BASELINE MD (Phi normalized) ================\n');
fprintf('exitflag = %d | J = %.6e\n', exitflag, Jval);
fprintf('Iterations: %d | FuncCount: %d\n', output.iterations, output.funcCount);

disp('Phi_hat (2x2):'); disp(Phi_hat);
disp('B1_hat  (n x 2):'); disp(B1_hat);

fprintf('Moment fit: ||zeta_hat - zeta_fit|| = %.3e\n', norm(sig_CMD - zeta_fit));
fprintf('==============================================================\n');

assert(numel(sig_CMD)==11, 'sig_CMD must be 11x1');
assert(all(size(mS_opt)==[11 11]), 'mS_opt must be 11x11');
assert(all(size(Sigma_u)==[n n]), 'Sigma_u size mismatch');
assert(all(size(mDDr)==[r*(r+1)/2, r^2]), 'mDDr size mismatch for r');

% choose 25bp = 0.25 if rates are in percentage points, or 0.0025 if in decimals
target_size = 1;
path_size   = 1;

s1 = target_size / B1_hat(1,1);   % FFR response to shock1
s2 = path_size   / B1_hat(2,2);   % 1Y  response to shock2

D  = diag([s1, s2]);

B1_norm  = B1_hat * D;
Phi_norm = Phi_hat / D;

%% --- CHECK IDENTIFIABILITY(Jacobian Rank Check) ---
% Define the function to compute zeta(theta) analitically
% theta = [vec(B1); a11; a21; a22]
zeta_model = @(th) func_zeta_theta(th, n, r, g_shock, Sigma_u, mDDr);

% Calculate the numeric Jacobian in the optimum theta_hat
J_func = @(th) func_zeta_theta(th, n, r, g_shock, Sigma_u, mDDr);
delta = 1e-6;
num_params = numel(theta_hat);
num_moments = numel(sig_CMD);
Jacobian = zeros(num_moments, num_params);

for i = 1:num_params
    th_plus = theta_hat; th_plus(i) = th_plus(i) + delta;
    th_minus = theta_hat; th_minus(i) = th_minus(i) - delta;
    Jacobian(:, i) = (J_func(th_plus) - J_func(th_minus)) / (2 * delta);
end

% Rank check
rank_J = rank(Jacobian);
cond_J = cond(Jacobian);

fprintf('\n--- Diagnostics ---\n');
fprintf('number of parameters (k): %d\n', num_params);
fprintf('Number of moments (m):   %d\n', num_moments);
fprintf('Rank of the Jacobian:   %d\n', rank_J);
fprintf('Condition Number:        %.2e\n', cond_J);

if rank_J < num_params
    warning('Warning: The Jacobian is not full rank (Rank < k). The model is locally not identified.');
else
    fprintf('Success: The Jacobian is full rank. The required condition for locally identifiability is satisfied.\n');
end
fprintf('------------------------------------\n');

%% 11. IRFs (monthly) from reduced-form VAR + structural impact B1_norm
% EstMdl: estimated VARM object from varm/estimate
% B1_norm: n x 2 impact matrix for the 2 identified shocks (Target, Path)

Hh = 48;                     % horizon in months (48 = 4 years)
p  = NLags;

% Grab reduced-form VAR coefficient matrices A{1..p} (n x n)
Ahat = EstMdl.AR;            % cell(p,1), each n x n

% Build MA coefficients Psi_h: y_t = sum_{h>=0} Psi_h u_{t-h}
Psi = zeros(n,n,Hh+1);
Psi(:,:,1) = eye(n);         % Psi_0

for h = 1:Hh
    acc = zeros(n,n);
    for L = 1:p
        if (h-L) >= 0
            acc = acc + Ahat{L} * Psi(:,:,h-L+1);
        end
    end
    Psi(:,:,h+1) = acc;
end

% Structural IRFs: irf(:,:,h) = Psi_h * B1_norm
IRF = zeros(n, g_shock, Hh+1);
for h = 0:Hh
    IRF(:,:,h+1) = Psi(:,:,h+1) * B1_norm;
end

% --- Quick sanity print (impact = h=0) ---
fprintf('\nImpact IRF (h=0), columns: [Target, Path]\n');
disp(IRF(:,:,1));

%% 12. PLOTS
shock_names = {'Target shock','Path shock'};
hgrid = 0:Hh;

for k = 1:g_shock
    figure('Name',shock_names{k}); 
    tiledlayout(n,1,'TileSpacing','compact','Padding','compact');
    for j = 1:n
        nexttile;
        plot(hgrid, squeeze(IRF(j,k,:)), 'LineWidth', 1.2);
        grid on;
        title(sprintf('%s response to %s', var_names{j}, shock_names{k}));
        xlim([0 Hh]);
        yline(0,'-');
    end
end

%% (optional) save to struct for later (bootstrap bands etc.)
out_irf.Hh        = Hh;
out_irf.var_names = var_names;
out_irf.shock_names = shock_names;
out_irf.B1_norm   = B1_norm;
out_irf.Phi_norm  = Phi_norm;
out_irf.Psi       = Psi;
out_irf.irf       = IRF;



%% REPLICATION TAB 1 (LAKDAWALA, 2019) - 
% Regression for FFR residual
mdl_ffr = fitlm(Z_m, U_m(:,1)); 
% Regression for 1Y residual 
mdl_1y  = fitlm(Z_m, U_m(:,2)); 

fprintf('\n--- REPLICATION TAB 1 (Sample: %d oss) ---\n', mdl_1y.NumObservations);

% Estraction F-statistic e p-value 
stats_1y = anova(mdl_1y,'component',1);
f_stat_1y = mdl_1y.ModelCriterion.AIC; 
f_val_1y = mdl_1y.ModelFitVsNullModel.Fstat;

fprintf('\nRESULTS FOR 1 YEAR RESIDUAL (expected R-squared: 0.101)\n');
fprintf('R-squared:          %.4f (Paper: 0.101)\n', mdl_1y.Rsquared.Ordinary);
fprintf('Adjusted R-squared:  %.4f (Paper: 0.0942)\n', mdl_1y.Rsquared.Adjusted);
fprintf('F-statistic (vs 0):  %.2f  (Paper: 14.73)\n', f_val_1y);

fprintf('\nESTIMATED COEFFICIENTS (1Y Residual):\n');
% Controlla se x2 (Path Factor) e vicino a 0.347
disp(mdl_1y.Coefficients);

fprintf('\nRESULTS FOR FFR RESIDUAL (Expected R-squared: 0.111)\n');
fprintf('R-squared:          %.4f (Paper: 0.111)\n', mdl_ffr.Rsquared.Ordinary);
fprintf('F-statistic (vs 0):  %.2f  (Paper: 18.91)\n', mdl_ffr.ModelFitVsNullModel.Fstat);

%% visualization of the instruments  (PROXY)
if ~exist('r', 'var'), r = size(Z_m, 2); end

figure('Name', 'External Instruments (Lakdawala, 2015)', 'Color', 'w', ...
       'Units', 'normalized', 'Position', [0.2, 0.3, 0.5, 0.5]);
   
t_instr = tiledlayout(r, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

instr_names = {'Target Factor (MP Surprise)', 'Path Factor (Forward Guidance)'};
proxy_colors = {[0.2, 0.4, 0.8], [0.8, 0.4, 0.2]}; 

for i = 1:r
    nexttile;
    
   
    fill_color = proxy_colors{i};
    area(dates_m, Z_m(:,i), 'FaceColor', fill_color, 'EdgeColor', fill_color, ...
         'FaceAlpha', 0.3, 'LineWidth', 1.5);
    
    hold on;
    
    plot(dates_m, zeros(size(dates_m)), 'k-', 'LineWidth', 0.8);
    
    title(instr_names{i}, 'FontSize', 12, 'FontWeight', 'bold', 'FontName', 'Times New Roman');
    set(gca, 'TickDir', 'out', 'Box', 'off', 'FontName', 'Times New Roman', 'FontSize', 10);
    grid on;
    set(gca, 'GridLineStyle', ':', 'GridAlpha', 0.4);
    
    datetick('x', 'yyyy', 'keeplimits');
    ylabel('Units');
    
    y_limit = max(abs(Z_m(:,i))) * 1.1;
    ylim([-y_limit, y_limit]);
end

xlabel(t_instr, 'Year', 'FontSize', 11, 'FontName', 'Times New Roman');
title(t_instr, 'Monetary Policy Proxies (Aligned Sample)', 'FontSize', 14, ...
      'FontWeight', 'bold', 'FontName', 'Times New Roman');


%% BOOTSTRAP with re-estimation of VAR (Full Uncertainty)
fprintf('\nInizio Bootstrap con ri-stima VAR (%d iterazioni)...\n', BootstrapIterations);
IRF_boot = zeros(n, g_shock, Hh+1, BootstrapIterations);
IRF_Point = IRF; 

% Estraction of the coefficients of the original VAR
Ahat_orig = EstMdl.AR; % Cell array p x 1
Const_orig = EstMdl.Constant;

for boot = 1:BootstrapIterations
    % --- 1. generating syntethic (Residual-based Bootstrap) ---
    Tm_curr = size(U_m, 1);
    % Wild Bootstrap for the residuals matched 
    eta = (randn(Tm_curr, 1) > 0) * 2 - 1; 
    U_b_matched = U_m .* repmat(eta, 1, n);
    Z_b = Z_m .* repmat(eta, 1, r);
    
    % we use the Wild Bootstrap on the entire set of residuals U
    T_total = size(U, 1);
    eta_total = (randn(T_total, 1) > 0) * 2 - 1;
    U_b_full = U .* repmat(eta_total, 1, n);
    
    % reconstructing the time series DataSet_b starting from the estimated
    % coefficient
    DataSet_b = zeros(size(AllDataSet));
    DataSet_b(1:NLags, :) = AllDataSet(1:NLags, :); % fixed initial condition
    for t = NLags+1 : size(AllDataSet,1)
        acc = Const_orig;
        for L = 1:NLags
            acc = acc + Ahat_orig{L} * DataSet_b(t-L, :)';
        end
        DataSet_b(t, :) = (acc + U_b_full(t-NLags, :)')'; 
    end

    % --- 2. RE-ESTIMATE THE VAR ON THE SYNTHETIC DATA ---
    Mdl_b = varm(n, NLags);
    Mdl_b.Constant = NaN(n,1);
    [EstMdl_b, ~, ~, ~] = estimate(Mdl_b, DataSet_b, 'Display', 'off');
    Ahat_b = EstMdl_b.AR;

    % --- 3. CALCULATING MOMENTS AND mS_opt ---
    W_b = [U_b_matched, Z_b];
    Sigma_Eta_b = (W_b' * W_b) / Tm_curr;
    S_u_b = Sigma_Eta_b(1:n, 1:n);
    S_uv_b = Sigma_Eta_b(1:n, n+1:end);
    S_vu_b = Sigma_Eta_b(n+1:end, 1:n);
    A_b = S_vu_b / S_u_b;
    S_b = A_b * S_uv_b;
    sig_curr = [ mDDr * S_b(:) ; S_vu_b(:) ]; 
    
    Omega_eta_b = 2 * mDD * kron(Sigma_Eta_b, Sigma_Eta_b) * mDD';
    Omega_lambda_b = P_matrix * Omega_eta_b * P_matrix';
    F_lambda_b = [-mDDr * kron(A_b, A_b) * (mDDn)' ,  2*mDDr*kron(A_b, eye(r)) , zeros(0.5*r*(r+1), 0.5*r*(r+1)) ; ...
                  zeros(n*r, 0.5*n*(n+1))         ,  eye(n*r)                 , zeros(n*r,        0.5*r*(r+1))];
    mS_curr = F_lambda_b * Omega_lambda_b * F_lambda_b';
    if cond(mS_curr) > 1e10; mS_curr = mS_curr + eye(size(mS_curr)) * 1e-9; end

    % --- 4. STRUCTURAL OPTIMIZATION ---
    theta0_boot = theta_hat .* (1 + 0.01 * randn(size(theta_hat)));
    f_boot = @(t) MD_boot_target(t, sig_curr, mS_curr, S_u_b, n, r, g_shock, mDDr);
    [theta_b, ~, exitflag_b, output_b] = fminunc(f_boot, theta0_boot, options);
    % --- 5. CALCULATING IRFs ---
    if exitflag_b > 0
        % Unpack B1
        B1_b = reshape(theta_b(1:n*g_shock), n, g_shock);
        B1_b_norm = B1_b * diag([target_size/B1_b(1,1), path_size/B1_b(2,2)]);
        
        % Calculating Psi_b (the matrix MA of the VAR)
        Psi_b = zeros(n,n,Hh+1);
        Psi_b(:,:,1) = eye(n);
        for h = 1:Hh
            acc_psi = zeros(n,n);
            for L = 1:NLags
                if (h-L) >= 0
                    acc_psi = acc_psi + Ahat_b{L} * Psi_b(:,:,h-L+1);
                end
            end
            Psi_b(:,:,h+1) = acc_psi;
        end
        
        % IRF for this iteration
        for h = 0:Hh
            IRF_boot(:,:,h+1,boot) = Psi_b(:,:,h+1) * B1_b_norm;
        end
    else
        IRF_boot(:,:,:,boot) = NaN; 
    end
    
    if mod(boot,10)==0; fprintf('Iteration %d/%d (VAR re-estimated)\n', boot, BootstrapIterations); end
end

% clearing failures
valid_idx = find(~isnan(squeeze(IRF_boot(1,1,1,:))));
IRF_boot = IRF_boot(:,:,:,valid_idx);
N_valid = numel(valid_idx);
%% 13. Final PLOT with bands SUP-T  - 
fprintf('\nGenerazione grafici unificati...\n');
alpha_coverage = 10; 
hgrid = 0:Hh;

figure('Name', 'Structural IRFs: Target vs Path Shock', 'Color', 'w', ...
       'Units', 'normalized', 'Position', [0.1, 0.1, 0.7, 0.85]);

t = tiledlayout(n, g_shock, 'TileSpacing', 'compact', 'Padding', 'compact');

for j = 1:n 
    for k = 1:g_shock 
        
        nexttile;
        
        % Data for the plot
        TETA_Point = squeeze(IRF_Point(j,k,:));
        TETA_Boot  = squeeze(IRF_boot(j,k,:,:));
        
        % simultaneous bands (Algorithm 3)
        SE_Boot = std(TETA_Boot, 0, 2);
        m_max = zeros(N_valid, 1);
        for b = 1:N_valid
            m_max(b) = max(abs(TETA_Boot(:,b) - TETA_Point) ./ SE_Boot);
        end
        
        C_crit = prctile(m_max, 100 - alpha_coverage);
        IRF_LB = TETA_Point - SE_Boot * C_crit;
        IRF_UB = TETA_Point + SE_Boot * C_crit;
        
        % --- PLOTTING ---
        hold on;
        
        % 1. Area (Grey)
        fill([hgrid, fliplr(hgrid)], [IRF_UB', fliplr(IRF_LB')], ...
             [0.85 0.85 0.85], 'EdgeColor', 'none', 'FaceAlpha', 0.8);
        
        % 2. limits of the bands
        plot(hgrid, IRF_UB, 'r--', 'LineWidth', 0.5); % Limite superiore
        plot(hgrid, IRF_LB, 'r--', 'LineWidth', 0.5); % Limite inferiore
        
        % 3. zero band
        plot(hgrid, zeros(size(hgrid)), 'k-', 'LineWidth', 0.5);
        
        % 4. point estimate
        plot(hgrid, TETA_Point, 'r', 'LineWidth', 2);
        
        % --- axes ---
        if j == 1
            title(shock_names{k}, 'FontSize', 12, 'FontWeight', 'bold', 'FontName', 'Times New Roman');
        end
        
        if k == 1
            ylabel(var_names{j}, 'FontSize', 11, 'FontWeight', 'bold', 'FontName', 'Times New Roman');
        end
        
        set(gca, 'TickDir', 'out', 'Box', 'off', 'FontName', 'Times New Roman', 'FontSize', 9);
        grid on;
        set(gca, 'GridLineStyle', ':', 'GridAlpha', 0.3);
        xlim([0 Hh]);
        
        if j == n
            xlabel('Months', 'FontSize', 10);
        end
        hold off;
    end
end

% Title
title(t, 'Structural Impulse Responses (90% Sup-T Simultaneous Bands)', ...
      'FontSize', 14, 'FontWeight', 'bold', 'FontName', 'Times New Roman');