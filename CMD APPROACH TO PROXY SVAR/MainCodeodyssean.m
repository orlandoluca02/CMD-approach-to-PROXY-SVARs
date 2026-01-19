%% REPLICATION & COMPARISON: ORIGINAL VS ODYSSIAN PATH SHOCK
clc; clear; close all;
addpath functions;

% Global variables
global NLags TAll sig_CMD mS_opt n r g_shock M Sigma_u mDDr

%% 1. settings and data
NLags = 12;
target_size = 1; 
path_size   = 1;
options = optimoptions('fminunc','Display','off','MaxFunctionEvaluations',200000,...
    'OptimalityTolerance',1e-10,'StepTolerance',1e-10,'MaxIterations',200000);

load(fullfile('Data','all_data.mat'));
data_matrix   = Dataset1.all_data;
all_dates_raw = Dataset1.all_dates;
var_idx   = [3 4 1 2]; % [FFR, 1Y, log(CPI), log(IP)]
var_names = {'FFR','1Y','log(CPI)','log(IP)'};
DataSet = data_matrix(:, var_idx);

n = 4; r = 2; g_shock = 2; M = n + r;

%% 2. filter data and estimate VAR
start_date = datenum('7/1/1979');
end_date   = datenum('12/1/2011');
start  = find(all_dates_raw == start_date, 1);
finish = find(all_dates_raw == end_date,   1);
VAR_dates  = all_dates_raw(start:finish);
AllDataSet = DataSet(start:finish, :);

Mdl = varm(n, NLags);
Mdl.Constant = NaN(n,1);
[EstMdl, ~, ~, U_full] = estimate(Mdl, AllDataSet);

% residuals and effective data
if size(U_full,1) == size(AllDataSet,1)
    nan_rows = any(isnan(U_full), 2);
    U = U_full(~nan_rows, :);
    rf_dates = VAR_dates(~nan_rows);
else
    U = U_full;
    rf_dates = VAR_dates(NLags+1:end);
end
%% 3. allignment instruments 
% --- A model: ORIGINAL (TP1_instr.mat) ---
data_orig = load(fullfile('Instruments','TP1_instr.mat')); 
[tf1, loc1] = ismember(rf_dates, data_orig.instr.ProxyDates);
U_m1 = U(tf1, :); 
Z_m1 = data_orig.instr.Proxy(loc1(tf1), :);

% --- B model: ODYSSEAN (TP1_pvt_res_instr.mat) ---
data_ody = load(fullfile('Instruments','TP1_pvt_res_instr.mat'));
[tf2, loc2] = ismember(rf_dates, data_ody.instr.ProxyDates);
U_m2 = U(tf2, :); 
Z_m2 = data_ody.instr.Proxy(loc2(tf2), :);

fprintf('Allignment completed: %d observations for Original, %d for Odyssean.\n', size(Z_m1,1), size(Z_m2,1));
%% 4. support matrix and initial values
D_M = DuplicationMatrixFunction(M);   mDD = (D_M' * D_M) \ D_M';
D_r = DuplicationMatrixFunction(r);   mDDr = (D_r' * D_r) \ D_r';
D_n = DuplicationMatrixFunction(n);   mDDn = (D_n' * D_n) \ D_n';
P_matrix = buildPmatrix_vech_to_blocks(n, r);

% theta0 (Starting Values) using the first set
Sigma_vu_init = (Z_m1' * U_m1) / size(U_m1,1);
Sigma_u_init  = (U_m1' * U_m1) / size(U_m1,1);
S0 = Sigma_vu_init * (Sigma_u_init \ Sigma_vu_init');
Phi0 = chol(S0,'lower');
B10 = (Phi0 \ Sigma_vu_init).';
B10(1,2) = 0; 
theta0 = [B10(:); log(Phi0(1,1)); Phi0(2,1); log(Phi0(2,2))];

%% 5. IDENTIFICATION AND IRFs
Hh = 48;
Psi = calculate_MA_coefficients(EstMdl, n, NLags, Hh);

fprintf('Estimating Odyssian Model...\n');
IRF_Odyssian = run_full_identification(U_m1, Z_m1, Psi, n, r, g_shock, options, theta0, P_matrix, mDD, mDDn, mDDr);

fprintf('Estimating Original Model...\n');
IRF_Original = run_full_identification(U_m2, Z_m2, Psi, n, r, g_shock, options, theta0, P_matrix, mDD, mDDn, mDDr);

%% 6. PLOT (PATH SHOCK)
hgrid = 0:Hh;
shock_k = 2; % 2 = Path Shock

figure('Name', 'Comparison: Mixed vs Odyssian Path Shock', 'Color', 'w', 'Units', 'normalized', 'Position', [0.2 0.1 0.4 0.8]);
tlo = tiledlayout(n, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

for j = 1:n
    nexttile;
    hold on;
    plot(hgrid, zeros(size(hgrid)), 'k-', 'LineWidth', 0.5);
    
    % Odyssian 
    p1 = plot(hgrid, squeeze(IRF_Odyssian(j, shock_k, :)), '--', 'Color', [0.2 0.4 0.8], 'LineWidth', 1.5);
    % Original 
    p2 = plot(hgrid, squeeze(IRF_Original(j, shock_k, :)), '--', 'Color', [0.8 0.2 0.2], 'LineWidth', 2);
    
    ylabel(var_names{j}, 'FontWeight', 'bold');
    grid on; xlim([0 Hh]);
    set(gca, 'TickDir', 'out', 'Box', 'off');
    
    if j == 1
        legend([p1, p2], {'Original Path (Delphic+Odyssean)', 'Pure Odyssen Path'}, 'Location', 'best', 'Box', 'off');
    end
end
title(tlo, 'Comparison of Forward Guidance Identifications', 'FontSize', 14, 'FontWeight', 'bold');
xlabel(tlo, 'Months after shock');

% =========================================================================
% SUPPORT FUNCTION
% =========================================================================

function IRF = run_full_identification(U_m, Z_m, Psi, n, r, g_shock, options, theta0, P_matrix, mDD, mDDn, mDDr)
    global sig_CMD mS_opt Sigma_u
    
    Tm = size(U_m,1);
    W = [U_m, Z_m]; Sigma_Eta = (W'*W)/Tm;
    Sigma_u = Sigma_Eta(1:n, 1:n);
    S_uv = Sigma_Eta(1:n, n+1:end);
    S_vu = Sigma_Eta(n+1:end, 1:n);
    A = S_vu / Sigma_u;
    S = A * S_uv;
    
    Omega_eta = 2 * mDD * kron(Sigma_Eta, Sigma_Eta) * mDD';
    Omega_lambda = P_matrix * Omega_eta * P_matrix';
    
    F_lambda = [-mDDr * kron(A, A) * (mDDn)', 2*mDDr*kron(A, eye(r)), zeros(0.5*r*(r+1), 0.5*r*(r+1));
                 zeros(n*r, 0.5*n*(n+1)), eye(n*r), zeros(n*r, 0.5*r*(r+1))];
    
    sig_CMD = [ mDDr * S(:) ; S_vu(:) ];
    mS_opt  = F_lambda * Omega_lambda * F_lambda';
    
    [theta_hat] = fminunc(@MinimumDistance_Partial_g2, theta0, options);
    B1_hat = reshape(theta_hat(1:n*g_shock), n, g_shock);
    
    % Normalization: 1 unit on FFR (shk1) and 1Y (shk2)
    D = diag([1/B1_hat(1,1), 1/B1_hat(2,2)]);
    B1_norm = B1_hat * D;
    
    Hh = size(Psi, 3) - 1;
    IRF = zeros(n, g_shock, Hh+1);
    for h = 0:Hh
        IRF(:,:,h+1) = Psi(:,:,h+1) * B1_norm;
    end
end

function Psi = calculate_MA_coefficients(EstMdl, n, p, Hh)
    Ahat = EstMdl.AR;
    Psi = zeros(n,n,Hh+1);
    Psi(:,:,1) = eye(n);
    for h = 1:Hh
        acc = zeros(n,n);
        for L = 1:p
            if (h-L) >= 0
                acc = acc + Ahat{L} * Psi(:,:,h-L+1);
            end
        end
        Psi(:,:,h+1) = acc;
    end
end