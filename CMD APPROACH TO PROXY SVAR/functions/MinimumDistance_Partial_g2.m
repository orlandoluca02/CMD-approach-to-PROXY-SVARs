function J = MinimumDistance_Partial_g2(theta)
% Partial-ID MD objective with g=r=2 and Phi normalization (lower-tri, +diag)
% ADDED: Lakdawala Zero Restriction B(1,2) = 0
global sig_CMD mS_opt Sigma_u n r g_shock mDDr

% --- unpack B1 ---
B1 = reshape(theta(1:n*g_shock), n, g_shock);

% --- RESTRICTION OF LAKDAWALA ---

B1(1, 2) = 0; 

% --- unpack Phi params ---
a11 = theta(n*g_shock + 1);
a21 = theta(n*g_shock + 2);
a22 = theta(n*g_shock + 3);

Phi = [exp(a11) 0;
       a21      exp(a22)];           

% --- implied moments ---
Sigma_vu = Phi * B1';                 % r x n
S = Sigma_vu * (Sigma_u \ Sigma_vu'); % r x r
zeta_model = [mDDr * S(:); Sigma_vu(:)];

% --- MD criterion (stable) ---
d = sig_CMD - zeta_model;
J = d' * (mS_opt \ d);
end