function IRF = get_structural_irf(U_m, Z_m, Psi, n, r, g_shock, Sigma_u, options, theta0_input)
    global mDDr sig_CMD mS_opt
    Tm = size(U_m,1);
    W = [U_m, Z_m]; Sigma_Eta = (W'*W)/Tm;
    % Calcolo zeta_hat e mS_opt 
    
    % optimization
    [theta_hat] = fminunc(@MinimumDistance_Partial_g2, theta0_input, options);
    B1_hat = reshape(theta_hat(1:n*g_shock), n, g_shock);
    
    % Normalization
    D = diag([1/B1_hat(1,1), 1/B1_hat(2,2)]);
    B1_norm = B1_hat * D;
    
    % IRF
    Hh = size(Psi, 3) - 1;
    IRF = zeros(n, g_shock, Hh+1);
    for h = 0:Hh
        IRF(:,:,h+1) = Psi(:,:,h+1) * B1_norm;
    end
end