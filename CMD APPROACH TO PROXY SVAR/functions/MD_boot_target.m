function J = MD_boot_target(theta, sig_boot, mS_boot, Sigma_u_boot, n, r, g_shock, mDDr)
    % Unpack B1
    B1 = reshape(theta(1:n*g_shock), n, g_shock);
    B1(1, 2) = 0; % Restriction of Lakdawala
    
    % Unpack Phi
    a11 = theta(n*g_shock + 1);
    a21 = theta(n*g_shock + 2);
    a22 = theta(n*g_shock + 3);
    Phi = [exp(a11) 0; a21 exp(a22)];
    
    % Implied moments
    Sigma_vu = Phi * B1';
    S = Sigma_vu * (Sigma_u_boot \ Sigma_vu');
    zeta_model = [mDDr * S(:); Sigma_vu(:)];
    
    % Criterion
    d = sig_boot - zeta_model;
    J = d' * (mS_boot \ d);
end