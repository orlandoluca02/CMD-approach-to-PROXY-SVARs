function zeta = func_zeta_theta(theta, n, r, g, Sigma_u, mDDr)
    % Unpack theta
    B1 = reshape(theta(1:n*g), n, g);
    a11 = theta(n*g+1); a21 = theta(n*g+2); a22 = theta(n*g+3);
    Phi = [exp(a11) 0; a21 exp(a22)];
    
    % Momenti teorici
    Sigma_vu_th = Phi * B1';
    S_th = Sigma_vu_th * (Sigma_u \ Sigma_vu_th');
    
    % Stack zeta
    zeta = [mDDr * S_th(:); Sigma_vu_th(:)];
end