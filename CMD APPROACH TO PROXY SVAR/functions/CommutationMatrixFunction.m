function K = CommutationMatrixFunction(n, m)
% Genera la matrice di commutazione K(n,m) tale che vec(A') = K(n,m) * vec(A)
% dove A è una matrice n x m.
% Dimensioni di K: (n*m) x (n*m)

I = reshape(1:n*m, [n, m]); % Crea una matrice di indici n x m
I_trans = I';               % Trasponi la matrice degli indici
idx = I_trans(:);           % Vettorizza la trasposta per ottenere l'ordine dei bit

K = eye(n*m);               % Crea una identità
K = K(idx, :);              % Riorordina le righe per ottenere la commutazione
end