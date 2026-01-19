function P = buildPmatrix_vech_to_blocks(n, r)
% Maps vech([Sigma_u Sigma_uv; Sigma_vu Sigma_v]) to:
% [vech(Sigma_u); vec(Sigma_vu); vech(Sigma_v)]

M = n + r;
kM = M*(M+1)/2;
kn = n*(n+1)/2;
kr = r*(r+1)/2;

% Build index map for vech positions: position of (i,j) with i>=j in vech
pos = zeros(M,M);
c = 0;
for j = 1:M
    for i = j:M
        c = c + 1;
        pos(i,j) = c;
    end
end

idx = [];

% 1) vech(Sigma_u): i,j in 1..n with i>=j
for j = 1:n
    for i = j:n
        idx(end+1,1) = pos(i,j);
    end
end

% 2) vec(Sigma_vu): Sigma_vu is r x n (rows n+1..M, cols 1..n), no symmetry
% Use column-major vec: stack columns j=1..n, for each i=n+1..M
for j = 1:n
    for i = n+1:M
        % element (i,j) is in lower triangle because i>j always here
        idx(end+1,1) = pos(i,j);
    end
end

% 3) vech(Sigma_v): i,j in n+1..M with i>=j
for j = n+1:M
    for i = j:M
        idx(end+1,1) = pos(i,j);
    end
end

% Build permutation matrix
P = zeros(kn + n*r + kr, kM);
for a = 1:numel(idx)
    P(a, idx(a)) = 1;
end
end
