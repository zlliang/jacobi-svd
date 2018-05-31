function A = jacobicyclic(A, tol)

if nargin < 2
    tol = 1e-7;
end

n = length(A);
omega = norm(A, 'fro');
V = eye(n);
eta = (tol * omega) ^ 2;
omega = omega^2 - sum(diag(A).^2);

while omega > eta
    for p = 1:n-1
        for q = p+1:n
            G = jacobi(A(p, p), A(p, q), A(q, q));
            A(:, [p, q]) = A(:, [p, q]) * G;
            A([p, q], :) = G' * A([p, q], :);
            V(:, [p, q]) = V(:, [p, q]) * G;
        end
    end
    omega = norm(A, 'fro')^2 - sum(diag(A).^2);
end



