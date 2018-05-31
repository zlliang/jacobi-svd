function A = jacobiclassic(A, tol)

if nargin < 2
    tol = 1e-7;
end

n = length(A);
omega = norm(A, 'fro');
V = eye(n);
eta = (tol * omega) ^ 2;
omega = omega^2 - sum(diag(A).^2);
while omega > eta
    [a, p] = max(abs(A - diag(diag(A))));
    [~, q] = max(a);
    p = p(q);
    if p > q
        temp = p;
        p = q;
        q = temp;
    end
    apq = A(p, q);
    G = jacobi(A(p, p), apq, A(q, q));
    A(:, [p, q]) = A(:, [p, q]) * G;
    A([p, q], :) = G' * A([p, q], :);
    V(:, [p, q]) = V(:, [p, q]) * G;
    omega = norm(A, 'fro')^2 - sum(diag(A).^2);
end

