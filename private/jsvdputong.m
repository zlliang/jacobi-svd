function [sigma, A, V] = jsvdputong(A, tol)

if nargin < 2
    tol = 1e-14;
end

[~, n] = size(A);

V = eye(n);
rots = 1;

sigma = zeros(1, n);
for i = 1:n
    sigma(i) = A(:, i)'*A(:, i);  % TODO
end

while rots >= 1
    rots = 0;
    for p = 1:n-1
        for q = p+1:n
            beta = A(:, p)'*A(:, q);
            if sigma(p)*sigma(q) > 0 && abs(beta) >= tol * sqrt(sigma(p)*sigma(q))  % TODO
                rots = rots + 1;
                [G, t] = jacobi(sigma(p), beta, sigma(q));
                sigma(p) = sigma(p) - beta*t;
                sigma(q) = sigma(q) + beta*t;
                A(:, [p, q]) = A(:, [p, q]) * G;
                V(:, [p, q]) = V(:, [p, q]) * G;
            end
        end
    end
end

[sigma, indices] = sort(sigma, 'descend');
A = A(:, indices);
V = V(:, indices);
for k = 1:n
    if sigma(k) == 0
        break;
    end
    sigma(k) = sqrt(sigma(k));
    A(:, k) = A(:, k) / sigma(k);
end
