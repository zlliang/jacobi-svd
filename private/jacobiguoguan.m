function A = jacobiguoguan(A, tol)

if nargin < 2
    tol = 1e-14;
end

n = length(A);
V = eye(n);
rots = 1;
while rots >= 1
    rots = 0;
    for p = 1:n-1
        for q = p+1:n
            if abs(A(p, q)) >= tol * sqrt(A(p, p) * A(q, q))
                rots = rots + 1;
                G = jacobi(A(p, p), A(p, q), A(q, q));
                A(:, [p, q]) = A(:, [p, q]) * G;
                A([p, q], :) = G' * A([p, q], :);
                V(:, [p, q]) = V(:, [p, q]) * G;
            end
        end
    end
end

