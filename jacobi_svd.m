function [sigma, U, V, i, j] = jacobi_svd(A, method)
% JACOBI_SVD    Singular value decomposition using one-side Jacobi method
%
% Here, the function computes singular values and corresponding singular vectors
% using Jacobi method. For experimentation, several speed-up strategies are
% avaliable to choose.
%
% argin:
%   A: The input matrix (size: m x n, and m > n)
%   method (optional): One of the following strategies:
%       'none',
%       'derijk',
%       'qr',
%       'derijk-qr'.
%
% usage:
%   sigma = JACOBI_SVD(A)
%       Only gets singular values: sigma = [s1; s2; ...; sn]
%   [sigma, U, V] = JACOBI_SVD(A)
%       Computes eigenvalues and corresponding singular vectors, U is a m x r
%       matrix of left singular vectors and V a n x n matrix of right singular
%       vectors, which is to say, U'*diag(sigma)*V = A
%
% -------------------------------------------------
% Experiments on Matrix Computations -- Spring 2018
% Author: Zilong Liang
% Date:   2018-06-01
% -------------------------------------------------

% Check inputs
if nargin < 2
    method = 'none';
end
[m, n] = size(A);

% Check if QR preprocessing is chosen, and recursively call the function itself,
% with QR preprocessing
if strcmp(method, 'qr') || strcmp(method, 'derijk-qr')
    
    if strcmp(method, 'qr')
        method = 'none';
    else
        method = 'derijk';
    end
    [Q, R, p] = qr(A, 'vector');
    R = R(1:n, 1:n);
    k = find(abs(diag(R)) < eps * norm(R, 'fro'), 1) - 1;
    if k  % k < n
        [Q1, R1] = qr(R(1:k, 1:n)');
        R1 = R1(1:k, 1:k);
        if nargout > 1
            [sigma, U1, V1, i, j] = jacobi_svd(R1', method);
            U = Q * blkdiag(U1, eye(m-k));
            U = U(:, 1:k);
            V = Q1 * blkdiag(V1, eye(n-k));
            V = V(p, :);
        else
            sigma = jacobi_svd(R1', method);
        end
    else  % k == n
        if nargout > 1
            [sigma, U1, V1, i, j] = jacobi_svd(R', method);
            U = Q * blkdiag(V1, eye(m-n));
            U = U(:, 1:n);
            V = U1(p(p), :);
        else
            sigma = jacobi_svd(R', method);
        end
    end
    return;
end

% Initialize
tol = 1e-14;
rots = 1;
sigma = sum(A.^2)';
if nargout > 1
    V = eye(n);
end

% Scanning
i = 0;
j = 0;
tolsigma = tol * norm(A, 'fro');
while rots >= 1
    i = i + 1;
    rots = 0;
    for p = 1:n-1
        if strcmp(method, 'derijk')
            [~, k] = max(sigma(p:n));
            k = k + p - 1;
            if k ~= p
                sigma([k, p]) = sigma([p, k]);
                A(:, [k, p]) = A(:, [p, k]);
                V(:, [k, p]) = V(:, [p, k]);
            end
        end
        for q = p+1:n
            beta = A(:, p)'*A(:, q);
            if sigma(p)*sigma(q) > tolsigma && ...
            abs(beta) >= tol * sqrt(sigma(p)*sigma(q))
                j = j + 1;
                rots = rots + 1;
                [G, t] = jacobi(sigma(p), beta, sigma(q));
                sigma(p) = sigma(p) - beta*t;
                sigma(q) = sigma(q) + beta*t;
                A(:, [p, q]) = A(:, [p, q]) * G;
                if nargout > 1
                    V(:, [p, q]) = V(:, [p, q]) * G;
                end
            end
        end
    end
end

% Post-processing
[sigma, indices] = sort(sigma, 'descend');
if nargout > 1
    U = A(:, indices);
    V = V(:, indices);
end
for k = 1:n
    if sigma(k) == 0
        sigma(k:end) = 0;
        U = U(:, 1:k-1);
        break;
    end
    sigma(k) = sqrt(sigma(k));
    if nargout > 1
        U(:, k) = U(:, k) / sigma(k);
    end
end

