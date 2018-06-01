function A = jacobi_eig(A, method, tol)
% JACOBI_EIG    Solve symmetric eigenvalue problem using Jacobi method
%
% Here, the function computes eigenvalues and corresponding eigenvectors using
% Jacobi method. For experimentation, three method ('classic', 'cyclic',
% 'threshold') can be chosen.
%
% argin:
%   A: The input matrix (must be real-symmetric)
%   method (optional): One of 'classic', 'cyclic' and 'threshold' (default:
%                      'threshold')
%   tol (optional): Stopping criteria
%
% usage:
%   lambda = JACOBI_EIG(A)
%       Only gets eigenvalues: lambda = [l1, l2, ..., ln]
%   [lambda, V] = JACOBI_EIG(A)
%       Computes eigenvalues and corresponding eigenvectors, here A =
%       V'*diag(lambda)*V
%
% -------------------------------------------------
% Experiments on Matrix Computations -- Spring 2018
% Author: Zilong Liang
% Date:   2018-05-31
% -------------------------------------------------

% Check inputs
if nargin < 2
    method = 'threshold';
end
if nargin < 3
    if strcmp(method, 'threshold')
        tol = 1e-14;
    else
        tol = 1e-7;
    end
end

% Initialize
n = length(A);
if nargout == 2
    V = eye(n);
end


if strcmp(method, 'classic')
    omega = norm(A, 'fro');
    eta = tol * omega;
    omega = sqrt(omega^2 - sum(diag(A).^2));
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
        if nargout == 2
            V(:, [p, q]) = V(:, [p, q]) * G;
        end
        omega = sqrt(omega^2 - 2 * apq^2);
    end
elseif strcmp(method, 'cyclic')
    omega = norm(A, 'fro');
    eta = tol * omega;
    omega = sqrt(omega^2 - sum(diag(A).^2));
    while omega > eta
        for p = 1:n-1
            for q = p+1:n
                G = jacobi(A(p, p), A(p, q), A(q, q));
                A(:, [p, q]) = A(:, [p, q]) * G;
                A([p, q], :) = G' * A([p, q], :);
                if nargout == 2
                    V(:, [p, q]) = V(:, [p, q]) * G;
                end
            end
        end
        omega = sqrt(norm(A, 'fro')^2 - sum(diag(A).^2));
    end
elseif strcmp(method, 'threshold')
    rots = 1;
    while rots >= 1
        rots = 0;
        for p = 1:n-1
            for q = p+1:n
                apq = A(p, q);
                app = A(p, p);
                aqq = A(q, q);
                if abs(apq) >= tol * sqrt(app * aqq)
                    rots = rots + 1;
                    G = jacobi(app, apq, aqq);
                    A(:, [p, q]) = A(:, [p, q]) * G;
                    A([p, q], :) = G' * A([p, q], :);
                    if nargout == 2
                        V(:, [p, q]) = V(:, [p, q]) * G;
                    end
                end
            end
        end
    end
else
    error('Choose method among ''classic'', ''cyclic'' and ''threshold''');
end



