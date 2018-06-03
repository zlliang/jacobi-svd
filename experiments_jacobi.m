%% Experiments on Jacobi SVD Method
%
% This file includes some experiments on Jacobi method for symmetric eigenvalue
% problems and general SVD problems.
%
%  -------------------------------------------------
%  Experiments on Matrix Computations -- Spring 2018
%  Author: Zilong Liang
%  Date:   2018-06-01
%  -------------------------------------------------


%% Experiment 1: Jacobi method is more accurate than QR

clear; close all; clc;

A = [1e40, 1e29, 1e19; 1e29, 1e20, 1e9; 1e19, 1e9, 1];
lambda1 = eig(A);
lambda2 = jacobi_eig(A);


%% Experiment 2: Accelerating strategies of Jacobi SVD algorithm
%
% This experiment is to test utility of several speed-up strategies in Jacobi
% SVD algorithm. For methods without accelerating, with de Rijk strategy or QR
% preprocessing, we compare scanning times and transformation times.

clear; close all; clc;

nlen = 20;
mtdlen = 4;
rep = 10;
ms = floor(linspace(50, 300, nlen));
ns = floor(linspace(25, 150, nlen));
ks = floor(linspace(10, 60, nlen));
methods = {'none', 'derijk', 'qr', 'derijk-qr'};
scan = zeros(nlen, mtdlen);
trans = zeros(nlen, mtdlen);
for i = 1:nlen
    for r = 1:rep
        A = rand(ms(i), ns(i));
        [U, D, V] = svd(A);
        A = U(:, 1:ks(i)) * D(1:ks(i), 1:ks(i)) * V(:, 1:ks(i))';
        for m = 1:mtdlen
            [~, ~, ~, s, t] = jacobi_svd(A, methods{m});
            scan(i, m) = scan(i, m) + s;
            trans(i, m) = trans(i, m) + t;
        end
    end
end
scan = scan ./ rep;
trans = trans ./ rep;
figure(1);
plot(ns, scan, '.-', 'LineWidth', 1);
legend(methods);
xlabel('dim: $2n \times n$');
ylabel('Scanning times');
setstyle(gca, 'latex');
title('Experiment: Scanning times');
figure(2);
plot(ns, sqrt(trans), '.-', 'LineWidth', 1);
legend(methods);
xlabel('dim: $2n \times n$');
ylabel('Transformation Times$^{1/2}$');
setstyle(gca, 'latex');
title('Experiment: Transformation times');


%% Experiment: In QR accelerating, choose R or R'
%
% Drmac and Veselic (2008) claim that choose R' instead of R in QR accelerating
% strategy may speed-up convergence. This experiment test this conclusion.

clear; close all; clc;

nlen = 20;
mtdlen = 2;
rep = 10;
ns = floor(linspace(25, 150, nlen));
ks = floor(linspace(10, 60, nlen));
methods = {'qr', 'qr-notranspose'};
scan = zeros(nlen, mtdlen);
trans = zeros(nlen, mtdlen);
for i = 1:nlen
    for r = 1:rep
        A = rand(ns(i), ns(i));
        
        [~, ~, ~, s, t] = jacobi_svd(A, 'derijk-qr');
        scan(i, 1) = scan(i, 1) + s;
        trans(i, 1) = trans(i, 1) + t;
        
        [~, s, t] = jacobi_svd_notranspose(A);
        scan(i, 2) = scan(i, 2) + s;
        trans(i, 2) = trans(i, 2) + t;
    end
end
scan = scan ./ rep;
trans = trans ./ rep;
figure(1);
plot(ns, scan, '.-', 'LineWidth', 1);
legend(methods);
xlabel('dim: $2n \times n$');
ylabel('Scanning times');
setstyle(gca, 'latex');
title('Experiment: Scanning times');
figure(2);
plot(ns, sqrt(trans), '.-', 'LineWidth', 1);
legend(methods);
xlabel('dim: $2n \times n$');
ylabel('Transformation Times$^{1/2}$');
setstyle(gca, 'latex');
title('Experiment: Transformation times');


%% Helper functions

% Helper function: Choose R itself in QR accelerating (derijk-qr)
function [sigma, i, j] = jacobi_svd_notranspose(A)
    if nargin < 2
        method = 'none';
    end
    [~, n] = size(A);
    [~, R, ~] = qr(A, 'vector');
    R = R(1:n, 1:n);
    k = find(abs(diag(R)) < eps * norm(R, 'fro'), 1) - 1;
    if k  % k < n
        [~, R1] = qr(R(1:k, 1:n)');
        R1 = R1(1:k, 1:k);
        if nargout > 1
            [sigma, ~, ~, i, j] = jacobi_svd(R1, 'derijk');
        else
            sigma = jacobi_svd(R1, method);
        end
    else  % k == n
        if nargout > 1
            [sigma, ~, ~, i, j] = jacobi_svd(R, 'derijk');
        else
            sigma = jacobi_svd(R, method);
        end
    end
end


