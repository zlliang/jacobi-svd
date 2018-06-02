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


%% Before experiments: Introduction
%
% This part introduces some special matrices in the following experiments.

clear; close all; clc;

%% Experiment 1: Jacobi method is more accurate than QR
A = [1e40, 1e29, 1e19; 1e29, 1e20, 1e9; 1e19, 1e9, 1];


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
    i
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


