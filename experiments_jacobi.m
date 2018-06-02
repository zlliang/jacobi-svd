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

nlen = 9;
mtdlen = 4;
rep = 5;
ms = floor(linspace(50, 300, nlen));
ns = floor(linspace(20, 100, nlen));
methods = {'none', 'derijk', 'qr', 'derijk-qr'};
scan = zeros(mtdlen, nlen);
trans = zeros(mtdlen, nlen);
for i = 1:nlen
    i
    for r = 1:rep
        A = rand(ms(i), ns(i));
        for m = 1:mtdlen
            [~, ~, ~, s, t] = jacobi_svd(A, methods{m});
            scan(m, i) = scan(m, i) + s;
            trans(m, i) = trans(m, i) + t;
        end
    end
end
scan = scan / rep;
trans = trans / rep;

