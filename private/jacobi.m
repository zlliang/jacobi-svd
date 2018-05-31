function [c, s, t] = jacobi(alpha, beta, gamma)
% JACOBI    Compute Jacobi rotation.
%
% The Jacobi rotation is a plane unitray similarity transformation:
%        [ c  s ]T [ alpha  beta ]  [ c  s ]  =  [ l1  0 ]
%        [-s  c ]  [ beta  gamma ]  [-s  c ]  =  [ 0  l2 ]
%
% usage:
%   [c, s, t] = JACOBI(alpha, beta, gamma)
%   [G, t] = JACOBI(alpha, beta, gamma), where G = [c, s; -s, c]
%   
%
% -------------------------------------------------
% Experiments on Matrix Computations -- Spring 2018
% Author: Zilong Liang
% Date:   2018-05-31
% -------------------------------------------------

if beta ~= 0
    tau = (gamma - alpha) / (2 * beta);
    if tau >= 0
        t = 1 / (tau + sqrt(1 + tau^2));
    else
        t = - 1 / (- tau + sqrt(1 + tau^2));
    end
    c = 1 / sqrt(1 + t^2);
    s = t * c;
else
    c = 1;
    s = 0;
    t = 0;
end

if nargout <= 2
    c = [c, s; -s, c];
    s = t;
end
