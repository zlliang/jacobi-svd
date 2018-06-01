%% Experiments on Jacobi SVD Method
%
% This file includes some experiments on Jacobi method for symmetric eigenvalue
% problems and general SVD problems.
%
%  -------------------------------------------------
%  Experiments on Matrix Computations -- Spring 2018
%  Author: Zilong Liang
%  Date:   2018-05-31
%  -------------------------------------------------


%% Before experiments: Introduction
%
% This part introduces some special matrices in the following experiments.

clear; close all; clc;

%% Experiment 1: Jacobi method is more accurate than QR
A = [1e40, 1e29, 1e19; 1e29, 1e20, 1e9; 1e19, 1e9, 1];


%% Experiment 2: Accelerating strategies of Jacobi SVD algorithm