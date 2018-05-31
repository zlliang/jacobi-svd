function sigma = jsvdpreprocessing(A)

[~, n] = size(A);

[~, R, ~] = qr(A, 'vector');
R = R(1:n, 1:n);

[sigma, ~, V] = jsvdderijk(R');

