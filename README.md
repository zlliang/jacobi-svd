# Jacobi SVD

[Jacobi eigenvalue algorithm](https://en.wikipedia.org/wiki/Jacobi_eigenvalue_algorithm) is a classical iterative algorithm to compute SVD or symmetric eigensystem. The advantage is that it can compute small eigenvalues (or singular values) more accurate than [QR algorithm](https://en.wikipedia.org/wiki/QR_algorithm), and some accelerating strategies have been proposed to speed up the Jacobi algorithm. This repository contains numerical experiments on this algorithm, especially on its accelerating strategies.

## Results
As an example, I carried out an experiment to test utility of several accelerating strategies in Jacobi SVD algorithm. For methods without accelerating, with de Rijk strategy or QR preprocessing, the experiment compares scanning times and transformation times.
<div align="center"><img src="figures/scan-times.svg" width="400px"><img src="figures/trans-times.svg" width="400px"></div>
According to the experiment, QR preprocessing can accelerating Jacobi algorithm much.

## References
1. 徐树方，钱江 (2011). _**矩阵计算六讲.**_ 北京：高等教育出版社．INFO: [Here](http://www.hep.edu.cn/book/details?uuid=52897a2b-1414-1000-bb7f-3fafc67de19c&objectId=oid:52897b11-1414-1000-bb82-3fafc67de19c).
2. Demmel, J., & Veselić, K. (1992). _**Jacobi’s method is more accurate than QR.**_ SIAM Journal on Matrix Analysis and Applications, 13(4), 1204-1245. DOI: [10.1137/0613074](https://doi.org/10.1137/0613074)
3. Drmač, Z., & Veselić, K. (2008). _**New fast and accurate Jacobi SVD algorithm. I.**_ SIAM Journal on matrix analysis and applications, 29(4), 1322-1342. DOI: [10.1137/050639193](https://doi.org/10.1137/050639193)
4. Drmač, Z., & Veselić, K. (2008). _**New fast and accurate Jacobi SVD algorithm. II.**_ SIAM Journal on Matrix Analysis and Applications, 29(4), 1343-1362. DOI: [10.1137/05063920X](https://doi.org/10.1137/05063920X)
