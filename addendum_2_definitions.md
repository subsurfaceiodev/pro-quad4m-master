
##### Definitions #####

Initial-grid in Figure 2 is the set of $N$ vertical and $M$ horizontal segments with endpoints at y-coordinates `[0.0, 250.0]` and at x-coordinates `[-100.0, 500.0]`, respectively. Horizontal and vertical sets intersect forming a grid of quadrilateral-elements $E$ , whose nodes ($G$) nomenclature is shown in Figure 3. The $M$ horizontal and the $N$ vertical segments can be seen as polygonal-chains $P^H$ and $P^V$ with endpoints $\left[G_{1,1}, G_{N,1}\right], \left[G_{1,2}, G_{N,2}\right], \ldots, \left[G_{1,M}, G_{N,M}\right]$, and with endpoints $\left[G_{1,1}, G_{1,M}\right], \left[G_{2,1}, G_{2,M}\right], \ldots, \left[G_{N,1}, G_{N,M}\right]$, respectively. Each polygonal-chain $P$ is the set of segments identified by nodes $G$ of that specific chain, e.g., the first-from-left vertical polygonal-chain $P_1^V$ (cf. Figures 2 and 3) is the set of segments $\left[G_{1,1}, G_{1,2}\right], \left[G_{1,2}, G_{1,3}\right], \ldots, \left[G_{1,M-1}, G_{1,M}\right]$ and contains the set of nodes $G_{1,1}, G_{1,2}, \ldots, G_{1,M}$.

[//]: # "      (1,M)---(2,M)---(3,M)-- . . . --(N,M) " 
[//]: # "        .       .       .               .   " 
[//]: # "        .       .       .               .   " 
[//]: # "        .       .       .               .   " 
[//]: # "      (1,3)---(2,3)---(3,3)-- . . . --(N,3) " 
[//]: # "        |       |       |               |   " 
[//]: # "      (1,2)---(2,2)---(3,2)-- . . . --(N,2) " 
[//]: # "        |       |       |               |   " 
[//]: # " NODE (1,1)---(2,1)---(3,1)-- . . . --(N,1) " 
[//]: # " -----------------------------------------------------------------------         "
[//]: # "http://www.latex2png.com                                                         "
[//]: # "cancella righe: '''math e ''' per latex2png                                      "
[//]: # "\setcounter{MaxMatrixCols}{20}                                                   "
```math
\begin{matrix}
P_1^V&&P_2^V&&P_3^V&&&&&&P_N^V&&\\
\Downarrow&&\Downarrow&&\Downarrow&&&&&&\Downarrow&&\\
G_{1,M}&\text{---}&G_{2,M}&\text{---}&G_{3,M}&\text{---}&\cdotp&\cdotp&\cdotp&\text{---}&G_{N,M}&\Leftarrow&P_M^H\\
\mid&&\mid&&\mid&&&&&&\mid&&\\
\cdotp&&\cdotp&&\cdotp&&&&&&\cdotp&&\\
\cdotp&&\cdotp&&\cdotp&&&&&&\cdotp&&\\
\cdotp&&\cdotp&&\cdotp&&&&&&\cdotp&&\\
\mid&&\mid&&\mid&&&&&&\mid&&\\
G_{1,3}&\text{---}&G_{2,3}&\text{---}&G_{3,3}&\text{---}&\cdotp&\cdotp&\cdotp&\text{---}&G_{N,3}&\Leftarrow&P_3^H\\
\mid&E_{1,2}&\mid&E_{2,2}&\mid&&&&&&\mid&&\\
G_{1,2}&\text{---}&G_{2,2}&\text{---}&G_{3,2}&\text{---}&\cdotp&\cdotp&\cdotp&\text{---}&G_{N,2}&\Leftarrow&P_2^H\\
\mid&E_{1,1}&\mid&E_{2,1}&\mid&&&&&&\mid&&\\
G_{1,1}&\text{---}&G_{2,1}&\text{---}&G_{3,1}&\text{---}&\cdotp&\cdotp&\cdotp&\text{---}&G_{N,1}&\Leftarrow&P_1^H
\end{matrix}
```
[//]: # " -----------------------------------------------------------------------         "
[//]: # "![initial-grid-notations](/utils/figures/initial-grid-notations.svg 'initial-grid-notations')\ " 
*Figure 3: schematic nomenclature for grid elements $E$, nodes $G$ and polygonal-chains $P$ (cf. Figure 2)*

On the other side, the soil-model is represented by a serie of $S$ closed-polygonal-chains $B_1, B_2, \ldots, B_S$, each one identified by indexes in `nodes` within the relevant stratum-block. E.g., $B_2$ in Figure 1 is the closed-polygonal-chain that shapes stratum "St. 2", composed by indexes `[52, 54, 55, 56, 57, 58, 59, 42, 43, 44, 45, 46, 50, 51]` (cf. block `stratum002`).

Considering a single open-polygonal-chain $P_i^j$ (with $j=H,V$ and $i=1,2,\ldots,M,1,2,\ldots,N$), let's define $D_i^j$ as the set of intersection points, between $P_i^j$ and each stratum closed-polygonal-chain $B_z$ (with $z=1,2,\ldots,S$). E.g.: in Figure 2, intersections between:
</p>

- $P_1^V$ and $B_1$ are:
  + $D_{1,1}^V$ as the first-from-bottom intersection (at coordinates `[-100.0, 0.0]`)
  + and $D_{1,2}^V$ as the second - and last - one (`[-100.0, 193.0]`)
</p>

- $P_{112}^V$ – which is around x-coordinate `200.0` [*m*] – and $B_1,B_3,B_6$ are:
  + $D_{112,1}^V,D_{112,2}^V,D_{112,3}^V,D_{112,4}^V$ respectively at `[200, 0.0]`, `[200, 115.0]`, `[200, 165.0]` and `[200, 185.0]`
</p>

- $P_4^H$ – on the other direction, at y-coordinate `12.5` [*m*] – and $B_1$ are:
  + $D_{4,1}^H,D_{4,2}^H$ respectively at `[-100.0, 12.5]` (i.e the first-from-left) and `[500.0, 12.5]`.
</p>
