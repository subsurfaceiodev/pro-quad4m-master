##### Algorithm #####

For each intersection point $D_{i,k}^j$ (with $i,j$ as aforementioned and $k=1, 2, \ldots$ – up to the number of intesections between $P_i^j$ and each $B_z$):

+ $V_{i,k}^j$ is the signed-distance between the intersection $D_{i,k}^j$ and its closest node $G^\ast$ among those of $P_i^j$

+ then, at each node $G^\dag$ in $P_i^j$:

  * $C_I=\left(1-d_I/d_T\right)^u$, where $d_I$ is the (*non*-signed) distance between the intersection point $D_{i,k}^j$ and the node $G^\dag$

  * $C_K=\left(d_K/d_T\right)^v,C_J=\left(d_J/d_T\right)^v$, where $d_K,d_J$ are the (*non*-signed) distances between each of the two endpoints of $P_i^j$ and the node $G^\dag$

  * finally, at each node $G^\ddag$ of the polygonal-chain $P_m^n$ which contain node $G^\dag$ and is transversal to $P_i^j$:

    - $C_O=\left(1-d_O/d_T\right)^w$, where $d_O$ is the (*non*-signed) distance between nodes $G^\dag$ and $G^\ddag$

    - if any $C_I, C_K, C_J, C_O$ is higher than 1.0 or lower than 0.0, it is truncated respectively to 1.0 or 0.0

    - node $G^\ddag$ is moved by a quantity equal to $L^j \cdot V_{i,k}^j \cdot C_I \cdot C_K \cdot C_J \cdot C_O$ along the direction $j$

The whole algorithm reported above is repeated $t$ times.
