
##### Algorithm #####

For each closed-polygonal-chain $B_z$, the following cycle is repeated several times (iterations), up to the first iteration at which none new module **(b)** is created (cf. Figure 4):

+ for each open-polygonal-chain $P^*=P_1^H,P_2^H,\ldots,P_M^H,P_1^V,P_2^V,\ldots,P_N^V$ (either horizontal or vertical):

  **find** modules **(a)** – more than one module can be found within a single polygonal-chain $P^*$ –, each one:
  
  - with the maximum-possible length $T^{+}\ge T$
  
  - where all elements are quadrilateral and associated to $B_z$
  
  - and where, on the yet to come module **(b)**, each element:
  
    * height is lower than $h_{MAX}$ aforementioned
        
    * shape-factors `H/V` and `V/H` are lower than those defined into inner-block `maximum_element_shape_factor` multiplied by relevant values within inner-block `element_shape_factor_multiplier` 
      
  **then**: any module **(a)** which is found and for which all the above constraints are fulfilled, is reshaped as a module **(b)**: 
  
  - both extremities $F_0,F_{T^+}$, which are initially composed by two quadrilateral-elements each (cf. Figure 4a), are reshaped as (cf. Figure 4b):
    
    * three triangular-elements each (if `triangular_extremities_tf` is set to **`true`**, i.e.: cyan bounds in Figure 4b are included)
    
    * one quadirateral and one triangular elements (if `triangular_extremities_tf` is set to **`false`**, i.e.: cyan bounds are omitted)
    
    with an exception when, among these extremities, one or both touch any conditioned-boundary (cf. inner-block `boundary_conditions`); in this latter case, extremities touching conditioned-boundaries are reshaped using just quadrilateral-elements, in the same way as it is done for central elements $F_1,F_2,\ldots,F_{T^+-1}$, where each couple of quadrilateral-elements is reshaped to a single quadrilateral-element (cf. Figure 4)
