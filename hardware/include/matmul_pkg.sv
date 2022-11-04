// Author: Xiaorui Yin

package matmul_pkg;
  // =================================================================
  // Parameter
  // =================================================================

  // Maximum size of the broadcast vector 
  localparam int unsigned MAX_BLEN = `ifdef MAX_BLEN `MAX_BLEN `else 64 `endif;
