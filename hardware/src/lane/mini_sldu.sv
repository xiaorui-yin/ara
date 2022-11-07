module mini_sldu
  import ara_pkg::*;
  import matmul_pkg::*;
#(
    parameter int unsigned NrLanes = 4
) (
    input  logic clk_i,
    input  logic rst_ni,
    // Interface with the previous mini sldu (bc_buffer)
    output elen_t                          bc_data_i,
    // Interface with the next mini sldu
    output elen_t                          bc_data_o,
    // Interface with the VMFPU
    input logic                            bc_data_ready_i,
    output elen_t                          bc_data_o,
    output logic                           bc_data_valid_o,
    input logic                            bc_data_invalidate_i
);

endmodule : mini_sldu
