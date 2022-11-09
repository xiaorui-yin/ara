module mini_sldu
  import ara_pkg::*;
  import matmul_pkg::*;
#(
    parameter int unsigned NrLanes = 4
) (
    input  logic clk_i,
    input  logic rst_ni,
    // From the previous lane
    /* input logic                            bc_data_ready_o, */
    output elen_t                          bc_data_i,
    /* output logic                           bc_data_valid_i, */
    // To the next lane
    /* input logic                            bc_data_ready_i, */
    output elen_t                          bc_data_o
    /* output logic                           bc_data_valid_o, */
    // first lane only
    /* input logic                            bc_data_invalidate_i */
);

  /* spill_register #( */
  /*   .T(elen_t) */
  /* ) i_bc_spill_register( */
  /*   .clk_i  (clk_i          ), */
  /*   .rst_ni (rst_ni         ), */
  /*   .valid_i(bc_data_valid_i), */
  /*   .ready_o(bc_data_ready_o), */
  /*   .data_i (bc_data_i      ), */
  /*   .valid_o(bc_data_valid_o), */
  /*   .ready_i(bc_data_ready_i), */
  /*   .data_o (bc_data_o      ) */
  /* ); */

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      bc_data_o <= '0;
    end else begin
      bc_data_o <= bc_data_i;
    end
  end

endmodule : mini_sldu
