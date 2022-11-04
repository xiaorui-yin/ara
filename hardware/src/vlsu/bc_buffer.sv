// Author: Xioarui Yin
// Description:
// Broadcast buffer receives the data read by the load unit from L2 cache,
// and send it to the first lane (lane0) 

module bc_buffer import ara_pkg::*; import rvv_pkg::*; #(
  parameter  int  unsigned NrLanes = 0,
  parameter  int  unsigned AxiDataWidth = 0,

  localparam int unsigned BufferCounterWidth = $clog2(MAX_BLEN)
) (
  input logic                            clk_i,
  input logic                            rst_ni,
  // Interface with the load unit
  input elen_t             [NrLanes-1:0] ldu_result_wdata_i, 
  input logic              [NrLanes-1:0] ldu_result_req_i,
  /* input strb_t             [NrLanes-1:0] ldu_result_be_i, */
  output logic             [NrLanes-1:0] ldu_result_gnt_o,
  output logic             [NrLanes-1:0] ldu_result_final_gnt_o,
  // Interface with the first lane
  input logic                            bc_data_ready_i,
  output elen_t                          bc_data_o,
  output logic                           bc_data_valid_o,
  input logic                            bc_data_invalidate_i
);

  import matmul_pkg::*;

  // =================================================================
  // Signals Declaration
  // =================================================================
  
  logic write_buffer_id_d, write_buffer_id_q,
        read_buffer_id_d, read_buffer_id_q;
  logic [1:0] buffer_flush;

  logic [1:0] buffer_full, buffer_empty, buffer_push, buffer_final_push,
              buffer_pop;

  assign buffer_flush = bc_data_invalidate_i ?
                        (read_buffer_id_q ? 2'b10 : 2'b01) :
                        2'b00;

  logic [ELEN*NrLanes-1:0] buffer_din;
  elen_t buffer_dout;

  // Serializer
  for (int lane = 0; lane < NrLanes; lane++) begin
    buffer_din[lane*ELEN +: ELEN] = ldu_result_wdata_i[lane];
  end


  for (genvar i = 0; i < 2; i++) begin: gen_bc_buffer
    re_readable_fifo #(
      .DEPTH(MAX_BLEN),
      .WR_DATA_WIDTH(ELEN * NrLanes),
      .RD_DATA_WIDTH(ELEN)
    ) i_bc_buffer (
      .clk_i,
      .rst_ni,
      .flush_i      (buffer_flush[i]),
      .testmode_i   (1'b0),
      .full_o       (buffer_full),
      .empty_o      (buffer_empty),
      .data_i       (buffer_din),
      .push_i       (buffer_push),
      .final_push_i (buffer_final_push),
      .data_o       (buffer_dout),
      .pop_i        (buffer_pop),
      .usage_o      (/* unused */)
    );
  end: gen_bc_buffer


  always_comb begin
      
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      bc_buffer_q <= '0;
    end else begin
      bc_buffer_q <= bc_buffer_d;
    end
  end

  // =================================================================
  // Asserations
  // =================================================================
  
  if (MAX_BLEN % NrLanes != 0)
    $error("The maximum broadcast vector length must be a multiple of the number of laens.")

endmodule
