// Author: Xioarui Yin
// Description:
// Broadcast buffer receives the data read by the load unit from L2 cache,
// and send it to the first lane (lane0)
// Only support fp32

module bc_buffer import ara_pkg::*; import rvv_pkg::*; #(
    parameter  int  unsigned NrLanes = 0,
    parameter  int  unsigned AxiDataWidth = 0,

    localparam int  unsigned MAX_BLEN = 32,
    localparam int  unsigned BufferCounterWidth = $clog2(MAX_BLEN)
  ) (
    input  logic                            clk_i,
    input  logic                            rst_ni,
    // Interface with the load unit
    input  logic              [NrLanes-1:0] ldu_result_req_i,
    /* input  vid_t              [NrLanes-1:0] ldu_result_id_i, */
    /* input  vaddr_t            [NrLanes-1:0] ldu_result_addr_i, */
    input  elen_t             [NrLanes-1:0] ldu_result_wdata_i,
    /* input  strb_t             [NrLanes-1:0] ldu_result_be_i, */
    output logic              [NrLanes-1:0] ldu_result_gnt_o,
    output logic              [NrLanes-1:0] ldu_result_final_gnt_o,
    // Interface with the first lane
    input  logic                            bc_ready_i,
    output elen_t                           bc_data_o,
    output logic                            bc_valid_o,
    input  logic                            bc_invalidate_i
  );

  // =================================================================
  // Ping-Pang buffer
  // =================================================================

  logic                    write_buffer_id_d, write_buffer_id_q,
                           read_buffer_id_d, read_buffer_id_q;
  logic [1:0]              buffer_flush;
  logic [1:0]              buffer_full, buffer_empty,
                           buffer_push, buffer_pop;
  logic [ELEN*NrLanes-1:0] buffer_din;
  elen_t [1:0]             buffer_dout;
  logic [1:0]              buffer_load_finished;

  for (genvar i = 0; i < 2; i++) begin: gen_re_readable_buffer
    re_readable_fifo #(
      .DEPTH(MAX_BLEN/2), // TODO: scale to the read data bus
      .WR_DATA_WIDTH(ELEN * NrLanes),
      .RD_DATA_WIDTH(ELEN)
    ) i_re_readable_fifo (
      .clk_i,
      .rst_ni,
      .flush_i        (buffer_flush[i]         ),
      .testmode_i     (1'b0                    ),
      .full_o         (buffer_full[i]          ),
      .empty_o        (buffer_empty[i]         ),
      .data_i         (buffer_din              ),
      .push_i         (buffer_push[i]          ),
      .data_o         (buffer_dout[i]          ),
      .pop_i          (buffer_pop[i]           ),
      .load_finished_o(buffer_load_finished[i] ),
      .usage_o        (/* unused */            )
    );
  end: gen_re_readable_buffer

  // =============================================================
  // Input Data Serialization
  // =============================================================

  always_comb begin
    buffer_din = '0;
    for (int i = 0; i < NrLanes; i++) begin
      // TODO: FP32 only
      buffer_din[32 * i +: 32]             = ldu_result_wdata_i[i][31:0];
      buffer_din[32 * (i + NrLanes) +: 32] = ldu_result_wdata_i[i][63:32];
    end
  end

  // ==============================================================
  // Buffer Write Control
  // ==============================================================

  always_comb begin
    buffer_push            = 2'b00;
    write_buffer_id_d      = write_buffer_id_q;
    ldu_result_gnt_o       = '0;
    ldu_result_final_gnt_o = '0;

    // Push if data is valid and the target FIFO is not full
    if (&ldu_result_req_i && ~buffer_full[write_buffer_id_q]) begin
      buffer_push[write_buffer_id_q] = 1'b1;
      ldu_result_gnt_o               = '1;
      ldu_result_final_gnt_o         = '1;
    end

    // Prepare the next round, switch to another buffer
    if (buffer_load_finished[write_buffer_id_q])
      write_buffer_id_d = ~write_buffer_id_q;
  end

  // ==============================================================
  // Buffer Read Control
  // ==============================================================

  // bc_invalidate_i is buffered in vmfpu
  assign bc_valid_o = ~buffer_empty[read_buffer_id_q] && ~bc_invalidate_i;
  assign bc_data_o  = buffer_dout[read_buffer_id_q];

  always_comb begin
    read_buffer_id_d = read_buffer_id_q;
    buffer_pop       = 2'b00;
    buffer_flush     = 2'b00;

    if (bc_ready_i && ~buffer_empty[read_buffer_id_q]) begin
      buffer_pop[read_buffer_id_q] = 1'b1;
    end

    if (bc_invalidate_i) begin
      buffer_flush[read_buffer_id_q] = 1'b1;
      read_buffer_id_d               = ~read_buffer_id_q;
    end
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      write_buffer_id_q <= 1'b0;
      read_buffer_id_q  <= 1'b0;
    end else begin
      write_buffer_id_q <= write_buffer_id_d;
      read_buffer_id_q  <= read_buffer_id_d;
    end
  end

  // =================================================================
  // Asserations
  // =================================================================

  if (MAX_BLEN % NrLanes != 0)
    $error("The maximum broadcast vector length must be a multiple of the number of laens.");

endmodule : bc_buffer
