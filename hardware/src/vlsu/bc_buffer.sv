// Author: Xioarui Yin
// Description:
// Broadcast buffer receives the data read by the load unit from L2 cache,
// and send it to the first lane (lane0)
// Only support fp32

module bc_buffer import ara_pkg::*; import rvv_pkg::*; #(
    parameter  int  unsigned NrLanes = 0,
    parameter  int  unsigned MAX_BLEN = 32,

    // Dependant parameters. DO NOT CHANGE!
    localparam int  unsigned DataWidth    = $bits(elen_t),
    localparam type          strb_t       = logic [DataWidth/8-1:0]
  ) (
    input  logic                            clk_i,
    input  logic                            rst_ni,
    // Interface with the load unit
    input  strb_t             [NrLanes-1:0] ldu_result_be_i,
    input  logic              [NrLanes-1:0] ldu_result_req_i,
    input  elen_t             [NrLanes-1:0] ldu_result_wdata_i,
    output logic              [NrLanes-1:0] ldu_result_gnt_o,
    output logic              [NrLanes-1:0] ldu_result_final_gnt_o,
    input  logic                            load_complete_i,
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
  logic [$clog2(NrLanes):0]      buffer_valid_cnt;
  

  for (genvar i = 0; i < 2; i++) begin: gen_re_readable_buffer
    re_readable_fifo #(
      .DEPTH(MAX_BLEN/2), // TODO: scale to the read data width
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
      .valid_cnt_i    (buffer_valid_cnt        ),
      .push_i         (buffer_push[i]          ),
      .data_o         (buffer_dout[i]          ),
      .pop_i          (buffer_pop[i]           ),
      .load_complete_i(load_complete_i         ),
      .usage_o        (/* unused */            )
    );
  end: gen_re_readable_buffer

  // =============================================================
  // Input Data Serialization
  // =============================================================
  logic [NrLanes-1:0]       valid_element;

  // Count how many valid elements
  popcount #(
    .INPUT_WIDTH (NrLanes)
  ) i_popcount (
    .data_i    (valid_element        ),
    .popcount_o(buffer_valid_cnt     )
  );

  always_comb begin
    buffer_din    = '0;
    valid_element = '0;
    for (int i = 0; i < NrLanes; i++) begin
      // TODO: FP32 only

      // valid 64-bit data
      if (i % 2 == 0) begin
        valid_element[i/2] = ldu_result_be_i[i][0];
        valid_element[NrLanes/2 + i/2] = ldu_result_be_i[i][4];
      end

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

      // Prepare the next round, switch to another buffer
      if (load_complete_i) write_buffer_id_d = ~write_buffer_id_q;
    end
  end

  // ==============================================================
  // Buffer Read Control
  // ==============================================================

  assign bc_valid_o = ~buffer_empty[read_buffer_id_q];
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
