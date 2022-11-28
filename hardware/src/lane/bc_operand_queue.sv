module bc_operand_queue import ara_pkg::*; (
    input  logic                           clk_i,
    input  logic                           rst_ni,
    // Interface with the previous lane / Broadcast Buffer
    output logic                           bc_ready_o,
    input  elen_t                          bc_data_i,
    input  logic                           bc_valid_i,
    // Interface with the next lane
    input  logic                           bc_ready_i,
    output elen_t                          bc_data_o,
    output logic                           bc_valid_o,
    // Interface with VMFPU (TODO: INT support)
    input  logic                           bc_vmfpu_ready_i,
    output elen_t                          bc_vmfpu_data_o,
    output logic                           bc_vmfpu_valid_o
    // // First lane only
    // input  logic                           bc_vmfpu_invalidate_i,
    // output logic                           bc_invalidate_o
);

  logic  bc_op_buffer_push, bc_op_buffer_pop;
  logic  bc_op_buffer_full, bc_op_buffer_empty;

  logic data_used_d, data_used_q;

  fifo_v3 #(
    .DEPTH(2     ),
    .dtype(elen_t)
  ) i_bc_op_buffer(
    .clk_i     (clk_i              ),
    .rst_ni    (rst_ni             ),
    .flush_i   (1'b0               ),
    .testmode_i(1'b0               ),
    .data_i    (bc_data_i            ),
    .push_i    (bc_op_buffer_push  ),
    .full_o    (bc_op_buffer_full  ),
    .data_o    (bc_data_o    ),
    .pop_i     (bc_op_buffer_pop   ),
    .empty_o   (bc_op_buffer_empty ),
    .usage_o   (/* Unused*/        )
  );

  assign bc_vmfpu_data_o = bc_data_i;
  assign bc_valid_o      = ~bc_op_buffer_empty;

  always_comb begin
    bc_ready_o = 1'b0;
    bc_op_buffer_push = 1'b0;
    bc_op_buffer_pop = 1'b0;

    bc_vmfpu_valid_o = 1'b0;//bc_valid_i;

    data_used_d = data_used_q;

    // if (bc_valid_i && bc_vmfpu_ready_i) begin
    //   if (~bc_op_buffer_full) begin
    //     // Ackownledge the data
    //     bc_ready_o = 1'b1;
    //     bc_op_buffer_push = 1'b1;
    //   end else begin
    //     // The next lane is not ready
    //     // Mark this data as used
    //     data_used_d = 1'b1;
    //   end
    // end
    // if (data_used_q) begin
    //   bc_vmfpu_valid_o = 1'b0;
    //   // Clear if the old data is pushed
    //   if (~bc_op_buffer_full) begin
    //     data_used_d = 1'b0;
    //     bc_ready_o = 1'b1;
    //     bc_op_buffer_push = 1'b1;
    //   end
    // end
    if (bc_valid_i && ~bc_op_buffer_full && ~data_used_q) begin
      bc_vmfpu_valid_o = 1'b1;
      bc_op_buffer_push = 1'b1;
      if (bc_vmfpu_ready_i) begin
        // Ackownledge the data
        bc_ready_o = 1'b1;
      end else begin
        // The next lane is not ready
        // Mark this data as used
        data_used_d = 1'b1;
      end
    end

    if (data_used_q) begin
      bc_vmfpu_valid_o = 1'b1;
      // Clear if the old data is pushed
      if (bc_vmfpu_ready_i) begin
        data_used_d = 1'b0;
        bc_ready_o = 1'b1;
      end
    end

    if (~bc_op_buffer_empty && bc_ready_i)
      bc_op_buffer_pop = 1'b1;
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if(~rst_ni) begin
      data_used_q <= 1'b0;
    end else begin
      data_used_q <= data_used_d;
    end
  end

endmodule : bc_operand_queue