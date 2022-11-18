module bc_operand_queue import ara_pkg::*; import matmul_pkg::*; (
    input  logic                           clk_i,
    input  logic                           rst_ni,
    // Interface with the previous lane
    output logic                           bc_ready_o,
    input  elen_t                          bc_data_i,
    output logic                           bc_valid_i,
    // Interface with the next lane
    input  logic                           bc_ready_i,
    output elen_t                          bc_data_o,
    output logic                           bc_valid_o,
    // Interface with VMFPU (TODO: INT support)
    input  logic                           bc_vmfpu_ready_i,
    output elen_t                          bc_vmfpu_data_o,
    output logic                           bc_vmfpu_valid_o
);

  // =================================================================
  // Local Broadcast Operand Buffer
  // =================================================================

  logic  bc_op_buffer_push, bc_op_buffer_pop;
  logic  bc_op_buffer_full, bc_op_buffer_empty;

  fifo_v3 #(
    .DEPTH(2     ),
    .dtype(elen_t)
  ) i_bc_op_buffer(
    .clk_i     (clk_i              ),
    .rst_ni    (rst_ni             ),
    .flush_i   (1'b0               ),
    .testmode_i(1'b0               ),
    .data_i    (bc_data_i          ),
    .push_i    (bc_op_buffer_push  ),
    .full_o    (bc_op_buffer_full  ),
    .data_o    (bc_vmfpu_data_o    ),
    .pop_i     (bc_op_buffer_pop   ),
    .empty_o   (bc_op_buffer_empty ),
    .usage_o   (/* Unused*/        )
  );

  // =================================================================
  // Broadcast Data Registers
  // =================================================================

  elen_t bc_data;
  logic  bc_valid, bc_ready;
  logic  bc_data_stored_d, bc_data_stored_q;

  spill_register #(
    .T(elen_t)
  ) i_bc_spill_register (
    .clk_i  (clk_i     ),
    .rst_ni (rst_ni    ),
    .valid_i(bc_valid_i),
    .ready_o(bc_ready_o),
    .data_i (bc_data_i ),
    .valid_o(bc_valid  ),
    .ready_i(bc_ready  ),
    .data_o (bc_data   )
  );

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      bc_data_stored_q <= 1'b0;
    end else begin
      bc_data_stored_q <= bc_data_stored_d;
    end
  end

  // =================================================================
  // Buffer Write Logic
  // =================================================================

  always_comb begin
    local_bc_buffer_push = 1'b0;
    bc_data_stored_d     = bc_data_stored_q;

    // To the next lane
    bc_data_o            = '0;
    bc_valid_o           = 1'b0;

    bc_ready             = 1'b0;

    // Receive a new data from the previous lane
    if (bc_valid && ~bc_op_buffer_full && ~bc_data_stored_q) begin
      local_bc_buffer_push = 1'b1;
      bc_data_stored_d     = 1'b1;
      bc_data_o            = bc_data;
      bc_valid_o           = 1'b1;

      // If the old data is acknowledged by the next lane
      // Acknowledge the new data
      if (bc_ready_i) begin
        bc_ready         = 1'b1;
        bc_data_stored_d = 1'b0;
      end
    end

    if (bc_data_stored_q && bc_ready_i) begin
      bc_ready         = 1'b1;
      bc_data_stored_d = 1'b0;
    end
  end

  // =================================================================
  // Buffer Read Logic
  // =================================================================

  assign bc_vmfpu_valid_o = ~bc_op_buffer_empty;
  always_comb begin
    local_bc_buffer_pop  = 1'b0;
    if (bc_vmfpu_ready_i) local_bc_buffer_pop = 1'b1;
  end

endmodule : bc_operand_queue
