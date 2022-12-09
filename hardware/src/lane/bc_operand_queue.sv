// Copyright 2021 ETH Zurich and University of Bologna.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51
//
// Author: Xiaorui Yin <yinx@student.ethz.ch>

module bc_operand_queue import ara_pkg::*; #(
    parameter  int          Lane0      = 0,
    parameter  int unsigned NrLanes    = 0
  ) (
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
  );

  elen_t bc_data;
  logic  bc_valid, bc_ready;

  if (!Lane0) begin : gen_bc_op_fifo
    stream_fifo #(
      .DEPTH       (2               ),
      .T           (elen_t          )
    ) i_bc_op_fifo(
      .clk_i       (clk_i           ),
      .rst_ni      (rst_ni          ),
      .flush_i     (1'b0            ),
      .testmode_i  (1'b0            ),
      .data_i      (bc_data_i       ),
      .valid_i     (bc_valid_i      ),
      .ready_o     (bc_ready_o      ),
      .data_o      (bc_data         ),
      .valid_o     (bc_valid        ),
      .ready_i     (bc_ready        ),
      .usage_o     (/* Unused */    )
    );
  end else begin : gen_fall_through
    assign bc_data    = bc_data_i;
    assign bc_valid   = bc_valid_i;
    assign bc_ready_o = bc_ready;
  end

  logic  data_used_d, data_used_q;

  assign bc_vmfpu_data_o  = bc_data;
  assign bc_data_o        = bc_data;

  always_comb begin
    bc_vmfpu_valid_o = bc_valid;
    bc_valid_o       = 1'b0;
    bc_ready         = 1'b0;

    data_used_d      = data_used_q;

    // The FPU is ready
    if (bc_valid && bc_vmfpu_ready_i && ~data_used_q) begin
      // The next lane is not ready (FIFO full)
      if (~bc_ready_i) begin
        // Don't ackownledge, but mark this data as used
        data_used_d = 1'b1;
      end else begin
        bc_valid_o  = 1'b1;
        bc_ready    = 1'b1;
      end
    end

    if (data_used_q) begin
      // The data is used
      bc_vmfpu_valid_o = 1'b0;
      // Clear if the next lane is ready
      if (bc_ready_i) begin
        data_used_d = 1'b0;
        bc_ready    = 1'b1;
        bc_valid_o  = 1'b1;
      end
    end
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if(~rst_ni) begin
      data_used_q <= 1'b0;
    end else begin
      data_used_q <= data_used_d;
    end
  end

endmodule : bc_operand_queue
