// Copyright 2018 ETH Zurich and University of Bologna.
// Copyright and related rights are licensed under the Solderpad Hardware
// License, Version 0.51 (the "License"); you may not use this file except in
// compliance with the License. You may obtain a copy of the License at
// http://solderpad.org/licenses/SHL-0.51. Unless required by applicable law
// or agreed to in writing, software, hardware and materials distributed under
// this License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

// Author: Florian Zaruba <zarubaf@iis.ee.ethz.ch>
// Modified by Xiaorui Yin

// Description:
// Data will be retained after reading, such that it can be read multiple times. Suuport data width conversion, but WR_DATA_WIDTH > RD_DATA_WIDTH
// Use flush_i to invalidate data and load new data.

module re_readable_fifo #(
    parameter int unsigned WR_DATA_WIDTH   = 32,   // default data width if the fifo is of type logic
    parameter int unsigned RD_DATA_WIDTH   = 32,   // default data width if the fifo is of type logic
    parameter int unsigned DEPTH           = 8,    // depth can be arbitrary from 0 to 2**32, read depth
    parameter type wr_dtype                = logic [WR_DATA_WIDTH-1:0],
    parameter type rd_dtype                = logic [RD_DATA_WIDTH-1:0],
    // DO NOT OVERWRITE THIS PARAMETER
    parameter int unsigned ADDR_DEPTH   = (DEPTH > 1) ? $clog2(DEPTH) : 1,
    parameter int unsigned RATIO        = WR_DATA_WIDTH / RD_DATA_WIDTH
)(
    input  logic     clk_i,            // Clock
    input  logic     rst_ni,           // Asynchronous reset active low
    input  logic     flush_i,          // flush the queue
    input  logic     testmode_i,       // test_mode to bypass clock gating
    // status flags
    output logic     full_o,           // queue is full
    output logic     empty_o,          // queue is empty
    output logic     [ADDR_DEPTH-1:0] usage_o,  // fill pointer
    // as long as the queue is not full we can push new data
    input  wr_dtype  data_i,           // data to push into the queue
    input  logic     push_i,           // data is valid and can be pushed to the queue
    input  logic     final_push_i,     // TODO
    // as long as the queue is not empty we can pop new elements
    output rd_dtype  data_o,           // output data
    input  logic     pop_i             // pop head from queue
);
    // local parameter
    // FIFO depth - handle the case of pass-through, synthesizer will do constant propagation
    localparam int unsigned FifoDepth = (DEPTH > 0) ? DEPTH : 1;
    // clock gating control
    logic gate_clock;
    // pointer to the read and write section of the queue
    logic [ADDR_DEPTH - 1:0] read_pointer_n, read_pointer_q, write_pointer_n, write_pointer_q;
    // keep a counter to keep track of the current queue status
    // this integer will be truncated by the synthesis tool
    logic [ADDR_DEPTH:0] status_cnt_n, status_cnt_q;
    // actual memory
    rd_dtype [FifoDepth - 1:0] mem_n, mem_q;

    assign usage_o = status_cnt_q[ADDR_DEPTH-1:0];

    // TODO
    logic load_complete_d, load_complete_q;
  
    logic [ADDR_DEPTH:0] write_data_cnt_d, write_data_cnt_q;

    if (DEPTH == 0) begin : gen_pass_through
        assign empty_o     = ~push_i;
        assign full_o      = ~pop_i;
    end else begin : gen_fifo
        assign full_o       = (load_complete_q) ? 1'b1 : (status_cnt_q == FifoDepth[ADDR_DEPTH:0]);
        assign empty_o      = (load_complete_q) ? 1'b0 : (status_cnt_q == 0);
    end
    // status flags

    // read and write queue logic
    always_comb begin : read_write_comb
        // default assignment
        read_pointer_n  = read_pointer_q;
        write_pointer_n = write_pointer_q;
        status_cnt_n    = status_cnt_q;
        data_o          = (DEPTH == 0) ? data_i : mem_q[read_pointer_q];
        mem_n           = mem_q;
        gate_clock      = 1'b1;

        write_data_cnt_d = write_data_cnt_q;
        load_complete_d  = (final_push_i) ? 1'b1 : load_complete_q;

        if (~load_complete_q) begin
          // push a new element to the queue
          if (push_i && ~full_o) begin
              // push the data onto the queue
              for (int i = 0; i < RATIO; i++)
                mem_n[write_pointer_q + i] = data_i[i * RD_DATA_WIDTH +: RD_DATA_WIDTH];
              /* mem_n[write_pointer_q] = data_i; */

              // un-gate the clock, we want to write something
              gate_clock = 1'b0;
              // increment the write counter
              if (write_pointer_q == FifoDepth[ADDR_DEPTH-1:0] - 1)
                  write_pointer_n = '0;
              else
                  /* write_pointer_n = write_pointer_q + 1; */
                  write_pointer_n = write_pointer_q + RATIO;

              // increment the overall counter
              /* status_cnt_n    = status_cnt_q + 1; */
              /* write_data_cnt_d = write_data_cnt_q + 1; */
              status_cnt_n    = status_cnt_q + RATIO;
              write_data_cnt_d = write_data_cnt_q + RATIO;
          end

          if (pop_i && ~empty_o) begin
              // read from the queue is a default assignment
              // but increment the read pointer...
              if (read_pointer_n == FifoDepth[ADDR_DEPTH-1:0] - 1)
                  read_pointer_n = '0;
              else
                  read_pointer_n = read_pointer_q + 1;
              // ... and decrement the overall count
              status_cnt_n   = status_cnt_q - 1;
          end

          // keep the count pointer stable if we push and pop at the same time
          if (push_i && pop_i &&  ~full_o && ~empty_o)
              status_cnt_n   = status_cnt_q;
        end else begin
          // read only state
          if (pop_i) begin
            if (read_pointer_n == write_data_cnt_q - 1)
                read_pointer_n = '0;
            else
                read_pointer_n = read_pointer_q + 1;
          end
        end
    end

    // sequential process
    always_ff @(posedge clk_i or negedge rst_ni) begin
        if(~rst_ni) begin
            read_pointer_q  <= '0;
            write_pointer_q <= '0;
            status_cnt_q    <= '0;
            load_complete_q <= 1'b0;
            write_data_cnt_q <= '0;
        end else begin
            if (flush_i) begin
                read_pointer_q  <= '0;
                write_pointer_q <= '0;
                status_cnt_q    <= '0;
                load_complete_q <= 1'b0;
                write_data_cnt_q <= '0;
             end else begin
                read_pointer_q  <= read_pointer_n;
                write_pointer_q <= write_pointer_n;
                status_cnt_q    <= status_cnt_n;
                load_complete_q <= load_complete_d;
                write_data_cnt_q <= write_data_cnt_d;
            end
        end
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if(~rst_ni) begin
            mem_q <= '0;
        end else if (!gate_clock) begin
            mem_q <= mem_n;
        end
    end

// pragma translate_off
`ifndef VERILATOR
    initial begin
        assert (DEPTH > 0)             else $error("DEPTH must be greater than 0.");
    end

    full_write : assert property(
        @(posedge clk_i) disable iff (~rst_ni) (full_o |-> ~push_i))
        else $fatal (1, "Trying to push new data although the FIFO is full.");

    empty_read : assert property(
        @(posedge clk_i) disable iff (~rst_ni) (empty_o |-> ~pop_i))
        else $fatal (1, "Trying to pop data although the FIFO is empty.");

    invalide_ratio: assert (RATIO == Nr_Lanes)
        else $error("The data width ratio must be the number of lanes.")  
`endif
// pragma translate_on

endmodule // re_readable_fifo
