source script/compile_kernel.sh

#############################
# Dropout
#############################

compile_dropout 64 768 4
compile_dropout 64 768 8
compile_dropout 64 768 16

compile_dropout 128 768 8
compile_dropout 256 768 16
compile_dropout 512 768 16

compile_dropout 64 1024 4
compile_dropout 64 1280 4

#############################
# ReLU
#############################

compile_relu 64 768 4
compile_relu 64 768 8
compile_relu 64 768 16

compile_relu 128 768 8
compile_relu 256 768 16
compile_relu 512 768 16

compile_relu 64 1024 4
compile_relu 64 1280 4

#############################
# LayerNorm
#############################

compile_layernorm 64 768 0 4
compile_layernorm 64 768 0 8
compile_layernorm 64 768 0 16

compile_layernorm 128 768 0 8
compile_layernorm 256 768 0 16
compile_layernorm 512 768 0 16

compile_layernorm 64 1024 0 4
compile_layernorm 64 1280 0 4

compile_layernorm 64 768 1 4
compile_layernorm 64 768 1 8
compile_layernorm 64 768 1 16

compile_layernorm 128 768 1 8
compile_layernorm 256 768 1 16
compile_layernorm 512 768 1 16

compile_layernorm 64 1024 1 4
compile_layernorm 64 1280 1 4

#############################
# Softmax
#############################

compile_softmax 64 64 0 4
compile_softmax 64 64 0 8
compile_softmax 64 64 0 16

compile_softmax 128 128 0 8
compile_softmax 256 256 0 16
compile_softmax 512 512 0 16

compile_softmax 64 64 1 4
compile_softmax 64 64 1 8
compile_softmax 64 64 1 16

compile_softmax 128 128 1 8
compile_softmax 256 256 1 16
compile_softmax 512 512 1 16
