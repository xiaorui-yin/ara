function compile_matmul(){
  rm matmul_opt/*.o
  rm matmul_opt/kernel/*.o
  sed -i "37c\(dim1, dim2, dim3) = (${1}, ${2}, ${3})" matmul_opt/script/gen_data.py
  make bin/matmul_opt config=${4}_lanes
  mv bin/matmul_opt bin/matmul_${1}x${2}x${3}_${4}_lanes
  mv bin/matmul_opt.dump bin/matmul_${1}x${2}x${3}_${4}_lanes.dump
}

function compile_matmul_old(){
  rm matmul/*.o
  rm matmul/kernel/*.o
  sed -i "37c\(dim1, dim2, dim3) = (${1}, ${2}, ${3})" matmul/script/gen_data.py
  make bin/matmul config=${4}_lanes
  mv bin/matmul bin/matmul_old_${1}x${2}x${3}_${4}_lanes
  mv bin/matmul.dump bin/matmul_old_${1}x${2}x${3}_${4}_lanes.dump
}

# ===============================================================================
# Square Matrix
# ===============================================================================

compile_matmul 32 32 32 4
compile_matmul 64 64 64 4
compile_matmul 128 128 128 4

compile_matmul 64 768 64 4
compile_matmul 128 768 64 4
compile_matmul 256 768 64 4
compile_matmul 512 768 64 4

# ===============================================================================
# Self Attention
# ===============================================================================

# Base
compile_matmul_old 64 768 64 4

# Sequence Length
compile_matmul_old 128 768 64 4
compile_matmul_old 256 768 64 4
compile_matmul_old 512 768 64 4

# Model Size
compile_matmul_old 64 1024 64 4
compile_matmul_old 64 1280 80 4

# Nr Lanes
compile_matmul_old 64 768 64 8
compile_matmul_old 64 768 64 16

# Sequence Length
compile_matmul 128 768 64 4
compile_matmul 256 768 64 4
compile_matmul 512 768 64 4

# Model Size
compile_matmul 64 1024 64 4
compile_matmul 64 1280 80 4


# ===============================================================================
# MLP
# ===============================================================================

# Base
compile_matmul_old 64 768 64 4

# Sequence Length
compile_matmul_old 128 768 64 4
compile_matmul_old 256 768 64 4
compile_matmul_old 512 768 64 4

# Model Size
compile_matmul_old 64 1024 64 4
compile_matmul_old 64 1280 80 4

# Nr Lanes
compile_matmul_old 64 768 64 8
compile_matmul_old 64 768 64 16

# Sequence Length
compile_matmul 128 768 64 4
compile_matmul 256 768 64 4
compile_matmul 512 768 64 4

# Model Size
compile_matmul 64 1024 64 4
compile_matmul 64 1280 80 4
