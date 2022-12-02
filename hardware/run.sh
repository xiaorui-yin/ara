function run() {
  a=$1
  b=$2
  c=$3
  d=$4
  rm -rf build
  make simc app=matmul_${a}x${b}x${c}_${d}_lanes config=${d}_lanes > log/matmul_${a}x${b}x${c}_${d}_lanes.log
}

function run_old() {
  a=$1
  b=$2
  c=$3
  d=$4
  rm -rf build
  make simc app=matmul_old_${a}x${b}x${c}_${d}_lanes config=${d}_lanes > log/matmul_old_${a}x${b}x${c}_${d}_lanes.log
}

run 32 32 32 4
run 64 64 64 4
run 128 128 128 4

# ===============================================================================
# Self Attention
# ===============================================================================

# # Base
# run_old 64 768 64 4
# 
# # Sequence Length
# run_old 128 768 64 4
# run_old 256 768 64 4
# run_old 512 768 64 4
# 
# # Model Size
# run_old 64 1024 64 4
# run_old 64 1280 80 4
# 
# # Nr Lanes 
# run_old 64 768 64 8
# run_old 64 768 64 16

# Sequence Length
run 128 768 64 4
run 256 768 64 4
run 512 768 64 4

# Model Size
run 64 1024 64 4
#run 64 1280 80 4
