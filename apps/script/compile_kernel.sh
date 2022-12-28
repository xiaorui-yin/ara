function compile_dropout(){
  rm dropout_t/*.o
  rm dropout_t/kernel/*.o
  sed -i "48c\(row, col) = (${1}, ${2})" dropout_t/script/gen_data.py
  make bin/dropout_t config=${3}_lanes
  mv bin/dropout_t bin/droupout_${1}x${2}_${3}_lanes
}

function compile_relu(){
  rm relu/*.o
  rm relu/kernel/*.o
  sed -i "43c\(row, col) = (${1}, ${2})" relu/script/gen_data.py
  make bin/relu config=${3}_lanes
  mv bin/relu bin/droupout_${1}x${2}_${3}_lanes
}

function compile_layernorm(){
  rm layernorm/*.o
  rm layernorm/kernel/*.o
  sed -i "34c\(row, col, transpose) = (${1}, ${2}, ${3})" layernorm/script/gen_data.py
  make bin/layernorm config=${4}_lanes
  mv bin/layernorm bin/layernorm_${1}x${2}_${3}_${4}_lanes
}

function compile_softmax(){
  rm softmax_opt/*.o
  rm softmax_opt/kernel/*.o
  sed -i "70c\(row, col, transpose) = (${1}, ${2}, ${3})" softmax/script/gen_data.py
  make bin/softmax_opt config=${4}_lanes
  mv bin/softmax_opt bin/softmax_opt_${1}x${2}_${3}_${4}_lanes
}

function compile_self_attention(){
  rm self_attention/*.o
  rm self_attention/kernel/*.o
  sed -i "41c\(n, d_model, dk, transpose) = (${1}, ${2}, ${3}, ${4})" self_attention/script/gen_data.py
  make bin/self_attention config=${5}_lanes
  mv bin/self_attention bin/self_attention_${1}x${2}_${3}_${4}_${5}_lanes
}

function compile_multihead_attention(){
  rm multihead_attention/*.o
  rm multihead_attention/kernel/*.o
  sed -i "47c\(n, d_model, h, transpose) = (${1}, ${2}, ${3}, ${4})" multihead_attention/script/gen_data.py
  make bin/multihead_attention config=${5}_lanes
  mv bin/multihead_attention bin/multihead_attention_${1}x${2}_${3}_${4}_${5}_lanes
}

function compile_feed_forward(){
  rm feed_forward/*.o
  rm feed_forward/kernel/*.o
  sed -i "46c\(n, d_model, transpose) = (${1}, ${2}, ${3})" feed_forward/script/gen_data.py
  make bin/feed_forward config=${4}_lanes
  mv bin/feed_forward bin/feed_forward_${1}x${2}_${3}_${4}_lanes
}
