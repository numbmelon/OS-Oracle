export PYTHONPATH="$(pwd):${PYTHONPATH}"

PROC_PER_NODE=8
NODE_COUNT=1
NODE_RANK=0
MASTER_ADDR=127.0.0.1

# support qwen2.5vl and qwen3vl series, openai series
CRITIC_MODEL_PATH="your_model_path"


torchrun \
  --nproc_per_node=${PROC_PER_NODE} \
  --nnodes=${NODE_COUNT} \
  --node_rank=${NODE_RANK} \
  --master_addr=${MASTER_ADDR} \
  --master_port=29613 \
  eval_bench.py \
  --critic_backend qwen \
  --critic_model_path "$CRITIC_MODEL_PATH"

# set 'critic_backend' to 'oai' if testing gpt or claude 
