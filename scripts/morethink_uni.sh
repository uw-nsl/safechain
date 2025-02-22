#!/bin/bash

##############################################
# Configuration
##############################################
MODEL_PATH="$1"
TENSOR_PARALLEL_SIZE="$2" 
GEN_DEVICE="$3"
EVAL_DEVICE="$4"

RUN_PY="$5"


VLLM_PORT="8000"
LOG_FILE="vllm_server_$(date +%s%N).log"
API_KEY="token-abc123"
HOST="localhost"
##############################################
cd ..
# (Optional) Cleanup any old log
rm -f "$LOG_FILE"
##############################################
# 1) Start vLLM server in the background
##############################################
echo "Starting vLLM server..."
CUDA_VISIBLE_DEVICES="$GEN_DEVICE"  vllm serve "$MODEL_PATH" \
  --dtype "auto" \
  --port "$VLLM_PORT" \
  --host "$HOST" \
  --api-key "$API_KEY" \
  --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
  > "$LOG_FILE" 2>&1 &

# Get the process ID of the server
VLLM_PID=$!

echo "vLLM server started with PID $VLLM_PID. Logs at $LOG_FILE"

##############################################
# 2) Wait until the server is ready
##############################################
# -- Option A: Watch the output log for a specific line (example below)
# vLLM uses an internal Uvicorn server. Usually, once the app is ready,
# you might see "Application startup complete."
# or "INFO:     Uvicorn running on ..."

# If you'd like to watch for a specific line, you can do:
while ! grep -q "Application startup complete." "$LOG_FILE" 2>/dev/null; do
  sleep 3
done
echo "vLLM server appears to be up. Proceeding..."

##############################################
# 3) Execute the Python program
##############################################
echo "Running the Python script"
if [ "$RUN_PY" == "gen" ]; then
  echo "Running morethink"
  RUN_FILE="resp_gen.py"

else
  RUN_FILE="pipeline.py"
fi

# choose the evaluation setup as you want

CUDA_VISIBLE_DEVICES="$EVAL_DEVICE" python "$RUN_FILE" --model "$MODEL_PATH" --data wildjailbreak --prompt morethink

CUDA_VISIBLE_DEVICES="$EVAL_DEVICE" python "$RUN_FILE" --model "$MODEL_PATH" --data strongreject --prompt morethink

CUDA_VISIBLE_DEVICES="$EVAL_DEVICE" python "$RUN_FILE" --model "$MODEL_PATH" --data wildjailbreak_small --prompt morethink  --temperature 0.6 --topp 0.95 --repeat_n 5

CUDA_VISIBLE_DEVICES="$EVAL_DEVICE" python "$RUN_FILE" --model "$MODEL_PATH" --data strongreject_small --prompt morethink  --temperature 0.6 --topp 0.95 --repeat_n 5


##############################################
# 4) Kill the server process once done
##############################################
echo "Stopping the vLLM server..."
kill "$VLLM_PID"

# (Optional) Wait for it to actually shut down
wait "$VLLM_PID" 2>/dev/null
echo "vLLM server stopped. Script finished."
