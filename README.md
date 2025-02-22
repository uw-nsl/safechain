<!-- # SafeChain -->
<!-- ## Safety of Language Models with Long Chain-of-Thought Reasoning Capabilities -->
<section class="hero" style="text-align: center;">
  <div class="hero-body">
    <div class="container is-max-desktop">
      <div class="columns is-centered">
        <div class="column has-text-centered">
          <h1 class="title is-1 publication-title">
        SafeChain: Safety of Language Models with Long Chain-of-Thought Reasoning Capabilities
          </h1>
          <div class="is-size-5 publication-authors">
            <!-- Paper authors -->
            <span class="author-block">
              <a href="http://fqjiang.work/" target="_blank">Fengqing Jiang</a><sup>1</sup>,
            </span>
            <span class="author-block">
              <a href="https://zhangchenxu.com/" target="_blank">Zhangchen Xu</a><sup>1</sup>,
            </span>
            <span class="author-block">
              <a href="https://yuetl9.github.io/" target="_blank">Yuetai Li</a><sup>1</sup>,
            </span>
            <span class="author-block">
              <a href="https://luyaoniu.github.io/" target="_blank">Luyao Niu</a><sup>1</sup>,
            </span>
            <br>
            <span class="author-block">
              <a href="https://zhenxianglance.github.io/" target="_blank">Zhen Xiang</a><sup>2</sup>,
            </span>
            <span class="author-block">
              <a href="https://yuchenlin.xyz/" target="_blank">Bill Yuchen Lin</a><sup>1</sup>,
            </span>
            <span class="author-block">
              <a href="https://aisecure.github.io/" target="_blank">Bo Li</a><sup>3</sup>,
            </span>
            <span class="author-block">
              <a href="https://labs.ece.uw.edu/nsl/faculty/radha/" target="_blank">Radha Poovendran</a><sup>1</sup>,
            </span>
          </div>
          <div class="is-size-5 publication-authors">
            <span class="author-block">
              <sup>1</sup>&nbsp;University of Washington &nbsp;&nbsp; 
              <sup>2</sup>&nbsp;University of Georgia &nbsp;&nbsp; 
              <sup>3</sup>&nbsp;University of Chicago
            </span>
          </div>
          <p style="color: red; font-weight: bold;">
            ⚠️ Warning: This paper contains model outputs that may be considered offensive.
          </p>
          <div class="column has-text-centered">
            <div class="publication-links">
              <!-- Arxiv PDF link -->
              <span class="link-block">
                <a href="https://arxiv.org/pdf/2502.12025" target="_blank"
                   class="external-link button is-normal is-rounded is-dark">
                  <span class="icon">
                    <i class="fas fa-file-pdf"></i>
                  </span>
                  <span><u>Paper</u></span>
                </a>
              </span>
              <!-- Github link -->
              <span class="link-block">
                <a href="https://github.com/YOUR REPO HERE" target="_blank"
                   class="external-link button is-normal is-rounded is-dark">
                  <span class="icon">
                    <i class="fab fa-github"></i>
                  </span>
                  <span><u>Project Page</u></span>
                </a>
              </span>
              <!-- Hugging Face link -->
              <span class="link-block">
                <a href="https://huggingface.co/datasets/UWNSL/SafeChain" target="_blank"
                   class="external-link button is-normal is-rounded is-dark">
                  <span><u>Dataset</u></span>
                </a>
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</section>


## News
- [2025/02/21] We released our code source.


## How to use our project

Before running our code, 
#### Build the environment
At the `safechain` dir, run
```shell
bash scripts/build_env.sh safechain
```

#### Some configuration setups
- Add your HF token to access models
- Update the model config (e.g., the number of GPUs for each model/add new models) at `config.py`
- If you are running models with API access, make sure to add the endpoint setup in `utils_model.py`. For using different model endpoint, also switch in the `config.py` (refer to the setup for `DeepSeek R1`).

#### Run our code

Our pipeline includes two steps, generate model response and evaluation.

To run the step seperately, you can try the following command. The 
```python
python resp_gen.py
```
# Command-Line Arguments Help

Below is a summary of the command-line arguments provided by the script, along with their descriptions and default values.

| **Argument**       | **Type** | **Default**                              | **Choices**         | **Description**                                                                                               |
|--------------------|----------|------------------------------------------|---------------------|---------------------------------------------------------------------------------------------------------------|
| `--model`          | `str`    | `RSM_LIST[0]` (first entry in RSM_LIST)  | entry in `RSM_LIST`          | The model name to use, selected from the list of available models in `RSM_LIST` in `config.py`.                              |
| `--data`           | `str`    | `strongreject`                           | entry in `EVAL_DATA`         | The dataset to use for evaluation, selected from `EVAL_DATA` in `config.py`.                                                 |
| `--prompt`         | `str`    | `normal`                                 | [`normal`, `zerothink`, `lessthink`, `overthink`]            | Setup for generation input (e.g., type of prompt).                                                            |
| `--system`         | `bool`   | `DEFAULT_GEN_CONFIG['system']`           | *None*             | Whether to override system prompt setup in `config.py`.                                                                  |
| `--temperature`    | `float`  | `DEFAULT_GEN_CONFIG['temperature']`      | *None*             | Sampling temperature for text generation (higher means more randomness).                                     |
| `--topp`           | `float`  | `DEFAULT_GEN_CONFIG['topp']`             | *None*             | Nucleus sampling probability (top-p).                                                                         |
| `--topk`           | `int`    | `DEFAULT_GEN_CONFIG['topk']`             | *None*             | Top-k sampling parameter.                                                                                    |
| `--max_tokens`     | `int`    | `DEFAULT_GEN_CONFIG['max_tokens']`       | *None*             | Maximum number of tokens to generate.                                                                        |
| `--repeat_n`       | `int`    | `DEFAULT_GEN_CONFIG['repeat_n']`         | *None*             | Number of samples to generate per prompt input.                                                              |
| `--n`              | `int`    | `-1`                                     | *None*             | Number of samples to use. Use `-1` to include all available samples.                                         |
| `--start_idx`      | `int`    | `0`                                      | *None*             | The starting index from which to use the dataset samples.                                                    |
| `--port`           | `int`    | `8000`                                   | *None*             | Port number (or used as an identifier in file naming, depending on your use case).                           |
| `--think_budget`   | `int`    | `10000`                                  | *None*             | Thinking Budget for internal "thinking" or hidden reasoning.                                    |
| `--enforce_num`    | `int`    | `10`                                     | *None*             | Enforced time limit for `MoreThink` setup.                                                 |

---

## Usage Example

```bash
python resp_gen.py \
    --model myModel \
    --data myDataset \
    --prompt "custom_prompt" \
    --system True \
    --temperature 0.8 \
    --topp 0.9 \
    --topk 40 \
    --max_tokens 120 \
    --repeat_n 5 \
    --n 100 \
    --start_idx 10 \
    --port 8080 \
    --think_budget 5000 \
    --enforce_num 20
```


And with the output file, run the following command:
```
python resp_eval.py --file file_name
```

The experiment can also be running in end-to-end manner, replacing `resp_gen.py` with `pipeline.py`


##### Running `MoreThink` Experiment
We provide an efficient implementation for `MoreThink` setup. You must first boost the vllm server then running the `resp_gen.py`. We also provide a script to run this setup. 

Under `scripts` dir, run

```
bash morethink_uni.sh  MODEL_PATH TENSOR_PARALLEL_SIZE GEN_DEVICE EVAL_DEVICE RUN_PY 
```
If `RUN_PY` is `gen`, the script will not run evaluation after response generation, it can help running experiment if you do not have enough GPU devices (e.g., only 1 GPU).







