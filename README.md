<h1 align='center' style="text-align:center; font-weight:bold; font-size:2.0em;letter-spacing:2.0px;"> SafeChain: Safety of Language Models with Long Chain-of-Thought Reasoning Capabilities </h1>
<!-- Authors -->
<p align="center">
    <!-- Use &nbsp; for spacing, <sup> for affiliations, and style="text-decoration: none;" for link decoration -->
    <a href="https://fqjiang.work/" target="_blank" style="text-decoration: none;">Fengqing Jiang<sup>1</sup></a>&nbsp;,&nbsp;
    <a href="https://zhangchenxu.com/" target="_blank" style="text-decoration: none;">Zhangchen Xu<sup>1</sup></a>&nbsp;,&nbsp;
    <a href="https://yuetl9.github.io/" target="_blank" style="text-decoration: none;">Yuetai Li<sup>1</sup></a>&nbsp;,&nbsp;
    <a href="https://luyaoniu.github.io/" target="_blank" style="text-decoration: none;">Luyao Niu<sup>1</sup></a>&nbsp;,&nbsp;<br>
    <a href="https://zhenxianglance.github.io/" target="_blank" style="text-decoration: none;">Zhen Xiang<sup>2</sup></a>&nbsp;,&nbsp;
    <a href="https://yuchenlin.xyz/" target="_blank" style="text-decoration: none;">Bill Yuchen Lin<sup>1</sup></a>&nbsp;,&nbsp;
    <a href="https://aisecure.github.io/" target="_blank" style="text-decoration: none;">Bo Li<sup>3</sup></a>&nbsp;,&nbsp;
    <a href="https://labs.ece.uw.edu/nsl/faculty/radha/" target="_blank" style="text-decoration: none;">Radha Poovendran<sup>1</sup></a>
    <br/><br/>
    <sup>1</sup>University of Washington&nbsp;&nbsp;&nbsp;
    <sup>2</sup>University of Georgia&nbsp;&nbsp;&nbsp;
    <sup>3</sup>University of Chicago
</p>
<!-- Warning -->
<p align="center" style="color: red;">
    <b><em>Warning: This paper contains model outputs that may be considered offensive.</em></b>
</p>


<!-- Links (Paper, GitHub, Dataset) -->
<p align="center" style="font-size: 1.2em;">
    <b>
        <a href="https://arxiv.org/pdf/2502.12025" target="_blank" style="text-decoration: none;">[Paper]</a>
    </b>
    &nbsp;&nbsp;
    <b>
        <a href="https://safe-chain.github.io/" target="_blank" style="text-decoration: none;">[Project Page]</a>
    </b>
    &nbsp;&nbsp;
    <b>
        <a href="https://huggingface.co/datasets/UWNSL/SafeChain" target="_blank" style="text-decoration: none;">[Dataset]</a>
    </b>
</p>

![](asset/safechain.jpg) 

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

Example:
```
bash morethink_uni.sh deepseek-ai/DeepSeek-R1-Distill-Llama-70B 4  "0,1,2,3" "2" "gen"
```


## Citation

If you find our work useful, please consider citing our paper:

```bibtex
@article{jiang2025safechain,
  title={SafeChain: Safety of Language Models with Long Chain-of-Thought Reasoning Capabilities},
  author={Jiang, Fengqing and Xu, Zhangchen and Li, Yuetai and Niu, Luyao and Xiang, Zhen and Li, Bo and Lin, Bill Yuchen and Poovendran, Radha},
  journal={arXiv preprint arXiv:2502.12025},
  year={2025}
}
```




