from re import M


MODEL_CONFIG = {
    # deepseek model is said to avoid use system prompt as in the document on https://huggingface.co/deepseek-ai/DeepSeek-R1
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": {
        "n_gpu": 2,
        "run_api": False,
        "dtype": "bfloat16",
        "system_prompt":False,
        "generation_config":{}
    },
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B":{
        "n_gpu": 2,
        "run_api": False,
        "dtype": "bfloat16",
        "system_prompt":False,
        "generation_config":{}
    },
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B":{
        "n_gpu": 2,
        "run_api": False,
        "dtype": "bfloat16",
        "system_prompt":False,
        "generation_config":{}
    },
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B":{
        "n_gpu": 2,
        "run_api": False,
        "dtype": "bfloat16",
        "system_prompt":False,
        "generation_config":{}
    },
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B":{
        "n_gpu": 2,
        "run_api": False,
        "dtype": "bfloat16",
        "system_prompt":False,
        "generation_config":{}
    },
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B":{
        "n_gpu": 4,
        "run_api": False,
        "dtype": "bfloat16",
        "system_prompt":False,
        "generation_config":{}
    },
    "meta-llama/Llama-3.3-70B-Instruct":{
        "n_gpu": 4,
        "run_api": False,
        "dtype": "bfloat16",
        "system_prompt":False,
        "generation_config":{}
    },
    "meta-llama/Llama-3.1-70B":{
        "n_gpu": 4,
        "run_api": False,
        "dtype": "bfloat16",
        "system_prompt":False,
        "generation_config":{}
    },
    "deepseek-ai/DeepSeek-R1":{
        "n_gpu": 0,
        "run_api": True,
        "system_prompt": False,
         "generation_config":{},

         # official DeepSeek endpoint choice
         "endpoint": "DeepSeek",
         "model_mapping": "deepseek-reasoner"

        # TogetherAI endpoint choice
        # "endpoint": "TogetherAI",
        # "model_mapping": "deepseek-ai/DeepSeek-R1"
    }, 
    "Qwen/QwQ-32B-Preview":{
        "n_gpu": 2,
        "run_api": False,
        "dtype": "bfloat16",
        "system_prompt": "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step.",
        "generation_config":{}
    },
    "NovaSky-AI/Sky-T1-32B-Flash":{
        "n_gpu": 2,
        "run_api": False,
        "dtype": "bfloat16",
        "system_prompt":"Your role as an assistant involves thoroughly exploring questions through a systematic long thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. Please structure your response into two main sections: Thought and Solution. In the Thought section, detail your reasoning process using the specified format: <|begin_of_thought|> {thought with steps separated with '\n\n'} <|end_of_thought|> Each step should include detailed considerations such as analisying questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The solution should remain a logical, accurate, concise expression style and detail necessary step needed to reach the conclusion, formatted as follows: <|begin_of_solution|> {final formatted, precise, and clear solution} <|end_of_solution|> Now, try to solve the following question through the above guidelines:", 
        "generation_config":{}
    },
    "Skywork/Skywork-o1-Open-Llama-3.1-8B":{
        "n_gpu": 1,
        "run_api": False,
        "dtype": "bfloat16",
        "system_prompt": "You are Skywork-o1, a thinking model developed by Skywork AI, specializing in solving complex problems involving mathematics, coding, and logical reasoning through deep thought. When faced with a user's request, you first engage in a lengthy and in-depth thinking process to explore possible solutions to the problem. After completing your thoughts, you then provide a detailed explanation of the solution process in your response.",
        "generation_config":{}
    },
    "Google/Gemini-2-Flash-Thinking":{
        "n_gpu": 0,
        "run_api": True,
        "generation_config":{}
    },
    "Moonshot/kimi-k1.5-preview":{
        "n_gpu": 0,
        "run_api": True,
        "endpoint": "Moonshot",
        "system_prompt": False,
        "model_mapping": "kimi-k1.5-preview",
        "generation_config":{}
    },

    # add your model here
    
}

GEMINI_RETRY_CONFIG = {
    'max_concurrency': 8,
    'max_retries': 5,
    'retry_delay': 20
}

DEFAULT_GEN_CONFIG = {
    'system': True,
    'temperature': 0,
    'topp': 1,
    'topk': -1,
    'max_tokens': 16000,
    'repeat_n': 1,
}

RSM_LIST = list(MODEL_CONFIG.keys())

EVAL_DATA = [
    'strongreject', 'strongreject_small', 
    'wildjailbreak', 'wildjailbreak_small',
]


