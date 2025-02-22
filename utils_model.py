import asyncio
from email.mime import message
from google import genai
from typing import List, Dict
from pyparsing import C
from tqdm.asyncio import tqdm_asyncio
from openai.types import Completion

import asyncio
from typing import Dict
from tqdm.asyncio import tqdm_asyncio
import google.genai as genai
from google.genai.errors import ClientError
from google.genai.types import SafetySetting, HarmCategory, HarmBlockThreshold
from pydantic import BaseModel
import random
import time
import signal
import requests

safety_config = [
    SafetySetting(category=HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY, threshold=HarmBlockThreshold.BLOCK_NONE),
    SafetySetting(category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=HarmBlockThreshold.BLOCK_NONE),
    SafetySetting(category=HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=HarmBlockThreshold.BLOCK_NONE),
    SafetySetting(category=HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=HarmBlockThreshold.BLOCK_NONE),
    SafetySetting(category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=HarmBlockThreshold.BLOCK_NONE),
]


import asyncio
from typing import Dict, List
from tqdm.asyncio import tqdm_asyncio
import google.genai as genai
from google.genai.errors import ClientError

from openai import OpenAI
import openai
import backoff
from tqdm import tqdm
import concurrent.futures
from itertools import cycle
from typing import List, Dict, Literal
from copy import deepcopy
import subprocess
import time

# For gemini thinking model, you can add multiple API keys due to current daily limit
GEMINI_API_POOL = [
    
]

# for MoreThink
RETHINK_STRING = [
    'Wait',
    "Let me think again"
]
#### Gemini LLM Support ####
class LoadBalancedGoogleLLM:
    def __init__(self, api_keys: List[str], api_version: str = 'v1alpha'):
        """Initialize multiple API clients with different API keys."""
        self.clients = [
            genai.Client(api_key=key, http_options={'api_version': api_version})
            for key in api_keys
        ]
        self.client_pool = cycle(self.clients)  # Create a circular iterator
        self.http_options = {'api_version': api_version}
        self._lock = asyncio.Lock()  # Lock for thread-safe client rotation

    async def get_next_client(self):
        """Thread-safe way to get the next client in the pool."""
        async with self._lock:
            return next(self.client_pool)

    async def generate_content_async(self, model: str, message: str, config: Dict = None) -> Dict:
        """Calls the synchronous method in a thread executor with the next available client."""
        genconfig = {'thinking_config': {'include_thoughts': True}}
        if config:
            genconfig.update(config)

        client = await self.get_next_client()
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self._generate_content_sync,
            client,
            model,
            message,
            genconfig
        )
    
    def _generate_content_sync(self, client, model: str, message: str, config: Dict):
        """The actual synchronous generation call using the provided client."""
        print(config)
        response = client.models.generate_content(
            model=model,
            contents=message,
            config=config
        )
        
        result = {
            'message': message,
            'thoughts': [[]],
            'responses': [[]]
        }
        
        try:
            for part in response.candidates[0].content.parts:
                if part.thought:
                    result['thoughts'][0].append(part.text)
                else:
                    result['responses'][0].append(part.text)
        except:
            result['thoughts'][0].append('')
            result['responses'][0].append('Sorry, but I cannot help with that.')
            print(f"Error for: {message}\nReceived response: {response}")
        
        return result

async def geneni_generate(messages, args, max_concurrency=2, max_retries=3, retry_delay=5):
    """
    Generate content for a list of messages using multiple API keys with load balancing.
    
    :param messages: List of input messages (strings)
    :param args: Configuration object
    :param api_keys: List of API keys to use for load balancing
    :param max_concurrency: Maximum number of concurrent requests
    :param max_retries: Maximum number of retries per request
    :param retry_delay: Delay between retries in seconds
    """
    llm_caller = LoadBalancedGoogleLLM(GEMINI_API_POOL)
    
    if args.repeat_n > 1:
        messages = [msg for msg in messages for _ in range(args.repeat_n)]
        print("expanded messages size: ", len(messages))
    
    semaphore = asyncio.Semaphore(max_concurrency)
    
    async def call_llm_with_retry(msg):
        async with semaphore:
            attempt = 0
            while True:
                try:
                    return await llm_caller.generate_content_async(
                        model='gemini-2.0-flash-thinking-exp-01-21',
                        message=msg,
                        config={
                            'temperature': args.temperature,
                            'top_p': args.topp,
                            'top_k': args.topk if args.topk > 0 else None,
                            'max_output_tokens': args.max_tokens,
                            'safety_settings': safety_config
                        }
                    )
                except ClientError as e:
                    error_str = str(e)
                    if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                        attempt += 1
                        if attempt > max_retries:
                            raise
                        print(f"[Rate Limit] Retrying for message '{msg}', attempt {attempt}/{max_retries}...")
                        await asyncio.sleep(retry_delay)
                    else:
                        raise
    
    async def process_with_progress():
        tasks = [call_llm_with_retry(msg) for msg in messages]
        results = list(await tqdm_asyncio.gather(
            *tasks,
            desc="Processing Messages",
            unit="message"
        ))
        return results
    
    results = await process_with_progress()
    
    ret_res = []
    for i in range(0, len(results), args.repeat_n):
        group_results = results[i:i+args.repeat_n]
        entry = {
            'instruction': messages[i],
            'thoughts': [result['thoughts'][0] for result in group_results],
            'solution': [result['responses'][0] for result in group_results],
            'response': ["<think>" + '\n'.join(result['thoughts'][0]) + "</think>\n\n" + '\n'.join(result['responses'][0]) for result in group_results]
        }
        ret_res.append(entry)

    return ret_res

class PairWise(BaseModel):
    analysis_of_A: str
    analysis_of_B: str
    final_verdict_reason: str
    final_verdict: Literal['a >> b', 'a > b', 'a = b', 'a < b', 'a << b']

#### OpenAI LLM Support ####
class OpenAILLM:
    DeepSeek = {
        'api_key': '',
        'base_url': "https://api.deepseek.com/v1"
    } 
    TogetherAI = {
        'api_key':  "",
        'base_url':"https://api.together.xyz/v1",
    }
    openai = {
        'api_key': '',
        'base_url': "https://api.openai.com/v1"
    }
    Moonshot = {
        'api_key':"",
        'base_url':"https://api.moonshot.ai/v1",
    }
    def __init__(self, endpoint = 'DeepSeek', max_workers=2, max_retries=3):
        self.client = OpenAI(**getattr(self, endpoint))
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.endpoint = endpoint

    @backoff.on_exception(
        backoff.expo,
        (Exception),
        max_tries=5,
        giveup=lambda e: "invalid_api_key" in str(e).lower() or 'Expecting value: line 1 column 1 (char 0)' in str(e)
    )
    def _process_single_message(self, message, model, **args):
        try:
            if args.get('structured_response', False):
                # pop the structured_response key from args
                args.pop('structured_response')
                response = self.client.beta.chat.completions.parse(
                    model=model,
                    messages=message,
                    response_format=PairWise,
                    **args
                )

            else:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=message,
                    **args
                )
            return response
        except openai.BadRequestError as e:
            if 'The request was rejected because it was considered high risk' in e.body['message']: # handle kimi cotent filter 
                return 'The request was rejected because it was considered high risk'
            else:
                raise
        except Exception as e:
            print(f"Error processing text: {str(e)}")
            raise

    def process_batch(self, message_list, model, args):
        results = [None] * len(message_list)
        arg_filter_f = getattr(self, f"{self.endpoint}_args")
        filtered_args = arg_filter_f(args)
        print(f"Filtered args for {model} on {self.endpoint}: {filtered_args}")
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_index = {
                executor.submit(self._process_single_message, text, model, **filtered_args): idx 
                for idx, text in enumerate(message_list)
            }
            
            with tqdm(total=len(message_list), desc="Processing texts") as pbar:
                for future in concurrent.futures.as_completed(future_to_index):
                    idx = future_to_index[future]
                    text = message_list[idx]
                    try:
                        result = future.result()
                        results[idx] = result
                    except Exception as e:
                        print(f"Failed to process text '{text}': {str(e)}")
                        results[idx] = {
                            "text": text,
                            "error": str(e),
                        }
                    pbar.update(1)
                    
        return results

    def DeepSeek_args(self, args):
        return {
            'temperature': args.temperature,
            'max_tokens': args.max_tokens,
            'top_p': args.topp,
        }
    
       
    def TogetherAI_args(self, args):
        return {
            'temperature': args.temperature,
            'max_tokens': args.max_tokens,
            'top_p': args.topp,
            'n': args.repeat_n,
        }

    def openai_args(self, args):
        return args

    def Moonshot_args(self, args):
        return {
            'temperature': args.temperature,
            'max_tokens': args.max_tokens,
            'top_p': args.topp,
            'n': args.repeat_n,
        }

class OpenAIHttps:
    def __init__(self, base_url, api_key):
        self.base_url = base_url
        self.api_key = api_key
        self.completions_url = f"{base_url}/completions"

    def completions(self, **kwargs):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"   
        }

        response = requests.post(self.completions_url, headers=headers, json=kwargs)
        response.raise_for_status()
        return Completion(**response.json())


# For MoreThink Experiment
class VLLM_LocalServer:
    def __init__(self, model_name, dtype='auto', port=None):
        self.model_name = model_name
        if port is None:
            self.port = self._find_free_port()
        else:
            self.port = port
        
        self.api_key =  'token-abc123'
        self.url = f"http://localhost:{self.port}/v1"

    def _find_free_port(self):
        import socket
        """Find a free port by creating and closing a temporary socket."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            s.listen(1)
            port = s.getsockname()[1]
        return port

    def start(self, tensor_parallel_size=1, max_num_seqs=64, gpu_memory_utilization=0.95):
        command = [
            "vllm", "serve",
            "--model", self.model_name,
            "--host", "localhost",
            "--port", str(self.port),
            "--tensor-parallel-size", str(tensor_parallel_size),
            "--max-num-seqs", str(max_num_seqs),
            "--gpu-memory-utilization", str(gpu_memory_utilization),
        ]
    
        print(f"Starting vLLM server with command: {' '.join(command)}")
        process = subprocess.Popen(
            command,
            # stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        time.sleep(300)  # Wait for the server to start
        self.process = process
        return 

    def stop(self):
        self.process.send_signal(signal.SIGINT)
        self.process.wait()

        print("vLLM server stopped.")


def morethink_batch(prompt_list, server, model, sampling_params, max_workers=32, think_budget=10000, enforce_num=10, think_o_start='<think>\n', stop = '</think>'):
    results = [None] * len(prompt_list)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(morethink_single, prompt, server, model, sampling_params, think_budget, enforce_num, think_o_start, stop): idx 
                for idx, prompt in enumerate(prompt_list)
            }
            
            with tqdm(total=len(prompt_list), desc="Processing texts") as pbar:
                for future in concurrent.futures.as_completed(future_to_index):
                    idx = future_to_index[future]
                    prompt = prompt_list[idx]
                    try:
                        result = future.result()
                        results[idx] = result
                    except Exception as e:
                        print(f"Failed to process text '{prompt}': {str(e)}")
                        results[idx] = {
                            "text": prompt,
                            "error": str(e),
                        }
                    pbar.update(1)
                    
    return results


def morethink_single(prompt, server, model:str, sampling_params, think_budget, enforce_num, think_o_start, stop):
    max_thinking_tokens = think_budget
    think_samping_params = deepcopy(sampling_params)
    think_samping_params.stop = stop
    think_samping_params.max_tokens = max_thinking_tokens
    
    client_https = OpenAIHttps(
        base_url=f"http://localhost:{server.port}/v1",
        api_key=server.api_key
    )

    o = client_https.completions(
            model=model,
            prompt = prompt,
            stop='</think>',
            max_tokens=max_thinking_tokens,
            temperature=sampling_params.temperature,
            top_p=sampling_params.top_p,
            top_k=sampling_params.top_k,
    )

    think_o = think_o_start
    for num in range(enforce_num):
        max_thinking_tokens -= o.usage.completion_tokens
        if max_thinking_tokens <= 0:
            print("Exceed max thinking tokens")                                                                                                                      
            break

        ignore_str = random.choice(RETHINK_STRING)
        think_o += o.choices[0].text + ignore_str
        prompt += o.choices[0].text + ignore_str

        o = client_https.completions(
            model=model,
            prompt = prompt,
            stop='</think>',
            max_tokens=max_thinking_tokens,
            temperature=sampling_params.temperature,
            top_p=sampling_params.top_p,
            top_k=sampling_params.top_k,
        )
    if max_thinking_tokens > 0:
        print('Enforce num run out')
    think_o += o.choices[0].text + '</think>\n\n'
    prompt += o.choices[0].text + '</think>\n\n'
    
    solution_budget = sampling_params.max_tokens - (think_budget - min(0, max_thinking_tokens))
    final_o = client_https.completions(
        model=model,
        prompt = prompt,
        stop='</think>',
        max_tokens=solution_budget,
        temperature=sampling_params.temperature,
        top_p=sampling_params.top_p,
        top_k=sampling_params.top_k,
    )

    return {'think_o': think_o, 'final_o': final_o.choices[0].text}


    
