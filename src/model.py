import time
import requests
import json
from tqdm import tqdm
import threading

class APIModel:

    def __init__(self, model, api_key, api_url) -> None:
        self.__api_key = api_key
        self.__api_url = api_url
        self.model = model
        

    def __req(self, text, temperature, stream=False, max_try=5):
        url = f"{self.__api_url}"
        
        # 构造请求载荷，加入 stream 参数
        pay_load_dict = {
            "model": f"{self.model}",
            "messages": [{
                "role": "user",
                "content": f"{text}"
            }],
            "temperature": temperature,
            "stream": stream  # 由参数控制是否开启流式
        }
        
        headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {self.__api_key}',
            'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
            'Content-Type': 'application/json'
        }

        def execute_request():
            # 注意：如果是 stream 模式，通常不需要设置超长的 timeout，这里维持你的逻辑
            response = requests.request(
                "POST", url, headers=headers, 
                data=json.dumps(pay_load_dict), 
                timeout=300 if not stream else 60,
                stream=stream # 告诉 requests 保持连接
            )
            
            if response.status_code != 200:
                raise Exception(f"Status Code: {response.status_code}, Text: {response.text[:200]}")

            if not stream:
                # --- 非流式处理 ---
                content =  json.loads(response.text)['choices'][0]['message']['content']
                if not content:
                    # Fallback to reasoning_content if content is None or empty
                    content = json.loads(response.text)['choices'][0]['message'].get('reasoning_content')
                if not content:
                    raise ValueError(f"response content is None or empty")
                return content
            else:
                # --- 流式处理 (Generator) ---
                def generate():
                    for line in response.iter_lines():
                        if line:
                            line_text = line.decode('utf-8').strip()
                            if line_text.startswith("data: "):
                                data_str = line_text[6:]
                                if data_str == "[DONE]":
                                    break
                                try:
                                    data_json = json.loads(data_str)
                                    # 注意：流式的路径通常是 delta 而不是 message
                                    content = data_json['choices'][0].get('delta', {}).get('content', '')
                                    if content:
                                        yield content
                                except Exception as e:
                                    print(f"Stream parsing error: {e}")
                return generate()

        # 重试逻辑封装
        for i in range(max_try + 1): # 0是初始请求，后面是 retry
            try:
                return execute_request()
            except Exception as e:
                if i < max_try:
                    print(f"[APIModel.__req] Attempt {i+1} failed: {e}, retrying...")
                    time.sleep(0.5)
                else:
                    print(f"Max retries reached. Failed.")
                    return None
    
    def chat(self, text, temperature=1):
        response = self.__req(text, temperature=temperature, max_try=5)
        return response

    def __chat(self, text, temperature, res_l, idx):
        
        response = self.__req(text, temperature=temperature)
        res_l[idx] = response
        return response
        
    def batch_chat(self, text_batch, temperature=0):
        max_threads=15 # limit max concurrent threads using model API
        res_l = ['No response'] * len(text_batch)
        thread_l = []
        for i, text in zip(range(len(text_batch)), text_batch):
            thread = threading.Thread(target=self.__chat, args=(text, temperature, res_l, i))
            thread_l.append(thread)
            thread.start()
            while len(thread_l) >= max_threads: 
                for t in thread_l:
                    if not t .is_alive():
                        thread_l.remove(t)
                time.sleep(0.3) # Short delay to avoid busy-waiting

        for thread in tqdm(thread_l):
            thread.join()
        return res_l
