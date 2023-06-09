import os
import openai
import time
import concurrent.futures

from abc import ABC, abstractmethod



class OpenAILanguageModel():
    def __init__(self, api_key, strategy="cot", evaluation_strategy="value", api_model="", enable_ReAct_prompting=True):
        if api_key == "" or api_key == None:
            api_key = os.environ.get("OPENAI_API_KEY", "")
        if api_key != "":
            openai.api_key = api_key
        else:
            raise Exception("Please provide OpenAI API key")
        
        if api_model == "" or api_model == None:
            api_model = os.environ.get("OPENAI_API_MODEL", "")
        if api_model != "":
            self.api_model = api_model
        else:
            self.api_model = "text-davinci-003"

        print(f'Using api_model {self.api_model}')

        self.use_chat_api = 'gpt' in self.api_model

        # reference : https://www.promptingguide.ai/techniques/react

        self.strategy = strategy
        self.evaluation_strategy = evaluation_strategy

    def openai_api_call_handler(self, prompt, max_tokens, temperature):
        while True:
            try:
                messages = [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                response = openai.ChatCompletion.create(
                    model=self.api_model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                text = response['choices'][0]['message']['content'].strip()
                with open("openai.logs", 'a') as log_file:
                    log_file.write("\n" + "-----------" + '\n' +"Prompt : "+ prompt+"\n"+"Response:\n"+text)

                return text

            except openai.error.RateLimitError as e:
                sleep_duratoin = os.environ.get("OPENAI_RATE_TIMEOUT", 30)
                print(f'{str(e)}, sleep for {sleep_duratoin}s, set it by env OPENAI_RATE_TIMEOUT')
                time.sleep(sleep_duratoin)


    
    def generate_text(self, prompt,):
        text = self.openai_api_call_handler(prompt, 1000, 0.5)
        return text

    # def solution(self, states, initial_prompt):

class OptimizedOpenAILanguageModel(OpenAILanguageModel):
    def __init__(self, api_key, strategy="cot", evaluation_strategy="value", cache_enabled=True, api_base="", api_model="", enable_ReAct_prompting=False):
        super().__init__(api_key, strategy, evaluation_strategy, api_base, api_model, enable_ReAct_prompting)
        self.cache_enabled = cache_enabled
        self.thought_cache = {}
        self.state_evaluation_cache = {}

    def parallel_generate_thoughts(self, states, k):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            thoughts = list(executor.map(lambda state: self.generate_thoughts(state, k), states))
            print(f"Parallel generated thoughts: {thoughts}")
        return thoughts

    def parallel_evaluate_states(self, states, inital_prompt):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            state_values = list(executor.map(self.evaluate_states, states, inital_prompt))
            print(f"Parallel evaluated state values: {state_values}")
        return state_values
    