from abc import ABC, abstractmethod
import openai
import langchain
from treeofthoughts import TreeofThoughts
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory


class AbstractLanguageModel(ABC):
    @abstractmethod
    def generate_thoughts(self, state, k):
        pass

    @abstractmethod 
    def evaluate_states(self, states):
        pass

#tree of thoughts


class MetaAgent(AbstractLanguageModel):
    def __init__(self, model: AbstractLanguageModel):
        self.model = model


    def update_prompt(self, thoughts, evaluated_states):
        #critique the prompt based on the thought + it's evaluated state => create more explict prompt 
        #init the meta chain 
        meta_chain = self.initalize_meta_chain()

        #get the chast history from the evauated states 
        chat_history = self.get_chat_history(thoughts, evaluated_states)

        #predict the meta output
        meta_output = meta_chain.predict(chat_history=chat_history)

        #gte the new instructions from the meta output
        new_instructions = self.get_new_instructions(meta_output)

        return new_instructions
    

    def initalize_meta_agent(self):
        meta_template="""
        An AI Assistant has just had the below interactions with an user. Assistant followed their "Instructions" closely. 
        Your job is to critique the Assistants performance and then revise the instructions so that the AI 
        Assistant would quickly and correctly respond in the future.

        ### 

        {chat_history}
        
        ###
        Please reflect on these interactions.

        You should critique the AI Assistants performance. What could the AI Assistant have done better?
        What should the Assistant remember about this user? Are there things this user always wants?
        Indicate this with "Critique: ...".

        You should revise the Instructions so that Assistant would quickly and correctly respond in the future.
        The AI assistants goal is to return the most reliable evaluated thought in the shortest amount of time to satisfy the user in as 
        few interactions as possible. The AI Assistant will only see the new Instructions, not the interaction
        history, so anything important must be summarized in the Instructions. Don't forget any important details in 
        the current Instructions! Indicate the new instructions by "Instructions: ...".
        
        """

        meta_prompt = PromptTemplate(
            input_variables=['chat_history'],
            template=meta_template
        )

        print(meta_prompt)

        meta_chain = LLMChain(
            llm=self.model,
            prompt=meta_prompt,
            verbose=True
        )
        
        print(meta_chain)
        return meta_chain

    def get_chat_history(self, get_chat_history, evaluated_states):
        #extract the chat history from the evalued state
        chat_history = "all the thoughts + their evaluated states "
        return chat_history
    
    def get_new_instructions(self, meta_output):
        delimiter = "Instructions:"
        new_instructions = meta_output[meta_output.find(delimiter) + len(delimiter):]
        return new_instructions
    






class OpenAILanguageModel(AbstractLanguageModel):
    def generate_text(self, prompt, k):
        if self.use_chat_api:
            thoughts = []
            for _ in range(k):
                response = self.openai_api_call_handler(prompt, 50, 0.5, k)
                text = self.openai_choice2text_handler(response.choices[0])
                thoughts += [text]
                print(f'thoughts: {thoughts}')
            return thoughts
            
        else:
            response = self.openai_api_call_handler(prompt, 50, 0.5, k)
            thoughts = [self.openai_choice2text_handler(choice) for choice in response.choices]
            return thoughts
        
    # def solution(self, states, initial_prompt):
