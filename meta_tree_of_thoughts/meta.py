from abc import ABC, abstractmethod
import openai
import langchain
from tree_of_thoughts import TreeofThoughts
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
        history, so anything important must be summarized in the Instructios. Don't forget any important details in 
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
        return new_instructions\
    






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



    def generate_thoughts(self, state, k, inital_prompt, updatedPrompt):
        if (type(state) == str):
            state_text = state
        else:
            state_text = '\n'.join(state)
        print("We receive a state of type", type(state), "For state: ", state, "\n\n")
        
        # prompt = f"Given the current state of reasoning: \n\n\n'{state_text}'\n\n\nGenerate the next best coherent thought to achieve the reasoning process and get the solution: "
        # prompt = f"Based on the current state of reasoning: \n\n\n'{state_text} Provide the next coherent thought that will help progress the reasoning process and reach an soluton "
        # prompt = f"These are the thoughts you've had: \n\n\n{state_text}, provide the next coherent thought that will help advance the reasoning process and reach an solution for this problem {inital_prompt}. Think sharply, think out of the box, predict failure. Do not leave any open questions. Unleash your mind."

        #we can just make this like this: prompt = {meta.agent.response(state_text, initial_prompt)}
        prompt = f"Considering the thoughts you've had until now:\n\n{state_text}\n\nDevise the next coherent thought that will aid in advancing the reasoning process and achieving a solution to {inital_prompt}. Assess various scenarios, think unconventionally, anticipate potential challenges, and resolve any outstanding queries. Tap into your mind's full potential and make certain no open questions remain."


        prompt += self.ReAct_prompt
        # print(prompt)
        thoughts = self.generate_text(prompt, k)
        # print(thoughts)
        print(f"Generated thoughts: {thoughts}")
        return thoughts

        
    def generate_solution(self, initial_prompt, state):
        if (type(state) == str):
            state_text = state
        else:
            state_text = '\n'.join(state)
        
        prompt = f"Considering the reasoning provided:\n\n'{state_text}'\n\nDevise the best possible solution for the task: {initial_prompt}"
        answer = self.generate_text(prompt, 1)
        # print(thoughts)
        print(f"General solution : {answer}")
        return answer

    def evaluate_states(self, states, inital_prompt):


        if self.evaluation_strategy == 'value':
            state_values = {}
            for state in states:
                if (type(state) == str):
                    state_text = state
                else:
                    state_text = '\n'.join(state)
                print("We receive a state of type", type(state), "For state: ", state, "\n\n")
                prompt = f"Given the current state of reasoning: '{state_text}', evaluate its value as a float between 0 and 1, become very pessimistic think of potential adverse risks on the probability of this state of reasoning achieveing {inital_prompt} and DO NOT RESPOND WITH ANYTHING ELSE: OTHER THAN AN FLOAT"
                
                response = self.openai_api_call_handler(prompt, 10, 1)
                try:
                    value_text = self.openai_choice2text_handler(response.choices[0])
                    # print(f'state: {value_text}')
                    value = float(value_text)
                    print(f"value: {value}")
                except ValueError:
                    value = 0  # Assign a default value if the conversion fails
                state_values[state] = value
            return state_values

        elif self.evaluation_strategy == 'vote':
            states_text = '\n'.join([' '.join(state) for state in states])

            prompt = f"Given the following states of reasoning, vote for the best state utilizing an scalar value 1-10:\n{states_text}\n\nVote, on the probability of this state of reasoning achieveing {inital_prompt} and become very pessimistic very NOTHING ELSE"

            response = self.openai_api_call_handler(prompt, 50, 1)

            print(f'state response: {response}')

            best_state_text = self.openai_choice2text_handler(response.choices[0])

            print(f"Best state text: {best_state_text}")

            best_state = tuple(best_state_text.split())

            print(f'best_state: {best_state}')

            return {state: 1 if state == best_state else 0 for state in states}

        else:
            raise ValueError("Invalid evaluation strategy. Choose 'value' or 'vote'.")
    # def solution(self, states, initial_prompt):
