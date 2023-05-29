
from abc import ABC, abstractmethod
import random
from meta_tree_of_thoughts.metaAgent import MetaAgent
class AbstractLanguageModel(ABC):
    @abstractmethod
    def generate_text(self, prompt):
        pass

class ThinkingAgent:

    def __init__(self, model: AbstractLanguageModel, strategy="cot", evaluation_strategy="value"):
        self.strategy = strategy
        self.evaluation_strategy = evaluation_strategy
        self.model = model
        self.MetaAgent = MetaAgent()

    def generate_thoughts(self, state, k, initial_prompt):
        if (type(state) == str):
            state_text = state
        else:
            state_text = '\n'.join(state)
        
        #this needs to be in meta agent prompt = f"Considering the thoughts you've had until now:\n\n{state_text}\n\nDevise the next coherent thought that will aid in advancing the reasoning process and achieving a solution to {inital_prompt}. Assess various scenarios, think unconventionally, anticipate potential challenges, and resolve any outstanding queries. Tap into your mind's full potential and make certain no open questions remain."
        prompt = self.MetaAgent.thinking_prompt
        prompt.replace("{state_text}", state_text)
        prompt.replace("{initial_prompt}", initial_prompt)
        thoughts = [self.model.generate_text(prompt) for i in range(0, k)]
        #randomly choosing one thought to give as the example of conversation
        chosen_thought = random.choice(thoughts)
        chat_history = f"AI:\n\n {prompt} \n Generated thought from prompt:\n {chosen_thought} "
        self.MetaAgent.update_prompt(chat_history, initial_prompt)
        return thoughts

        
    def generate_solution(self, initial_prompt, chain_of_thoughts):
        if (type(chain_of_thoughts) == str):
            state_text = chain_of_thoughts
        else:
            state_text = '\n'.join(chain_of_thoughts)
        
        prompt = f"Considering the reasoning provided:\n\n'{state_text}'\n\nDevise the best possible solution for the task: {initial_prompt}"
        answer = self.model.generate_text(prompt)

        return answer

    def evaluate_states(self, states, inital_prompt):

        state_values = {}
        for state in states:
            print("We receive a state of type", type(state), "For state: ", state, "\n\n")
            
            old_thoughts = '\n'.join(state[:-1])

            latest_generated_thought = state[-1]
            
            prompt = f"""To achieve the following goal: '{inital_prompt}' value the context of the past thoughts and more importantly the latest generated thought you had AS A FLOAT BETWEEN 0 AND 1\n
            Past thoughts:\n\n
            {old_thoughts}\n       
            Evaluate the latest thought as a value between 0 and 1 based on how likely it is it achieve the inital goal: '{inital_prompt}'\n
            Latest thought:\n
            {latest_generated_thought}\n
            Evaluation AS A FLOAT BETWEEN 0 and 1:\n DO NOT RETURN ANYTHING ELSE"""
            
            # prompt = f"Given the current thought of reasoning: '{state_text}', evaluate its value as a float between 0 and 1, become very pessimistic think of potential adverse risks on the probability of this state of reasoning achieveing {inital_prompt} and DO NOT RESPOND WITH ANYTHING ELSE: OTHER THAN AN FLOAT"

            response = self.model.generate_text(prompt)
            try:
                value_text = response
                # print(f'state: {value_text}')
                value = float(value_text)
                print(f"value: {value}")
            except ValueError:
                value = 0  # Assign a default value if the conversion fails
            state_values[state] = value
        


        return state_values
    

# #for meta agent
# chat_history = f"{chain_of_thoughts: value}"
# #architectural
# #1 ======> chain of thoughts + values for every thought
# #2 =====> 1 thought + 1 state => new prompt
# #3 =====> every thought is evaluated + the chain of thougths + values are then evaluated

# self.thinking_prompt = self.metaAgent.updateInstructions(chat_history)