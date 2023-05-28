
from abc import ABC, abstractmethod

class AbstractLanguageModel(ABC):
    @abstractmethod
    def generate_text(self, prompt):
        pass

class ThinkingAgent:

    def __init__(self, model: AbstractLanguageModel, strategy="cot", evaluation_strategy="value"):
        self.strategy = strategy
        self.evaluation_strategy = evaluation_strategy
        self.model = model
    
    def generate_thoughts(self, state, k, inital_prompt):
        if (type(state) == str):
            state_text = state
        else:
            state_text = '\n'.join(state)
        
        prompt = f"Considering the thoughts you've had until now:\n\n{state_text}\n\nDevise the next coherent thought that will aid in advancing the reasoning process and achieving a solution to {inital_prompt}. Assess various scenarios, think unconventionally, anticipate potential challenges, and resolve any outstanding queries. Tap into your mind's full potential and make certain no open questions remain."

        thoughts = [self.model.generate_text(prompt) for i in range(0, k)]

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