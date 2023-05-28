from abc import ABC, abstractmethod

class AbstractLanguageModel(ABC):
    @abstractmethod
    def generate_thoughts(self, state, k):
        pass

    @abstractmethod 
    def evaluate_states(self, states):
        pass



class MetaAgent(AbstractLanguageModel):
    def __init__(self, model: AbstractLanguageModel):
        self.model = model

    def generate_thoughts(self, state, k):
        thoughts = self.model.generate_thoughts(state, k)
        return thoughts
    
    def evaluate_states(self, states):
        evaluated_states = self.model.evaluate_states(states)
        new_prompt = self.updated_prompt(evaluated_states)
        return new_prompt
    
    def update(self, evaluated_states):
        #critique the prompt based on the thought + it's evaluated state => create more explict prompt 
        return new_prompts