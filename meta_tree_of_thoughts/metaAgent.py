from abc import ABC, abstractmethod
import openai
import langchain
from dotenv import load_dotenv
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory

load_dotenv()


#tree of thoughts


class MetaAgent():
    def __init__(self):
        self.initalize_meta_agent()

    def get_new_instructions(meta_output):
        new_instructions = meta_output.split("Instructions:")[-1]
        return new_instructions
    
    def update_prompt(self, chat_history, user_goal):
        chain = LLMChain(
            llm=self.LLM ,
            prompt=self.meta_prompt,
            verbose=True
        )
        meta_output = chain.run(chat_history=chat_history, old_instructions=self.thinking_prompt, user_goal=user_goal)
        print(meta_output)
        #gte the new instructions from the meta output
        new_instructions = self.get_new_instructions(meta_output)
        variables_required = ["{state_text}", "{initial_prompt}"]
        has_required_variables = all(var in variables_required for var in variables_required)
        if not has_required_variables:
            print("Instructions failed to mutate")
        else:
            self.thinking_prompt = new_instructions
    

    def initalize_meta_agent(self):
        self.thinking_prompt = "Considering the thoughts you've had until now:\n \
        \n'{old_thoughts}'\n\nDevise the next coherent thought that will aid in advancing the reasoning process and to achieve the following goal '{user_goal}'. \
        Assess various scenarios, think unconventionally, anticipate potential challenges, and resolve any outstanding queries. \
        Tap into your mind's full potential and make certain no open questions remain."

        meta_template="""
You need to change the following thinking instructions; {old_instructions} and make it more explicit and descriptive based on increasing its likelihood of achieving the user goal:
{user_goal}

Thinking instructions will be used by an AI assistant to help it think through next step for achieving the user goal, but the above instructions are not good enough.
You have to make them tailored towards the user goal: {user_goal}.

An AI model has just had the below interactions with a user, using the above thinking instructions to achieve the user's goal. Model did not generate thoughts reliable enough to achieve the user goal {user_goal}
Your job is to critique the model's performance using the old thinking instructions and then revise the instructions so that the AI 
model would quickly and correctly respond in the future to achieve the user goal.

###
Old thinking instructions to modify:

{old_instructions}

The strings '{{old_thoughts}}' and the string '{{user_goal}}'  have to appear in the new instructions as they will respectively be used by the A model to store it's old thoughts, and the user's goal when it runs that instruction

### 
AI model's interaction history with the user
    
{chat_history}
 
###
Please reflect on these interactions.

You should critique the models performance in this interaction in respect to achieving the user's goals. What could the AI model have done better?
Indicate this with "Critique: ....

You should then revise the Instructions so that Assistant would quickly and correctly respond in the future.
The AI model's goal is to return the most reliable evaluated thought in the shortest amount of time to work to achieve the user's goal as 
few interactions as possible. The AI Assistant will only see the new Instructions, not the interaction
history, so anything important must be summarized in the Instructions. Don't forget any important details in 
the current Instructions! Indicate the new instructions by "Instructions: ..."

Very important: The string '{{old_thoughts'}} and the string '{{user_goal}}' have to appear in the new instructions as they will respectively be used by the A model to store it's old thoughts, and the user's goal when it runs that instruction
"""

        self.meta_prompt = PromptTemplate(
            input_variables=[ 'old_instructions', 'user_goal', 'chat_history'],
            template=meta_template
        )

        self.LLM = ChatOpenAI(temperature=0)
        #get the chast history from the evauated states 

myModel = MetaAgent()

userGoal = "Find the best next move for white in the chess position e4, e5, bc4, bc5, Qf3, h7"

chatHistory = "Considering the thoughts you've had until now:\n \
\n'Find the best next move for white in the chess position e4, e5, bc4, bc5, Qf3, h7\n The first thought is to identify the weak squares in this position'\n\nDevise the next coherent thought that will aid in advancing the reasoning process and achieving a solution to 'Find the best next move for white in the chess position e4, e5, bc4, bc5, Qf3, h7'. \
Assess various scenarios, think unconventionally, anticipate potential challenges, and resolve any outstanding queries. \
 Tap into your mind's full potential and make certain no open questions remain.\
 Thought generated:\
 What does black want to do next and what can white respond in that situation?"
print(chatHistory)
myModel.update_prompt(chatHistory, userGoal)


    # def run(self, task, max_iters=3, max_meta_iters=5, threshold=0.9):
    #     failed_phrase = 'task failed'
    #     success_phrase = 'task succeeded'
    #     key_phrases = [success_phrase, failed_phrase]

    #     instructions = 'None'
    #     for i in range(max_meta_iters):
    #         print(f'[Episode {i+1}/{max_meta_iters}]')
    #         chain = self.initialize_chain(instructions, memory=None)
    #         output = chain.predict(human_input=task)
    #         for j in range(max_iters):
    #             print(f'(Step {j+1}/{max_iters})')
    #             print(f'Assistant: {output}')
    #             print(f'Human: ')
    #             human_input = input()
    #             if any(phrase in human_input.lower() for phrase in key_phrases):
    #                 break
    #             output = chain.predict(human_input=human_input)
    #         if success_phrase in human_input.lower():
    #             print(f'You succeeded! Thanks for playing!')
    #             return
    #         meta_chain = self.initialize_meta_chain()
    #         meta_output = meta_chain.predict(chat_history=self.get_chat_history(chain.memory))
    #         print(f'Feedback: {meta_output}')
    #         instructions = self.get_new_instructions(meta_output)
    #         print(f'New Instructions: {instructions}')
    #         print('\n' + '#' * 80 + '\n')

    #         # Check if the evaluated states have reached the threshold or a solution has been found
    #         evaluated_states = self.evaluate_states(chain.memory)
    #         if evaluated_states >= threshold:
    #             print(f'Optimal solution found! Thanks for playing!')
    #             return

    #     print(f'You failed! Thanks for playing!')


    #we can pass in evaluated states if higher than 0.9 stop the loop of improving the instructions if not keep going 



#feed user goal initial instructions and their states and solution => create new and refined instructions, needs to imrpove itself forever, i