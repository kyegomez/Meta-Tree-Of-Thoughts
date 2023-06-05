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

    def get_new_instructions(self, meta_output):
        new_instructions = meta_output.split("Instructions:")[-1]
        return new_instructions
    
    def update_prompt(self, chat_history, user_goal):
        chain = LLMChain(
            llm=self.LLM ,
            prompt=self.meta_prompt,
            verbose=True
        )
        meta_output = chain.run(chat_history=chat_history, old_instructions=self.thinking_prompt, objective=user_goal)
        #gte the new instructions from the meta output
        new_instructions = self.get_new_instructions(meta_output)
        print("New thinking instructions: ", new_instructions)
        variables_required = ["{old_thoughts}"]
        has_required_variables = all(var in variables_required for var in variables_required)
        if not has_required_variables:
            print("Instructions failed to mutate")
        else:
            self.thinking_prompt = new_instructions
    

    def initalize_meta_agent(self):
        self.thinking_prompt = "You're Athena, an AI model devoted to helping Humans by any means necessary. You're purpose is to generate a series of solutions to comply with the user's instructions, you must generate solutions on the basis of determining the most reliable solution in the shortest amount of time. The user has provided you with: {objective} complete this instruction BY ANY MEANS NECESSARY by considering the solutions you've had until now:\n\
\n'{old_thoughts}'\n\n Think about the next best step to achive {objective}.\
If you already have enough thoughts to achieve the goal, start improving some of the steps and verify that you are perfectly meeting the goal '{objective}'.\n Next step:"

        meta_template="""
You need to change the following solutions instructions;\n'{old_instructions}'\n To make the solutions directly solving the user objective '{objective}'

Solutions instructions will be used by an AI assistant to direct it to create the thoughts to progress in achieving the user goal: '{objective}'.
The Solutions instructions have to lead to thoughts that make the AI progress fast in totally achieving the user goal '{objective}'. The Solutions generated have to be sharp and concrete, and lead to concrete visible progress in achieving the user's goal.


An AI model has just had the below interactions with a user, using the above solutions instructions to progress in achieve the user's goal. AI Model's generated thoughts don't lead to good enough progress in achieving: '{objective}'
Your job is to critique the model's performance using the old solution instructions and then revise the instructions so that the AI 
model would quickly and correctly respond in the future to concretely achieve the user goal.

Old thinking instructions to modify:

###
{old_instructions}
###
The strings '{{old_thoughts}}' and the string '{{objective}}'  have to appear in the new instructions as they will respectively be used by the AI model to store it's old thoughts, and the user's goal when it runs that instruction

AI model's interaction history with the user:

###
{chat_history}
###

Please reflect on these interactions.

You should critique the models performance in this interaction in respect to why the solutions it gave aren't directly leading to achieving the user's goals. What could the AI model have done better to be more direct and think better?
Indicate this with "Critique: ....

You should then revise the Instructions so that Assistant would quickly and correctly respond in the future.
The AI model's goal is to return the most reliable solution that leads to fast progressing in achieving the user's goal in as few interactions as possible.
The solutions generated should not turn around and do nothing, so if you notice that the instructions are leading to no progress in solving the user goal, modify the instructions so it leads to concrete progress.
The AI Assistant will only see the new Instructions the next time it thinks through the same problem, not the interaction
history, so anything important to do must be summarized in the Instructions. Don't forget any important details in
the current Instructions! Indicate the new instructions by "Instructions: ..."

VERY IMPORTANT: The string '{{old_thoughts'}} and the string '{{objective}}' have to appear in the new instructions as they will respectively be used by the AI model to store it's old thoughts, and the user's goal when it runs that instruction
"""

        self.meta_prompt = PromptTemplate(
            input_variables=[ 'old_instructions', 'objective', 'chat_history'],
            template=meta_template
        )

        self.LLM = ChatOpenAI(temperature=0)
        #get the chast history from the evauated states 
