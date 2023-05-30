import os
# from tree_of_thoughts.openaiModels import OpenAILanguageModel
# from tree_of_thoughts.treeofthoughts import TreeofThoughts
from meta_tree_of_thoughts.treeofthoughts import TreeofThoughts
from meta_tree_of_thoughts.thinkingAgent import ThinkingAgent
from meta_tree_of_thoughts.openaiModel import OpenAILanguageModel

model = OpenAILanguageModel(api_key='', api_model='gpt-3.5-turbo')

search_algo = "BFS"
tree_of_thoughts = TreeofThoughts(model, search_algo)

#choose search algorithm('BFS' or 'DFS')
search_algorithm = "BFS"

# value or vote
evaluation_strategy = "value"

tree_of_thoughts= TreeofThoughts(model, search_algorithm)

input_problem = "use 4 numbers and basic arithmetic operations (+-*/) to obtain 24"

#call the solve emthod with the input problem and other params
solution = tree_of_thoughts.solve(input_problem)

#use the solution in your production environment
print(f"solution: {solution}")