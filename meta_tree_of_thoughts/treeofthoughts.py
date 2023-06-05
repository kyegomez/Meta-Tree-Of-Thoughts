import os
import time
import json
import logging 
import argparse
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from meta_tree_of_thoughts.thinkingAgent import ThinkingAgent
import numpy as np

# for each rejected path store the reason for rejection and then pass the reason -> thought generator function
# thought -> evaluated (0.3, 'This is a bad_decision = 2 + 23 + 232323 does not = 24') -> thought generato_functin

# class TreeofThoughts:
#     def __init__(self, model, search_algorithm):
#         self.model = model
#         self.thinkingAgent = ThinkingAgent(self.model)
#         self.search_algorithm = search_algorithm
#         self.tree: Dict[str, Dict[str, float]] = {
#             "nodes": {}
#         }

#     def solve(self, initial_prompt: str, 
#               num_thoughts: Optional[int] = 3, 
#               max_steps: Optional[int] = 3, 
#               max_states: Optional[int] = 5, 
#               value_threshold: Optional[float] = 0.5,
#               confidence_threshold: Optional[float] = 0.8, 
#               max_iterations: Optional[int] = 40, 
#               convergence_threshold: Optional[float] = None, 
#               convergence_count: Optional[int] = None) -> str:
#         self.file_name = f"logs/tree_of_thoughts_output_{self.search_algorithm}.json"
#         try:
#             best_thoughts = ""
#             if self.search_algorithm == 'BFS':
#                 result = self.tot_bfs(initial_prompt, num_thoughts, max_steps, max_states, value_threshold)
#                 if result:
#                     self.save_tree_to_json(self.file_name)
#                     best_thoughts = result
#             elif self.search_algorithm == 'DFS':
#                 result = self.tot_dfs(initial_prompt, num_thoughts, max_steps, value_threshold, 
#                                         confidence_threshold=confidence_threshold, max_iterations=max_iterations, convergence_threshold=convergence_threshold, 
#                                         convergence_count=convergence_count)
#                 if result:
#                     self.save_tree_to_json(self.file_name)
#                     best_thoughts = result
#             if best_thoughts:
#                 solution = self.thinkingAgent.generate_solution(initial_prompt, best_thoughts)
#                 if solution:
#                     return solution
#             else:
#                 raise ValueError("Invalid search algorithm. Choose 'BFS' or 'DFS'.")
#         except KeyboardInterrupt:
#             logger.error("Keyboard interrupt detected.")
#         except ValueError as e:
#             logger.error(f"Error: {e}")
#         finally:
#             logger.info("Saving the current tree and metrics.")
#             self.save_tree_to_json(self.file_name)
    
#     def logNewState(self, state, evaluation):
#         state = " ==> ".join(state)
#         self.tree["nodes"][state] = evaluation
#         self.save_tree_to_json(self.file_name)    
        
#     def tot_bfs(self, initial_prompt, num_thoughts, max_steps, max_states, pruning_threshold):
#         current_states = [[f"My goal is to offer the most optimal to this user request '{initial_prompt}'"]]
#         state_values = {}
#         for step in range(1, max_steps + 1):
#             for state in current_states:
#                 thoughts = self.thinkingAgent.generate_thoughts(state, num_thoughts, initial_prompt)
#                 newStates = []
#                 for thought in thoughts:
#                     flattened_state = (*state, thought)
#                     newStates.append(flattened_state)

#                 evaluated_thoughts = self.thinkingAgent.evaluate_states(newStates, initial_prompt)

#                 selected_states = []
#                 for state, value in evaluated_thoughts.items():
#                     if value >= pruning_threshold:
#                         selected_states.append(state)
#                         state_values[state] = value
#                         self.logNewState(state, value)
#             if(len(selected_states) >1):
#                 current_states = selected_states[:max_states]
                
#         if (len(current_states) == 1):
#             return initial_prompt
#         # print(current_states, state_values)
#         best_state = max(current_states, key=lambda state: state_values[state])
#         print(f'best_state: {best_state}')

#         return best_state

#     def tot_dfs(self, 
#                 initial_prompt: str, 
#                 num_thoughts: any,
#                 max_steps: int,
#                 value_threshold, 
#                 pruning_threshold=0.5, 
#                 confidence_threshold=None, max_iterations=None, convergence_threshold=None, convergence_count=None):
#         output = []
#         iteration_count = 0
#         consecutive_convergence_count = 0
#         prev_best_value = None
#         file_name = f"logs/tree_of_thoughts_output_{self.search_algorithm}.json"

#         def dfs(state, step):
#             nonlocal consecutive_convergence_count, prev_best_value, iteration_count, output
#             if step > max_steps:
#                 thought = self.thinkingAgent.generate_thoughts(state, 1, initial_prompt)
#                 value = self.thinkingAgent.evaluate_states({state}, initial_prompt)[state]
#                 output.append((thought, value))

#                 if confidence_threshold is not None and value >= confidence_threshold:
#                     return True

#                 if prev_best_value is not None and convergence_threshold is not None:
#                     if abs(value - prev_best_value) < convergence_threshold:
#                         consecutive_convergence_count += 1
#                     else:
#                         consecutive_convergence_count = 0

#                 prev_best_value = value
#                 iteration_count += 1

#                 if (max_iterations is not None and iteration_count >= max_iterations) or (convergence_count is not None and consecutive_convergence_count >= convergence_count):
#                     return True

#                 return False

#             for next_state in sorted(self.thinkingAgent.generate_thoughts(state, num_thoughts, initial_prompt)):
#                 state_value = self.thinkingAgent.evaluate_states({next_state}, initial_prompt)[next_state]
#                 logger.info(f"State: {next_state}, Value: {state_value}")

#                 if state_value > value_threshold and (pruning_threshold is None or state_value >= pruning_threshold):
#                     child = (*state, next_state)
#                     if dfs(child, step + 1):
#                         return True

#             self.save_tree_to_json(file_name)
#             return False

#         dfs([[initial_prompt]], 1)
#         best_state = max(output, key=lambda x: x[1])
#         return best_state[0]

#     def save_tree_to_json(self, file_name):
#         os.makedirs(os.path.dirname(file_name), exist_ok=True)

#         with open(file_name, 'w') as json_file:
#             json.dump(self.tree, json_file, indent=4)

#     def print_tree(self, 
#                    node: str, 
#                    depth=0):
#         thought = self.tree["metrics"]["thoughts"].get(node, "")
#         evaluation = self.tree["metrics"]["evaluations"].get(node, "")

#         tree_info = f"{'  ' * depth}Node: {node}, Thought: {thought}, Evaluation: {evaluation}\n"

#         for child, parent in self.tree["nodes"].items():
#             if parent == node:
#                 tree_info += self.print_tree(child, depth + 1)
#                 print(f'tree info: {tree_info}')

#         return tree_info






class TreeofThoughts:
    def __init__(self, model):
        self.model = model
        self.tree: Dict[str, Dict[str, Union[float, Dict[str, Any]]]] = {
            "nodes": {},
        }
        self.best_state = None
        self.best_value = float("-inf")
        self.history = [] #added line initalize history


    def save_tree_to_json(self, file_name):
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        with open(file_name, 'w') as json_file:
            json.dump(self.tree, json_file, indent=4)

    def logNewState(self, state, evaluation):
        if not (type(state) == str):
            state = " | ".join(state)
        if state in self.tree['nodes']:
            self.tree['nodes'][state]['thoughts'].append(evaluation)
        else:
            self.tree['nodes'][state] = {'thoughts': [evaluation]}

    def adjust_pruning_threshold_precentile(self, evaluated_thoughts, percentile):
        values = np.array(list(evaluated_thoughts.values()))
        if values.size == 0:
            return 0 
        return max(np.percentile(values, percentile), 0.1)
    

    def adjust_pruning_threshold_moving_average(self, evaluated_thoughts, window_size):
        values = list(evaluated_thoughts.values())
        if len(values) < window_size:
            return np.mean(values) if values else 0
        else:
            return max(np.mean(values[-window_size:]), 0.1)




class MonteCarloTreeofThoughts(TreeofThoughts):
    def __init__(self, model, objective="balance"):
        super().__init__(model)
        self.objective = objective
        self.solution_found = False
        self.tree: Dict[str, Dict[str, Union[float, Dict[str, Any]]]] = {
            "nodes": {},
            "metrics": {"thoughts": {}, "evaluations": {}},
        }


    def optimize_params(self, num_thoughts, max_steps, max_states):
        if self.objective == 'speed':
            num_thoughts = max(1, num_thoughts - 1)
            max_steps = max(1, max_steps - 1)
            max_states = max(1, max_states - 1)
        elif self.objective == 'reliability':
            num_thoughts += 1
            max_steps += 1
            max_states += 1
        elif self.objective == 'balanace':
            if self.solution_found:
                num_thoughts = max(1, num_thoughts - 1)
                max_steps = max(1, max_steps - 1)
                max_states = max(1, max_states - 1)
            else:
                num_thoughts += 1
                max_steps += 1
                max_states += 1
        
        return num_thoughts, max_steps, max_states

    def solve(self,
              initial_prompt: str,
              num_thoughts: int,
              max_steps: int,
              max_states: int,
              pruning_threshold: float,
            #   sleep_time: float,
              ):
        file_name = str(initial_prompt)
        self.file_name = f"logs/tree_of_thoughts_output_{file_name}.json"
        return self.monte_carlo_search(
            initial_prompt,
            num_thoughts,
            max_steps,
            max_states,
            pruning_threshold,
            # sleep_time,
        )
#v3
    def monte_carlo_search(self,
                        initial_prompt: str,
                        num_thoughts: int,
                        max_steps: int,
                        max_states: int,
                        pruning_threshold: float,
                        ):
        current_states = [initial_prompt]
        state_values = {}
        visit_counts = {initial_prompt: 0}
        transposition_table = {}

        best_state = None
        best_value = float('-inf')

        for step in range(1, max_steps + 1):
            selected_states = []

            for state in current_states:
                if state in transposition_table:
                    state_value = transposition_table[state]
                else:
                    time.sleep(1)
                    thoughts = self.model.generate_thoughts(state, num_thoughts, initial_prompt)
                    time.sleep(1)
                    evaluated_thoughts = self.model.evaluate_states(thoughts, initial_prompt)

                    for thought, value in evaluated_thoughts.items():
                        flattened_state = (state, thought) if isinstance(state, str) else (*state, thought)
                        transposition_table[flattened_state] = value

                for thought, value in evaluated_thoughts.items():
                    flattened_state = (state, thought) if isinstance(state, str) else (*state, thought)

                    if flattened_state not in visit_counts:
                        visit_counts[flattened_state] = 0

                    if visit_counts[state] > visit_counts[flattened_state] and visit_counts[flattened_state] > 0:
                        ucb1_value = value + np.sqrt(2 * np.log(visit_counts[state]) / visit_counts[flattened_state])

                        if ucb1_value >= pruning_threshold:
                            selected_states.append(flattened_state)
                            state_values[flattened_state] = value

                            # Update the best state if the current state value is greater than the best value
                            if value > best_value:
                                best_state = flattened_state
                                best_value = value

                visit_counts[state] += 1

            if len(selected_states) > max_states:
                current_states = selected_states[:max_states]
            self.save_tree_to_json(self.file_name)

        # if best_state is not None:
        #     solution = self.model.generate_solution(initial_prompt, best_state)
        #     return solution
        # else:
        #     solution = None

        # return None
        solution = self.model.generate_solution(initial_prompt, best_state)
        return solution if solution else best_state