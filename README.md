# Meta TOT - Meta Tree of Thoughts
Meta TOT (Meta Tree of Thoughts) aims to enhance the Tree of Thoughts (TOT) language algorithm by using a secondary agent to critique and improve the primary agent's prompts. This innovative approach allows the primary agent to generate more accurate and relevant responses based on the feedback from the secondary agent.

# Architectural 

[Architectural overview in figma](https://www.figma.com/file/IlXAyYdu7HOYM4wUYVVB0U/META-TOT?type=whiteboard&node-id=0%3A1&t=xbtnQTA8uBhnLCU4-1)


# Introduction
The Meta TOT project aims to improve the performance of language models by using a secondary agent to critique and update the prompts of the primary agent. This approach allows the primary agent to generate more accurate and relevant responses based on the feedback from the secondary agent.

The project is built on top of the Tree of Thoughts (TOT) with an basic OPENAI Language model, which is a powerful and flexible framework for generating human-like responses in a conversational setting.

# Installation
To get started with Meta TOT, clone the repository and install the required dependencies:

```
git clone https://github.com/kyegomez/Meta-Tree-Of-Thoughts.git
cd Meta-Tree-Of-Thoughts
pip install -r requirements.txt
```

# Usage
After installing the dependencies, you can run the example script to see Meta TOT in action:

` python example.py `

This script demonstrates how the Meta TOT agent critiques and updates the prompts of the primary TOT agent to generate more accurate and relevant responses.

# Roadmap

Prototype Development: Develop a working prototype of the Meta TOT agent that can critique and update the prompts of the primary TOT agent with 3 simple classes language model -> <- meta agent -> Tree of Thoughts where the evaluated thoughts and evaluated state of that thought are passed into the meta agent as input and then the output of the meta agent is sent back to the primary agent.


Performance Evaluation: Evaluate the performance of the Meta TOT agent in various conversational settings and compare it to the baseline TOT agent.

Optimization: Implement various optimization techniques to improve the performance and efficiency of the Meta TOT agent.

Integration: Integrate the Meta TOT agent with popular chatbot frameworks and platforms.


# Optimizations
Model Compression: Use model compression techniques like pruning and quantization to reduce the size and computational requirements of the language models.

Caching: Implement caching mechanisms to store and reuse previously generated responses and critiques to improve response time and reduce computational overhead.

Parallelization: Leverage parallel processing techniques to speed up the evaluation and critique process.
Fine-tuning: Fine-tune the language models on domain-specific datasets to improve their performance in specific conversational contexts.

# Contributing
We welcome contributions from the community! If you're interested in contributing to the Meta TOT project, please check out the CONTRIBUTING.md file for guidelines and best practices.

# License
Meta TOT is released under the  Apache License License.



# Share with Friends
If you find Meta TOT interesting and useful, please share it with your friends and colleagues! You can use the following link to share the project:

Meta TOT - Meta Tree of Thoughts on GitHub

Together, we can build a powerful and innovative language model that generates more accurate and relevant responses in conversational settings.

