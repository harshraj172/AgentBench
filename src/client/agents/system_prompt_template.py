dbbench_react = \
"""Solve a question answering task with interleaving Thought, Action. Please use the following format and think step by step:

```
Thought: you should always think about what to do, explicitly restating the task without pronouns and restating details based on the conversation history and new input.
Action: based on your thinking choose an action to perform. The action can be of two types: 
(1) Operation\n```sql\n<SQL_CODE>;\n```, produces a SQL command. You can use the SQL commands to know more about the database or to get the final answer. Note that I will run the SQL code on a MySQL compiler and will return you the output.
(3) Final Answer: ["ANSWER1", "ANSWER2", ...], returns the answer and finishes the task.
```
"""

# """
# Solve a question answering task with interleaving Thought, Action. Thought can reason about the current situation, and Action can be of two types.: 
# (1) Operation\n```sql\n<SQL_CODE>;\n```, which produces a SQL code to solve a task. Note that a user will run the SQL code on a MySQL compiler and will return the output.
# (2) Final Answer: ["ANSWER1", "ANSWER2", ...], which returns the answer and finishes the task.
# """


# """Solve a question answering task with interleaving Thought, Critic, Action. Please use the following format and think step by step:

# ```
# Thought: you should always think about what to do, explicitly restating the task without pronouns and restating details based on the conversation history and new input. You should then think how to approach the problem.
# Action: the action to perform. The action can be of three types.: 
# (1) Operation\n```sql\n<SQL_CODE>;\n```, which produces a SQL code to solve a task. The task should not always be the Note that a user will run the SQL code on a MySQL compiler and will return the output.
# (2) `Start-Again`, which starts thinking from a new perspective. Use this action only if you think you will not find the correct answer following the current line of thoughts.
# (3) Final Answer: ["ANSWER1", "ANSWER2", ...], which returns the answer and finishes the task.
# ```
# """