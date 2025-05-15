To use a prebuilt agent with RAG over memories (such as in the TextWorld environment) and internal management of it's own trajectories and any logic non specific to an environment, install FAISS first. 
`pip install faiss-cpu` or `pip install -r requirments.txt`

Agents are purely optional, as helpers to keep the environments cleaner, especially in environments with multiple players. Feel free to create your own or ignore entirely and have all LLM logic

## AtroposAgent
Has the ability to force (via prefill) thinking tokens, add some specific system prompt to anything coming from the environment, add memories to incoming observations (using FAISS for a light, in-memory data store), summarise it's own thinking blocks (to truncate excessive verbiage), and standardise it's tool calling and parsing. For now uses completions APIs to remain compatible with most servers and role templates.

## AtroposRM

Can be used by AtroposAgent or an environment to provide Q value estimates on actions (similar to other LLM-as-a-judge examples such as `math_server.py`). It's intended to allow for using a generative reward model (ie, it's trained to produce thinking blocks and a float value to represent the Q score). If used directly in an environment, you can even use it's judgements to co-train the GenRM, which can even be the same LLM you're using as the policy model - take a look the Textworld example to see how this is done