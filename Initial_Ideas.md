## The Idea - Combiine the Training of LLM via Atropos with the Digital Red Queen: Adversarial Program Evolution in Core War with LLMs

This would upgrade the ability of LLMs to evolve and adapt to new challenges by competing against each other in simulated environments. As I understand no one else allows the AI to train itself against other AIs in a competitive environment at the LLM level, explore how we can use gaming and war to improve the LLMs' ability to adapt and evolve.


Train LLMs (with Atropos) to compete against each other in simulated environments. 

We are also able to train LLMs with the Atropos framework, which is a robust, scalable framework for Reinforcement Learning Environments with LLMs.

The goal: provide a flexible, scalable, and standardized platform to accelerate LLM-based RL research across diverse, interactive settings.

The framework supports collecting, distributing and evaluating LLM trajectories through diverse environments including:

Atropos is a robust, scalable framework for Reinforcement Learning Environments with LLMs.

The goal: provide a flexible, scalable, and standardized platform to accelerate LLM-based RL research across diverse, interactive settings.

The framework supports collecting, distributing and evaluating LLM trajectories through diverse environments including:

📚 Dataset environments	GSM8K, MMLU, Custom HF Datasets	Evaluate and improve LLM performance on static data
🎮 Online environments	Blackjack, Taxi, Text-based games	Train LLMs through interactive game-based learning
🤖 RLAIF and RLHF	LLM Judge/Reward Models	Fine-tune LLMs using human feedback and alignment
🔄 Multi-Turn RL	deepresearch, internal tool calling	Train LLMs on complex multi-step interactions
💻 Code Execution	MBPP, HumanEval (via coding_server.py)	Train LLMs to generate and execute code
🖼️ Multimodal	OCR VQA, Clevr (via multimodal_dpo/)	Train LLMs on tasks involving vision and language


To understand the future of the world, stick AI systems in a petri dish:

…Evolving LLMs to attack other LLMs…
Researchers with Japanese AI startup Sakana have looked at what happens when they evolve LLM-based agents to fight against one another in a competitive programming game from the 1980s called Core War. The results show that “large language models (LLMs) drive an adversarial evolutionary arms race in this domain, where programs continuously adapt to defeat a growing history of opponents rather than a static benchmark”. This research approach gestures both at ways researchers might better study how LLM-dominated niches in the economy or national security world might unfold, and also hints at the strange AI world we’re heading into.

What is Core War? “Core War is a competitive programming game played out in a shared block of computer memory, called the “Core,” where two or more assembly programs fight for survival”, Sakana writes. “Each program, known as a “warrior”, is written in an assembly language called Redcode. These programs are tasked with crashing their competitors while keeping their own processes alive. The simulation runs by alternating between the programs, executing one instruction at a time. A warrior “attacks” by writing invalid instructions (DAT commands) into the memory slots occupied by opponents, causing them to crash upon execution.”

DRQ: To evolve their programs, the authors use a technique they call Digital Red Queen. “DRQ uses MAP-Elites, a quality-diversity algorithm, to optimize warriors within each round, preventing diversity collapse during search. By playing against all previous round champions, DRQ avoids cyclic adaptations across rounds, consistent with techniques in prior work”, they write. “We find that as DRQ is run for many rounds, warriors gradually become more generally robust, as measured by their performance against unseen human-designed warriors.”
Each warrior calls out to GPT-4 mini (”preliminary experiments did not show significant performance increase with larger models), and is given a prompt which describes the Core War environment as well as a manual for the Redcode assembly language. “To generate a new warrior, the LLM is given a user prompt instructing it to produce a novel Redcode program. To mutate an existing warrior, the LLM is provided with the original program and instructed to modify it in ways that could improve performance.”

Evolution works: Unsurprisingly, evolving agents is very effective:

A one-shot warrior defeats 1.7% of human warriors.

Best-of-N sampling produces a set of warriors that can defeat 22.1% of human warriors

“Evolutionary optimization against each human warrior generates a specialized warrior for every opponent; this set can collectively defeat 89.1% of human warriors and defeat or tie 96.3%.”

Why this matters - where Core Wars goes, so does the world: The world is going to look a lot like Core Wars - millions of AI agents will be competing against one another in a variety of domains, ranging from cybersecurity to economics, and will be optimizing themselves in relation to achieving certain competitive criteria. The result will be sustained, broad evolution of AI systems and the software harnesses and tooling they use to get stuff done. This means that along with human developers and potential AI-designed improvements, we’ll also see AI systems improve from this kind of broad competitive pressure.

“The cybersecurity arms race between offense and defense is well underway,” Sakana writes. “Studying these adversarial dynamics in an artificial testbed like Core War offers critical insights into how such races might unfold and the kinds of strategies that may emerge.”

Read the blog post: Digital Red Queen: Adversarial Program Evolution in Core War with LLMs (Sakana).
https://sakana.ai/drq/ January 08, 2026

Sakana AI
We are taking this technology far beyond adversarial competitive programming to unlock a new era of AI-driven discovery.

OTHER works by Sakana AI:

https://sakana.ai/ai-scientist/ - The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery

https://sakana.ai/llm-squared/ - Can LLMs invent better ways to train LLMs?

https://sakana.ai/shinka-evolve/ - ShinkaEvolve: Evolving New Algorithms with LLMs, Orders of Magnitude More Efficiently September 25, 2025

https://sakana.ai/asal/ - Automating the Search for Artificial Life

https://sakana.ai/ahc058/ - Sakana AI Agent Wins AtCoder Heuristic Contest (First AI to Place 1st)


Sakana AI is at the forefront of AI-driven discovery. 

