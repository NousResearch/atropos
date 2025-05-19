![Delegation RL](./images/image.png)

# Delegation RL
## Training LLMs to Delegate Tasks to Sub-Agents to Improve Performance
### Subjective Track
Multi-agent LLMs are becoming increasingly popular. Multi-agent frameworks enable LLMs to delegate tasks and execute them in parallel, which can significantly improve efficiency and performance. Current LLMs are not designed to handle multi-agent interactions natively. LLMs are finetuned to perform tasks in a single-agent setting and have limited capabilities when it comes to multi-agent interactions.

Delegation RL solves this problem by providing an environment for training an LLM to delegate tasks to sub-agents to improve its capabilities with the hope that a single user request could be delegated to multiple sub-agents, executing in parallel, and then the results can be aggregated to provide a final, more thorough response.

# How it Works
Delegation RL utilizes a simple hierarchical structure where a main agent (the delegator) can delegate tasks to multiple sub-agents (the delegates). Given a user request, the delegator breaks it down into 3-5 smaller tasks, which are then assigned to the sub-agents. Each sub-agent processes its task and returns a response. The delegator then aggregates these responses to provide a final answer to the user.

![Delegation RL vs Single Agent](./images/imagevs.png)

In order to evaluate the performance of the delegation system, we utilize LLM-as-a-Judge to evaluate the final response from the delegator compared with a response from a vanilla (single-agent) LLM. The judge scores each response and the delegator is scored based on the quality of its final response compared to the vanilla LLM response.

# Quickstart
To get started, run the following command and ensure that your OPENAI_API_KEY is set:
```bash
uv run environments/hack0/delegation_rl/delegation_env.py process
```

This will run the example delegation environment which uses GPT-4.1-nano for all models calls under the hood. To use different models, you can change the configurations in the `delegation_env.py` file. All output will be saved in `artifacts.json`. 


# Video
https://youtu.be/NxPz5DSyw50

# Wandb
https://api.wandb.ai/links/justinolcott-byu/30tnjy7d
- No added metrics

# Examples
## Good Example (this outperformed a single-agent)
**Examples of the Environment scoring a good example and a bad example:**
Blockchain technology is a decentralized digital ledger system that securely records transactions across multiple computers, ensuring no single entity has control over the entire database. Its core principles include decentralization\u2014distributing data to prevent centralized control; transparency\u2014allowing all participants to view transaction histories; and immutability\u2014once data is entered, it cannot be altered or deleted, maintaining the integrity of the records. These features make blockchain a secure, transparent, and tamper-proof method for managing digital information.\n\nBeyond its most well-known application in cryptocurrencies like Bitcoin, blockchain has a wide range of potential uses across various sectors:\n\n1. **Supply Chain Management:** Blockchain enhances transparency and traceability of products by recording each step in the supply chain. For example, Walmart uses blockchain to track food products, enabling quick identification of contamination sources, improving safety, and reducing fraud.\n\n2. **Healthcare Records:** Blockchain can securely store patient data, ensuring data integrity and privacy. It allows authorized healthcare providers to access complete and accurate records seamlessly. Projects like MedRec aim to give patients control over their health data while maintaining security.\n\n3. **Voting Systems:** Blockchain can facilitate transparent, tamper-resistant electronic voting, increasing trust in electoral processes. Pilot projects are exploring blockchain-based voting to ensure election integrity and prevent fraud.\n\nThese applications offer significant benefits, including increased transparency, improved traceability, enhanced security, reduced fraud, and streamlined processes, leading to higher trust and efficiency. However, implementing blockchain solutions also presents challenges. These include scalability limitations, high initial costs, lack of comprehensive regulatory frameworks, technical complexity, and resistance to change within organizations. Additionally, concerns about data privacy, energy consumption, and integration with existing systems need to be carefully managed.\n\nIn summary, blockchain technology holds transformative potential beyond cryptocurrencies, promising to revolutionize industries by providing secure, transparent, and efficient solutions for a variety of real-world applications.

## Bad Example (this underperformed a single agent):
Blockchain technology is a decentralized digital ledger system that records transactions across multiple computers, ensuring data integrity without the need for a central authority. Its core features include decentralization, transparency, and immutability. Decentralization distributes data across a network of nodes, reducing single points of failure and boosting security. Transparency allows all participants to verify transactions via a shared ledger, fostering trust. Immutability ensures that once data is recorded, it cannot be altered or deleted, maintaining the permanence and integrity of records.\n\nBeyond its well-known application in cryptocurrencies like Bitcoin, blockchain has numerous potential uses across various sectors:\n\n1. **Supply Chain Management:** Blockchain enables real-time, tamper-proof tracking of goods from their origin to the end consumer. This enhances transparency, reduces fraud, improves traceability, and increases efficiency by providing an immutable record of each step in the supply chain.\n\n2. **Healthcare Data Management:** Blockchain can securely store and share patient health records, ensuring data integrity and privacy. It allows for interoperability among different healthcare providers and empowers patients to control access to their information, improving the accuracy and security of medical data.\n\n3. **Voting Systems:** Blockchain can facilitate secure, transparent, and tamper-proof electronic voting. Its decentralized nature can help prevent fraud, enable verifiable audits, and boost public trust in electoral processes.\n\nWhile blockchain offers significant benefits such as increased transparency, security, and efficiency, implementing it outside of cryptocurrency faces challenges. These include high setup costs, scalability limitations, the need for clear regulatory frameworks, technical complexity, and resistance from organizations accustomed to traditional methods.\n\nOverall, blockchain technology holds transformative potential across many industries, promising more secure, transparent, and efficient systems beyond digital currencies.