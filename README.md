# Atropos - Nous Research's LLM RL Gym

![newatr-02](https://github.com/user-attachments/assets/e9b64e10-340e-48f2-835c-ae28fa14730a)

<div align="center">

*In Greek mythology, Atropos was the eldest of the three Fates.  While her sisters spun and measured the threads of mortal lives, Atropos alone held the shears that would cut these threads, determining the final destiny of each soul. Just as Atropos guided souls to their ultimate fate, this system guides language models toward their optimal potential through reinforcement learning.*

</div>

<div align="center">
</div>
<div id="badges" align="center">
  <a href="https://huggingface.co/NousResearch">
    <img src="https://img.shields.io/badge/NousResearch-orange?style=for-the-badge&logo=huggingface&logoColor=white" alt="HuggingFace"/>
  </a>
  <a href="https://nousresearch.com">
    <img src="https://img.shields.io/badge/NousResearch.com-white?style=for-the-badge&logo=data:image/png;base64,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" alt="Website"/>
  </a>
  <a href="https://x.com/NousResearch">
    <img src="https://img.shields.io/badge/@NousResearch-black?style=for-the-badge&logo=X&logoColor=white" alt="@NousResearch"/>
  </a>
</div>

Atropos is a Language Model Reinforcement Learning Environments framework for collecting and evaluating LLM trajectories through diverse environments including:

<div align="center">

| Environment Type          | Examples                                   | Purpose                                            |
|---------------------------|--------------------------------------------|----------------------------------------------------|
| 📚 Dataset environments   | GSM8K, MMLU                                | Evaluate and improve LLM performance on static data|
| 🎮 Online environments    | Crosswords, Hangman                        | Train LLMs through interactive game-based learning |
| 🤖 RLAIF and RLHF         | LLM Judge/Reward Models                    | Fine-tune LLMs using human feedback and alignment  |
| 🔄 Multi-Turn RL          | deepresearch, internal tool calling        | Train LLMs on complex multi-step interactions      |

</div>

Atropos is a robust, scalable framework for **Reinforcement Learning Environments with LLMs**. Key features:

- **Multi-Turn & Asynchronous RL:** Efficiently supports complex, multi-turn, and asynchronous interactions, decoupling environment steps from policy updates.
- **Inference Agnostic:** Integrates with standard inference APIs (e.g., OpenAI, vLLM, SGLang), enabling easy switching between LLM providers and frameworks.
- **Trainer Independent:** Offers a standardized training interface for experimenting with different RL algorithms and frameworks without major code changes.
- **Scalable & Decentralized:** Easily scale by launching more environment instances (locally or across decentralized resources) that contribute rollouts to a central service.
- **Diverse Environment Integration:** Manages many varied environment types concurrently for heterogeneous, multi-modal training.

The goal: provide a flexible, scalable, and standardized platform to accelerate LLM-based RL research across diverse, interactive settings.

## 🎉 Upcoming Atropos Hackathon: LLM RL Environments

Join us in San Francisco on May 18th, 2025 for an exciting hackathon focused on building and experimenting with LLM RL Environments! This in-person event will bring together researchers and developers interested in advancing the field of LLM reinforcement learning.

More details coming soon! Follow us on Twitter [@NousResearch](https://x.com/NousResearch) to stay updated.


---

## Experimental results from models trained using Atropos' environments

We have been able to achieve significant improvements on specific domains or tasks with Atropos - Below are some of the results.

**Tool Calling Environment Results:**

<div align="center">

| Berkeley Function Calling Benchmark Type | Base Model | With Atropos RL | Improvement |
|---------------|------------|-----------------|-------------|
| Parallel Tasks| 10%        | 46%            | **4.6x** ⬆️ |
| Simple Tasks  | 21%        | 51.75%         | **2.5x** ⬆️ |

</div>

Model Artifact:
https://huggingface.co/NousResearch/DeepHermes-ToolCalling-Specialist-Atropos


Environment Used:
https://github.com/NousResearch/atropos/blob/main/environments/tool_calling_server.py

---

**Financial Fundamentals Prediction Environment Results**:

<div align="center">

| Metric | Initial Accuracy | With Atropos RL | Improvement |
|--------|-----------------|-----------------|-------------|
| Directional Prediction Eval Accuracy | 20% | 50% | **2.5x** 📈 |

</div>

Model Artifact:
https://huggingface.co/NousResearch/DeepHermes-Financial-Fundamentals-Prediction-Specialist-Atropos

Environment Used:
https://github.com/NousResearch/atropos/blob/main/environments/fundamental_prediction_environment.py

---

## RLAIF Experiment Artifacts
Using the RLAIF Environment to change the personality of the model, we have produced several artifacts of interesting and weird personalities.

**DeepHermes Egregore v1 and v2 8B:**

https://huggingface.co/NousResearch/DeepHermes-Egregore-v1-RLAIF-8b-Atropos
https://huggingface.co/NousResearch/DeepHermes-Egregore-v2-RLAIF-8b-Atropos

**DeepHermes Ascension Maze 8B:**

https://huggingface.co/NousResearch/DeepHermes-AscensionMaze-RLAIF-8b-Atropos

---

## Navigating the Repo

| Category | Description |
|----------|------------|
| 📁 [`atroposlib/`](atroposlib/) | Core library containing base classes and utilities |
| 🎮 [`environments/`](environments/) | Collection of ready-to-use RL environments |
| 📚 [`example_trainer/`](example_trainer/) | Example training scripts and configurations |

Key Documents:
- [Base Environment Class](atroposlib/envs/README.md) - Documentation for creating custom environments
- [Environments Overview](environments/README.md) - Documentation for existing environments
- [Full Environment Config Options](CONFIG.md) - Documentation for creating custom environments
- [Example Trainer](example_trainer/README.md) - Getting started with training
- [Slurm Guide](SLURM.md) - Guide for using Atropos with Slurm for distributed inference
- [Contributing Guide](CONTRIBUTING.md) - Guidelines for contributors
- [License](LICENSE) - Apache 2.0 license details

---

## Installation

Get your Python 3.10 (or later) environment ready, then simply pip install:

```bash
pip install atroposlib
```

If you're looking to get into developing the repo or using the environments:


```bash
pip install -e .                # for using
pip install -e .[dev]           # for development
pip install -e .[examples]      # for running examples
pip install -e .[all]           # for everything
```

**Important:** If you're committing to the repository, please install the pre-commit hooks:
```bash
pre-commit install
```

---

### Quick Start Guide

1. **Create Your First Environment**
   - Review our [Base Class Documentation](atroposlib/envs/README.md) to understand the core concepts
   - Check out existing environments in the [`environments/`](environments) directory for examples

2. **Run an Example Environment**
   ```bash
   # Start the API server and run the GSM8K environment
   run-api & python environments/gsm8k_server.py serve \
       --tokenizer_name="Qwen/Qwen2.5-1.5B-Instruct" \
       --model_name="Qwen/Qwen2.5-1.5B-Instruct"
   ```

3. **Training Your Model**
   - Follow our [training example guide](example_trainer/README.md) for detailed instructions
   - Monitor progress through our built-in logging and reporting system:
     - Completion lengths
     - Evaluation accuracies
     - Full rollouts and scores

You can use multiple environments at once, just point them all to the same server.

Environments come with detailed logging and reporting support, runs track completion lengths, eval accuracies, full rollouts and scores, and more:

![image](https://github.com/user-attachments/assets/153a2932-191a-42e3-8da9-25a1b05abb8e)

---

## Debugging Tools

The trajectory-handler provides several debugging tools to help environment developers test and understand their environments locally without requiring the full distributed infrastructure.

*   **Flexible Model Provider Support:** Atropos natively supports any model provider that adheres to the OpenAI API standard. Simply provide the provider's base URL and your API key, and Atropos can integrate with their models seamlessly for testing or running environments locally.

After launching the API and your selected environments (e.g. `run-api & python environments/gsm8k_server.py serve`), you are then able to view them to get a quick look, or try to prepare some datasets for some offline training:

*   **View Run (`view-run`):** Launch a Gradio UI to inspect batches of rollouts generated by your environment runs. This is useful for visually debugging the interactions and data flow.
*   **Offline Data Generation:** Use `atropos-sft-gen` and `atropos-dpo-gen` to collect rollouts from environments and convert them into formats suitable for Supervised Fine-Tuning (SFT) or Direct Preference Optimization (DPO).

---

## Citation

If you have found the library helpful in your work, you can cite this repository as:

```latex
@misc{atropos,
  title = {{Atropos - An Async First Environment Rollout Controller}},
  author = {Dakota Mahan, Roger Jin, Teknium, Shannon Sands, Artem Yatsenko, Jai Suphavadeeprasit, Karan Malhotra, Chen Guang, Joe Li},
  url = {https://www.github.com/NousResearch/Atropos},
  month = {4},
  year = {2025},
  version = {0.1},
}
```

---

## Contributing

Atropos is built by the open-source AI community, and relies on our amazing contributors! Please see our [contributing](CONTRIBUTING.md) guide for more details on our code formatting, testing, etc.
Please follow the [Code of Conduct](CODE_OF_CONDUCT.md).

---

## License
Atropos is licensed as Apache 2.0, see the [LICENSE](LICENSE) file here for more information
