import asyncio
from concurrent.futures import ProcessPoolExecutor
from typing import Optional, Tuple

from datasets import load_dataset
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
from math_verify.errors import TimeoutException

from atroposlib.envs.eval import EvalBase, eval_runner, pass_at_k
from atroposlib.envs.server_handling.server_manager import ServerManager

hermes_system_prompt = (
    "You are a deep thinking AI, you may use extremely long chains of thought to deeply consider the "
    "problem and deliberate with yourself via systematic reasoning processes to help come to a correct "
    "solution prior to answering. You should enclose your thoughts and internal monologue inside <think> "
    "</think> tags, and then provide your solution or response to the problem."
)


def score_answer(gold, resp) -> Optional[bool]:
    try:
        gold_parsed = parse(
            gold,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
    except (Exception, TimeoutException, KeyError, TypeError, NotImplementedError):
        return None
    if len(gold_parsed) != 0:
        try:
            answer_parsed = parse(
                resp,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            boxed="all",
                            units=True,
                        ),
                        # Ensures that boxed is tried first
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )
        except (
            Exception,
            TimeoutException,
            KeyError,
            TypeError,
            NotImplementedError,
        ):
            # Can't parse, so we skip
            return None
        # Reward 1 if the content is the same as the ground truth, 0 otherwise
        try:
            return verify(answer_parsed, gold_parsed)
        except (
            Exception,
            TimeoutException,
            KeyError,
            TypeError,
            NotImplementedError,
        ):
            return None
    return None


class AIME24(EvalBase):
    """
    AIME24 Eval Environment

    kwargs:
        use_system_prompt (bool): Whether to use the system prompt in the evaluation.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mp_executor = ProcessPoolExecutor(8)

    def setup_data(self):
        aime_test_data = load_dataset("HuggingFaceH4/aime_2024", split="train")
        data = list()
        for item in aime_test_data:
            data.append(
                {
                    "problem": item["problem"],
                    "answer": item["answer"],
                }
            )
        return data

    async def run_item(
        self, server: ServerManager, data_item: dict
    ) -> Tuple[dict, list]:
        """
        An abstract method that must be overridden in a subclass to define how a
        specific item should be processed. This method encapsulates the logic required
        to run or process the given data item on the provided server instance.

        Args:
            server (ServerManager): An instance of ServerManager used to manage and
                interact with server operations during the item processing.
            data_item (dict): A dictionary representing the data item to be processed.
                The structure and content of the dictionary would depend on the
                specific application and use case.

        Returns:
            Tuple[dict, list]: A tuple where the first element is a dictionary
                containing the processed results or output of the operation, and
                the second element is a list containing any additional data generated
                or collected during the item's processing.
        """
        answer = data_item["answer"]
        question = data_item["problem"]
        use_sys_prompt = getattr(self, "use_system_prompt", False)
        async with server.managed_server() as managed:
            messages = (
                [{"role": "system", "content": hermes_system_prompt}]
                if use_sys_prompt
                else []
            )
            messages.append({"role": "user", "content": question})
            completion = await self.chat_completion(managed, messages)
        loop = asyncio.get_event_loop()
        gold = "\\boxed{" + answer + "}" if "\\boxed" not in answer else answer
        tasks = []
        for choice in completion.choices:
            resp = choice.message.content.split("</think>")[-1]
            tasks.append(
                loop.run_in_executor(self.mp_executor, score_answer, gold, resp)
            )
        rewards = await asyncio.gather(*tasks)
        rewards = [1.0 if reward else 0.0 for reward in rewards]
        passing = sum(rewards)
        n = self.get_generation_params()["n"]
        pass_at_k_val = getattr(self, "pass_at_k", 1)
        acc_at_k = pass_at_k(n, passing, getattr(self, "pass_at_k", 1))
        print(acc_at_k, n, passing, pass_at_k_val)
        key = f"pass@{pass_at_k_val}"
        if n != pass_at_k_val:
            key += f":{n}"
        return {
            key: acc_at_k,
        }, [
            {
                "messages": messages,
                "answer": choice.message.content,
                "score": rewards[i],
            }
            for i, choice in enumerate(completion.choices)
        ]


if __name__ == "__main__":
    asyncio.run(
        eval_runner(
            AIME24(
                pass_at_n=1,
                pass_at_n_samples=4,
                temperature=1.0,
                max_tokens=32768,
                use_system_prompt=True,
            )
        )
    )
