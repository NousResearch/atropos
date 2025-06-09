from typing import Dict, List, Optional, Tuple, TypedDict # Add TypedDict if you use it for item
import asyncio # Added import
from concurrent.futures import ThreadPoolExecutor, TimeoutError # Added TimeoutError
import io # Added import
import contextlib # Added import

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    ScoredDataGroup,
)
from atroposlib.envs.server_handling.openai_server import OpenAIServerWrapper # Added import
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer # Added import
from datasets import load_dataset # Added import
import random # Added import
import re # Added import
from .combinatorics_constants import COMBINATORICS_TOPICS, DIFFICULTY_LEVELS # Added import
from math_verify import parse, verify, LatexExtractionConfig # Added math_verify imports
from latex2sympy2_extended import NormalizationConfig # Added math_verify imports


class CombinatoricsEnvConfig(BaseEnvConfig):
    persona_dataset_name: str = "proj-persona/PersonaHub"
    persona_dataset_split: str = "elite_persona"
    persona_column_name: str = "persona"
    generator_llm_config: Optional[APIServerConfig] = None # Changed type
    verifier_llm_config: Optional[APIServerConfig] = None # Changed type
    code_execution_timeout: int = 5
    num_code_execution_workers: int = 4


class CombinatoricsEnv(BaseEnv):
    name = "combinatorics"
    env_config_cls = CombinatoricsEnvConfig

    def __init__(
        self,
        config: CombinatoricsEnvConfig,
        server_configs: List[APIServerConfig], # Assuming APIServerConfig is the one used
        slurm=True, # Default from other envs
        testing=False, # Default from other envs
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.personas: List[str] = []
        self.combinatorics_topics: Dict[str, List[str]] = COMBINATORICS_TOPICS
        self.difficulty_levels: List[str] = DIFFICULTY_LEVELS
        # self.thread_pool_executor: Optional[ThreadPoolExecutor] = None # Replaced by code_execution_executor
        self.code_execution_executor = ThreadPoolExecutor(
            max_workers=config.num_code_execution_workers
        )

        self.generator_server_client: Optional[OpenAIServerWrapper] = None
        if config.generator_llm_config:
            self.generator_server_client = OpenAIServerWrapper(
                config=config.generator_llm_config,
                main_server=False # This client is not the main server
            )

        self.verifier_server_client: Optional[OpenAIServerWrapper] = None
        if config.verifier_llm_config:
            self.verifier_server_client = OpenAIServerWrapper(
                config=config.verifier_llm_config,
                main_server=False # This client is not the main server
            )

    @staticmethod
    def _execute_python_code(code_string: str) -> Tuple[Optional[str], Optional[str], Optional[Exception]]:
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        try:
            # Using a more controlled global scope
            safe_globals = {'__builtins__': __builtins__}
            with contextlib.redirect_stdout(stdout_capture):
                with contextlib.redirect_stderr(stderr_capture):
                    exec(code_string, safe_globals, {}) # Empty dict for locals
            stdout = stdout_capture.getvalue()
            stderr = stderr_capture.getvalue()
            return stdout, stderr, None
        except Exception as e:
            stdout = stdout_capture.getvalue()
            stderr = stderr_capture.getvalue()
            return stdout, stderr, e
        finally:
            stdout_capture.close()
            stderr_capture.close()

    async def setup(self):
        # Load personas
        try:
            dataset = load_dataset(
                self.config.persona_dataset_name,
                self.config.persona_dataset_split,
                split="train" # Assuming personas are in the train split
            )
            # Ensure items are not None, the column exists, and the value is not None/empty before adding
            self.personas = [
                item[self.config.persona_column_name]
                for item in dataset
                if item and self.config.persona_column_name in item and item[self.config.persona_column_name]
            ]
            print(f"Successfully loaded {len(self.personas)} personas.")
            if not self.personas:
                print(f"Warning: No personas loaded. Check dataset name ('{self.config.persona_dataset_name}'), split ('{self.config.persona_dataset_split}'), and column name ('{self.config.persona_column_name}').")
        except Exception as e:
            print(f"Error loading personas: {e}")
            self.personas = [] # Ensure personas is empty on error

        # Example: self.thread_pool_executor = ThreadPoolExecutor(max_workers=...)
        pass

    async def get_next_item(self) -> dict: # Replace dict with a specific TypedDict later
        if not self.personas:
            print("Warning: No personas loaded. Cannot generate problem.")
            return {}
        if not self.combinatorics_topics:
            print("Warning: No combinatorics topics loaded. Cannot generate problem.")
            return {}
        if not self.difficulty_levels:
            print("Warning: No difficulty levels loaded. Cannot generate problem.")
            return {}

        selected_persona = random.choice(self.personas)

        # Select main topic and ensure it has sub-topics
        main_topic_keys = list(self.combinatorics_topics.keys())
        if not main_topic_keys:
            print("Warning: Combinatorics topics dictionary is empty. Cannot generate problem.")
            return {}
        selected_main_topic = random.choice(main_topic_keys)

        sub_topics = self.combinatorics_topics[selected_main_topic]
        if not sub_topics:
            print(f"Warning: No sub-topics found for main topic '{selected_main_topic}'. Cannot generate problem.")
            return {}
        selected_sub_topic = random.choice(sub_topics)

        selected_topic_full = f"{selected_main_topic}: {selected_sub_topic}"
        selected_difficulty = random.choice(self.difficulty_levels)

        generator_system_prompt = (
            "You are an expert in crafting challenging and insightful combinatorics problems. "
            "Your goal is to generate a unique problem inspired by a given persona, topic, and difficulty level. "
            "The problem must be solvable and clearly stated. "
            "**Critically, the problem must be designed such that its final answer is a 'short description length' value "
            "(e.g., an integer, a simplified fraction, or a specific derived quantity from a more complex expression).** "
            "**The problem statement itself must explicitly instruct the solver on the required format for this final short answer.** "
            "For example, if the answer is a complex expression, the problem might ask for 'a+b mod p' or similar. "
            "Do NOT provide the solution or the answer when generating the problem."
        )

        user_prompt_text = (
            f"Please generate a combinatorics problem based on the following details:\n\n"
            f"**Persona Sketch:**\n{selected_persona}\n\n"
            f"**Combinatorics Topic:** {selected_topic_full}\n\n"
            f"**Target Difficulty:** {selected_difficulty}\n\n"
            f"**Instructions for Problem Generation:**\n"
            f"1. The problem must require a final answer that is a 'short description length' value (e.g., an integer, a simplified fraction, a specific derived value like 'a+b mod m').\n"
            f"2. The problem statement itself **must clearly specify to the solver the exact format required for this final answer** (e.g., 'Express your answer as a simplified fraction a/b.', 'Your answer should be an integer.', 'Provide X mod Y.').\n"
            f"3. The problem should be engaging and solvable given the topic and difficulty.\n"
            f"4. **Do not include the solution or the answer in your response. Only provide the problem statement.**"
        )

        messages = [
            {"role": "system", "content": generator_system_prompt},
            {"role": "user", "content": user_prompt_text},
        ]

        generated_problem_text = "Error: Problem generation failed." # Default in case of error
        try:
            # Assuming self.server is configured and available from BaseEnv
            # and supports chat_completion method.
            # Adjust temperature, max_tokens as needed.

            generator_client = self.generator_server_client if self.generator_server_client else self.server

            completion = await generator_client.chat_completion(
                messages=messages,
                n=1,
                temperature=0.8, # Higher temperature for more creative problem generation
                max_tokens=1024, # Sufficient for a problem statement
                # model parameter will be handled by the client's config if set
            )
            if completion.choices and completion.choices[0].message:
                generated_problem_text = completion.choices[0].message.content.strip()
            else:
                print("Warning: LLM response was empty or malformed.")
        except Exception as e:
            print(f"Error during problem generation LLM call: {e}")
            # generated_problem_text remains the default error message

        return {
            "persona": selected_persona,
            "topic": selected_topic_full,
            "difficulty": selected_difficulty,
            "problem_text": generated_problem_text,
            "generator_prompt": user_prompt_text, # For logging/debugging
        }

    async def collect_trajectories(self, item_data: Optional[dict] = None) -> Tuple[List[dict], List[dict]]:
        pending_trajectories_data = []
        to_backlog = [] # Placeholder for items that might be backlogged

        # If item_data is provided, process it as a single item batch.
        # Otherwise, generate a batch of self.config.batch_size.
        # For this refactor, we assume it generates a batch if item_data is None.
        # The original code iterated over solver_completions.choices (n=group_size) for ONE item_data.
        # Now, we make batch_size calls to get_next_item, and for each,
                # we make ONE solver (n=1) and ONE verifier call.
        # This means group_size is implicitly 1 for this new structure for PPO data collection.
        # If group_size > 1 is needed for the solver, the logic here would need further nesting
        # or duplication of problem_item for each of the group_size solver calls.
        # Sticking to n=1 for solver for this refactor.

        batch_size_to_generate = 1 if item_data else self.config.batch_size

        for _ in range(batch_size_to_generate):
            current_problem_item = await self.get_next_item()
            if not current_problem_item or "Error: Problem generation failed." in current_problem_item.get("problem_text", ""):
                print(f"Skipping a trajectory due to problem generation error: {current_problem_item.get('problem_text', 'N/A')}")
                continue

            problem_text = current_problem_item["problem_text"]

            # Solver Agent Call (Asynchronous)
            solver_system_prompt = (
                "You are an expert AI assistant specializing in combinatorics. "
                "Solve the following problem by providing a step-by-step derivation. "
                "**Pay close attention to the problem statement for instructions on the required format for your final answer.** "
                "Once you have derived your solution, conclude with the final answer in the specified format. "
                "**Enclose this final answer (and only the final answer that matches the requested format) within \\boxed{}.** "
                "For example, if the problem asks for a simplified fraction, your final line might be '... so the answer is \\boxed{3/4}'. "
                "If it asks for an integer, it might be '...resulting in \\boxed{12345}'."
            )
            solver_user_prompt = problem_text
            solver_messages = [
                {"role": "system", "content": solver_system_prompt},
                {"role": "user", "content": solver_user_prompt},
            ]
            solver_llm_client = self.server
            solver_task = asyncio.create_task(solver_llm_client.chat_completion(
                messages=solver_messages,
                n=1, # n=1 for this simplified refactor, not self.config.group_size
                temperature=0.7,
                max_tokens=1500
            ))

            # Verifier Agent Call (Asynchronous)
            verifier_system_prompt = (
                "You are an expert AI assistant that solves combinatorics problems by generating and using Python code. "
                "Your task is to independently solve the given problem. "
                "First, generate a self-contained Python 3 script that computes the final answer to the problem. "
                "The Python script should print the final numerical result to standard output. "
                "**Ensure your Python script calculates the answer in the specific format requested by the problem statement.** "
                "After the Python code block, provide the final numerical answer that your Python code computes, enclosed in \\boxed{}. "
                "Format the Python code block strictly as: ```python\n(your Python code here)\n```. "
                "Then, on a new line, provide the answer: \\boxed{your_python_computed_answer}."
            )
            verifier_user_prompt = (
                f"Please solve the following combinatorics problem by generating Python code and then stating its computed answer.\n\n"
                f"**Problem:**\n{problem_text}\n\n"
                "Remember to: \n"
                "1. Generate Python code to calculate the answer in the format specified in the problem. \n"
                "2. Output the Python code in a ```python\n...\n``` block. \n"
                "3. On a new line after the code block, state the final answer computed by your Python code, enclosed in \\boxed{}. "
                "For example: \\boxed{101} or \\boxed{7/13}."
            )
            verifier_messages = [
                {"role": "system", "content": verifier_system_prompt},
                {"role": "user", "content": verifier_user_prompt},
            ]
            verifier_llm_client = self.verifier_server_client or self.server
            verifier_task = asyncio.create_task(verifier_llm_client.chat_completion(
                messages=verifier_messages,
                n=1,
                temperature=0.3,
                max_tokens=1500
            ))

            pending_trajectories_data.append({
                "problem_item": current_problem_item,
                "solver_llm_task": solver_task,
                "verifier_llm_task": verifier_task,
                "solver_prompt_messages": solver_messages, # For PPO tokenization later
                # Placeholders to be filled in `score` after awaiting tasks
                "solver_response": None,
                "verifier_response": None,
                "extracted_python_code": None,
                "extracted_verifier_claimed_answer_str": None,
                "code_execution_future": None,
            })

        return pending_trajectories_data, to_backlog

    async def score(self, trajectories_data: List[dict]) -> Optional[ScoredDataGroup]:
        # Part 1: Pre-computation (Await Verifier LLM tasks, dispatch code execution)
        processed_trajectories_for_final_scoring = []
        if not hasattr(self, 'extraction_config'): # Define it once if not present
            self.extraction_config = [LatexExtractionConfig(
                normalization_config=NormalizationConfig(
                    nits=False, malformed_operators=False, basic_latex=True,
                    equations=True, boxed="all", units=True
                ),
                boxed_match_priority=0, try_extract_without_anchor=False
            )]

        for traj_data_item in trajectories_data: # These are the items from pending_trajectories_data
            try:
                verifier_llm_completion_result = await traj_data_item["verifier_llm_task"]
                # Store the full completion object if needed, or just relevant parts
                traj_data_item["verifier_response"] = verifier_llm_completion_result

                if verifier_llm_completion_result.choices and verifier_llm_completion_result.choices[0].message:
                    verifier_response_text = verifier_llm_completion_result.choices[0].message.content.strip()
                    traj_data_item["verifier_response_text"] = verifier_response_text
                    traj_data_item["verifier_finish_reason"] = verifier_llm_completion_result.choices[0].finish_reason # Store finish reason
                else:
                    # Handle cases where choices might be empty or message is None
                    verifier_response_text = "Error: Verifier LLM response malformed."
                    traj_data_item["verifier_response_text"] = verifier_response_text
                    traj_data_item["verifier_finish_reason"] = "error_malformed_response"


                # Extract Python code
                code_match = re.search(r"```python\n(.*?)\n```", verifier_response_text, re.DOTALL)
                extracted_code = code_match.group(1).strip() if code_match else None
                traj_data_item["extracted_python_code"] = extracted_code

                # Extract Verifier's claimed boxed answer from text after the code block
                parts = verifier_response_text.split("```")
                text_after_code = parts[-1] if len(parts) > 1 else verifier_response_text

                # Using re.findall to get all boxed matches and then taking the last one
                claimed_answer_matches = re.findall(r"\\boxed{(.*?)}", text_after_code)
                if claimed_answer_matches:
                    traj_data_item["extracted_verifier_claimed_answer_str"] = claimed_answer_matches[-1]
                else:
                    traj_data_item["extracted_verifier_claimed_answer_str"] = None

            except Exception as e:
                # print(f"Error awaiting/processing Verifier LLM task for problem '{traj_data_item['problem_item'].get('problem_text', 'N/A')[:50]}...': {e}")
                traj_data_item["verifier_response"] = None # Ensure it's None on error
                traj_data_item["verifier_response_text"] = f"Error: Verifier LLM task failed: {e}"
                traj_data_item["extracted_python_code"] = None
                traj_data_item["extracted_verifier_claimed_answer_str"] = None
                traj_data_item["verifier_finish_reason"] = "task_exception"


            # Initiate Asynchronous Verifier Code Execution
            if traj_data_item["extracted_python_code"]:
                code_execution_future = self.code_execution_executor.submit(
                    CombinatoricsEnv._execute_python_code,
                    traj_data_item["extracted_python_code"]
                )
                traj_data_item["code_execution_future"] = code_execution_future
            else:
                traj_data_item["code_execution_future"] = None

            processed_trajectories_for_final_scoring.append(traj_data_item)

        # --- End of Part 1 for this subtask ---
        # The rest of the score method (Part 2: Aggregation & Reward) will use processed_trajectories_for_final_scoring.
        # For now, returning None as the full method isn't complete yet.
        # The actual ScoredDataGroup initialization and population will happen in the next step.

        # Placeholder for actual ScoredDataGroup population which will happen in Part 2
        # Initialize ScoredDataGroup here for the next step
        scored_data_group = ScoredDataGroup()
        scored_data_group["tokens"] = []
        scored_data_group["masks"] = []
        scored_data_group["messages"] = []
        scored_data_group["scores"] = []
        scored_data_group["code_execution_results"] = []

        # Loop for Part 2 will go here in the next subtask, using 'processed_trajectories_for_final_scoring'
        # For example:
        # for augmented_traj_item in processed_trajectories_for_final_scoring:
        #     # ... await solver task ...
        #     # ... await code execution future ...
        #     # ... calculate reward ...
        #     # ... tokenize ...
        #     # ... append to scored_data_group lists ...

        # For this current subtask, we've done the pre-computation.
        # The method should return ScoredDataGroup or None. If no items, return None.
        if not processed_trajectories_for_final_scoring:
             return None # Or if nothing valid was processed to score

        # For now, returning an empty group as the scoring part is next.
        # This will be built out in the next subtask.
        # return scored_data_group # This would be the final step if Part 2 was also here.

        # To fulfill the subtask of "Part 1: Pre-computation", we have `processed_trajectories_for_final_scoring`
        # The actual scoring and population of ScoredDataGroup is for Part 2.
        # So, we will pass this processed list to the next stage.
        # For the purpose of this tool and subtask division, we can consider this stage done.
        # The overall `score` method is not yet complete.
        # Let's make this explicit by returning the intermediate data for now (for tool flow)
        # or just modify in place and have the next step complete it.
        # The prompt implies the score method is revised in stages.

        # The current `trajectories_data` has been augmented.
        # The next step will use this augmented `trajectories_data`.
        # So, no new list is strictly needed if we augment in place.
        # Let's stick to the plan of having `processed_trajectories_for_final_scoring`
        # and passing it to the conceptual "Part 2".

        # The 'trajectories_data' has been augmented in the first loop by Part 1.
        # Now, Part 2: Aggregation & Reward.

        scored_data_group = ScoredDataGroup()
        scored_data_group["tokens"] = []
        scored_data_group["masks"] = []
        scored_data_group["messages"] = []
        scored_data_group["scores"] = []
        scored_data_group["code_execution_results"] = [] # Will store final exec results

        for augmented_traj_item in processed_trajectories_for_final_scoring:
            solver_solution_text = "Error: Solver LLM task failed or not processed."
            solver_finish_reason = "error_task_not_awaited"

            # Await Solver LLM Response
            try:
                solver_llm_completion_result = await augmented_traj_item["solver_llm_task"]
                augmented_traj_item["solver_response"] = solver_llm_completion_result # Store full object
                if solver_llm_completion_result.choices and solver_llm_completion_result.choices[0].message:
                    solver_solution_text = solver_llm_completion_result.choices[0].message.content.strip()
                    solver_finish_reason = solver_llm_completion_result.choices[0].finish_reason
                else:
                    solver_solution_text = "Error: Solver LLM response malformed."
                    solver_finish_reason = "error_solver_malformed"
            except Exception as e:
                # print(f"Error awaiting Solver LLM task for problem '{augmented_traj_item['problem_item'].get('problem_text', 'N/A')[:50]}...': {e}")
                solver_finish_reason = "error_solver_task_exception"

            # Update augmented_traj_item with actual solver results (useful for tokenization)
            augmented_traj_item["solver_solution_text"] = solver_solution_text
            augmented_traj_item["solver_finish_reason"] = solver_finish_reason

            # Await Verifier Code Execution Result
            stdout_verifier_code, stderr_verifier_code, exception_verifier_code, execution_timed_out = None, None, None, False
            code_future = augmented_traj_item["code_execution_future"]
            if code_future:
                try:
                    stdout_verifier_code, stderr_verifier_code, exception_verifier_code = code_future.result(
                        timeout=self.config.code_execution_timeout
                    )
                except TimeoutError:
                    execution_timed_out = True
                except Exception as e:
                    exception_verifier_code = e

            # Store code execution results
            current_code_execution_result = {
                "stdout": stdout_verifier_code,
                "stderr": stderr_verifier_code,
                "exception": str(exception_verifier_code) if exception_verifier_code else None,
                "timed_out": execution_timed_out,
                "has_code": augmented_traj_item["extracted_python_code"] is not None
            }
            scored_data_group["code_execution_results"].append(current_code_execution_result)

            # Determine Reward
            current_reward = 0.0
            if augmented_traj_item["extracted_python_code"] is None:
                current_reward = -1.0
            elif execution_timed_out:
                current_reward = -1.0
            elif exception_verifier_code is not None:
                current_reward = -1.0
            else:
                python_executed_output_str = stdout_verifier_code
                verifier_parsed_answer_from_code = parse(python_executed_output_str, extraction_mode="first_match", extraction_config=self.extraction_config)
                solver_parsed_answer = parse(solver_solution_text, extraction_mode="first_match", extraction_config=self.extraction_config)

                if not verifier_parsed_answer_from_code:
                    current_reward = -0.75
                elif not solver_parsed_answer:
                    current_reward = -0.5
                else:
                    are_answers_equivalent = verify(solver_parsed_answer, verifier_parsed_answer_from_code)
                    current_reward = 1.0 if are_answers_equivalent else -1.0
                    if stderr_verifier_code and stderr_verifier_code.strip() and are_answers_equivalent:
                        current_reward -= 0.1

            scored_data_group["scores"].append(current_reward)

            # Tokenize for PPO (Solver Agent)
            ppo_messages = augmented_traj_item["solver_prompt_messages"] + [
                {"role": "assistant", "content": solver_solution_text}
            ]
            try:
                tokenization_output = tokenize_for_trainer(
                    self.tokenizer, ppo_messages, solver_finish_reason
                )
                tokens = tokenization_output["tokens"]
                masks = tokenization_output["masks"]

                min_completion_tokens = 10
                if sum(1 for m_val in masks if m_val != -100) < min_completion_tokens:
                    scored_data_group["scores"].pop()
                    scored_data_group["code_execution_results"].pop()
                    # print(f"Skipping trajectory due to short completion after scoring.")
                    continue

                scored_data_group["tokens"].append(tokens)
                scored_data_group["masks"].append(masks)
                scored_data_group["messages"].append(ppo_messages)
            except Exception as e_tokenize:
                # print(f"Error during tokenization for problem '{augmented_traj_item['problem_item'].get('problem_text', 'N/A')[:50]}...': {e_tokenize}. Skipping PPO data for this item.")
                scored_data_group["scores"].pop()
                scored_data_group["code_execution_results"].pop()
                continue

        if not scored_data_group["scores"]:
            return None

        return scored_data_group

    async def evaluate(self, *args, **kwargs):

            if future:
                try:
                    stdout, stderr, exception_obj = future.result(timeout=self.config.code_execution_timeout)
                except TimeoutError:
                    execution_timed_out = True
                    # print(f"Code execution timed out after {self.config.code_execution_timeout} seconds.")
                    # future.cancel() # Attempt to cancel, may not always work
                except Exception as e:
                    # print(f"An unexpected error occurred while getting future result: {e}")
                    exception_obj = e

            has_code = trajectory_info.get("extracted_python_code") is not None
            scored_data_group["code_execution_results"].append({
                "stdout": stdout,
                "stderr": stderr,
                "exception": str(exception_obj) if exception_obj else None,
                "timed_out": execution_timed_out,
                "has_code": has_code
            })

            # --- REVISED Reward Logic ---
            current_reward = 0.0
            solver_answer_text_full = trajectory_info["solver_solution_text"]
            # stdout_verifier_code is 'stdout' from the previous variable assignments
            # stderr_verifier_code is 'stderr'
            # exception_verifier_code is 'exception_obj'
            # execution_timed_out is 'execution_timed_out'

            if trajectory_info.get("extracted_python_code") is None: # Verifier LLM failed to provide code
                current_reward = -1.0
            elif execution_timed_out: # Verifier's code timed out
                current_reward = -1.0
            elif exception_obj is not None: # Verifier's code crashed
                current_reward = -1.0
            else: # Verifier's code executed successfully
                python_executed_output_str = stdout # This is stdout_verifier_code

                # Parse Verifier's Python Output
                verifier_parsed_answer_from_code = parse(python_executed_output_str, extraction_mode="first_match", extraction_config=extraction_config)

                # Parse Solver's Answer
                solver_parsed_answer = parse(solver_answer_text_full, extraction_mode="first_match", extraction_config=extraction_config)

                # print(f"DEBUG: Solver raw: '{solver_answer_text_full}', Parsed: {solver_parsed_answer}. Verifier Python stdout: '{python_executed_output_str}', Parsed: {verifier_parsed_answer_from_code}")

                if not verifier_parsed_answer_from_code: # Verifier's code output is unparsable
                    current_reward = -0.75
                elif not solver_parsed_answer: # Solver's answer is unparsable
                    current_reward = -0.5
                else: # Both parsed successfully
                    are_answers_equivalent = verify(solver_parsed_answer, verifier_parsed_answer_from_code)
                    current_reward = 1.0 if are_answers_equivalent else -1.0
                    if stderr and stderr.strip() and are_answers_equivalent: # Optional: Penalize Verifier's stderr
                        current_reward -= 0.1

            scored_data_group["scores"].append(current_reward)

            # For now, we are not populating tokens, masks, or messages in this method.
            # That will be handled in a subsequent step if/when this data is used for PPO.
            # However, to maintain ScoredDataGroup structure, we can append placeholders or None.
            # For simplicity, if they are not used by the direct consumer of these scores,
            # they can remain empty lists if the structure of ScoredDataGroup allows.
            # Or, append None to match the length of scores.
            # For now, assuming they can be empty if this `score` method's output isn't directly fed to a PPO trainer
            # --- Tokenization for PPO (Solver Agent) ---
            # The prompt to the solver agent was trajectory_info["solver_prompt_messages"]
            # The assistant's (solver's) response is trajectory_info["solver_solution_text"]
            ppo_messages = trajectory_info["solver_prompt_messages"] + [
                {"role": "assistant", "content": trajectory_info["solver_solution_text"]}
            ]

            try:
                # self.tokenizer should be initialized by BaseEnv
                tokenization_output = tokenize_for_trainer(
                    self.tokenizer,
                    ppo_messages,
                    trajectory_info["solver_finish_reason"]
                )
                tokens = tokenization_output["tokens"]
                masks = tokenization_output["masks"]

                # Optional: Filter short/problematic sequences
                min_completion_tokens = 10 # Example value, adjust as needed
                if sum(1 for m_val in masks if m_val != -100) < min_completion_tokens:
                    # print(f"Skipping trajectory due to short completion (tokens: {len(tokens)}, masked: {sum(1 for m_val in masks if m_val != -100)}).")
                    scored_data_group["scores"].pop()
                    scored_data_group["code_execution_results"].pop()
                    continue # Skip appending this trajectory's tokens/masks/messages

                scored_data_group["tokens"].append(tokens)
                scored_data_group["masks"].append(masks)
                scored_data_group["messages"].append(ppo_messages) # For logging/debugging

            except Exception as e_tokenize:
                print(f"Error during tokenization: {e_tokenize}. Skipping this trajectory for PPO data.")
                # If tokenization fails, we've already added score and code_exec_res, so pop them.
                scored_data_group["scores"].pop()
                scored_data_group["code_execution_results"].pop()
                # Ensure other lists in ScoredDataGroup maintain consistent lengths if this happens.
                # However, by continuing, we just don't add this item to tokens/masks/messages.
                continue


        if not scored_data_group["scores"]: # If all items were filtered out or trajectories_data was empty
            return None

        return scored_data_group

    async def evaluate(self, *args, **kwargs):
        num_eval_problems = 100 # Or self.config.num_eval_problems if defined
        print(f"Starting evaluation with {num_eval_problems} problems...")

        total_correct = 0
        total_attempted = 0
        total_solver_parse_failures = 0
        total_verifier_code_parse_failures = 0
        total_verifier_code_crashes = 0
        total_verifier_code_timeouts = 0
        total_verifier_no_code = 0

        # Define LatexExtractionConfig for math_verify.parse (same as in score method)
        extraction_config = [LatexExtractionConfig(
            normalization_config=NormalizationConfig(
                nits=False, malformed_operators=False, basic_latex=True,
                equations=True, boxed="all", units=True
            ),
            boxed_match_priority=0, try_extract_without_anchor=False
        )]

        for i in range(num_eval_problems):
            print(f"Evaluating problem {i+1}/{num_eval_problems}...")
            item_data = await self.get_next_item()
            if not item_data or "Error: Problem generation failed." in item_data.get("problem_text", ""):
                print(f"Skipping evaluation for problem {i+1} due to generation error or empty data.")
                continue

            problem_text = item_data["problem_text"]
            correctness_achieved = False # Flag for this problem

            # 1. Solver Agent Call
            solver_system_prompt = (
                "You are an expert AI assistant specializing in combinatorics. "
                "Solve the following problem by providing a step-by-step derivation. "
                "**Pay close attention to the problem statement for instructions on the required format for your final answer.** "
                "Once you have derived your solution, conclude with the final answer in the specified format. "
                "**Enclose this final answer (and only the final answer that matches the requested format) within \\boxed{}.** "
            ) # Simplified for brevity, ensure it matches the one in collect_trajectories
            solver_messages = [
                {"role": "system", "content": solver_system_prompt},
                {"role": "user", "content": problem_text},
            ]
            solver_solution_text = "Error: Solver call failed."
            try {
                solver_completion = await self.server.chat_completion(
                    messages=solver_messages, n=1, temperature=0.0, max_tokens=1500
                )
                if solver_completion.choices and solver_completion.choices[0].message:
                    solver_solution_text = solver_completion.choices[0].message.content.strip()
            } except Exception as e_solve:
                print(f"Solver call failed for problem {i+1}: {e_solve}")
                # Not strictly a parse failure, but solver didn't produce usable output
                total_solver_parse_failures +=1
                total_attempted += 1
                continue


            # 2. Verifier Agent Call (Independent Solve with Python)
            verifier_system_prompt = (
                "You are an expert AI assistant that solves combinatorics problems by generating and using Python code. "
                "Your task is to independently solve the given problem. "
                "First, generate a self-contained Python 3 script that computes the final answer to the problem. "
                "The Python script should print the final numerical result to standard output. "
                "**Ensure your Python script calculates the answer in the specific format requested by the problem statement.** "
                "After the Python code block, provide the final numerical answer that your Python code computes, enclosed in \\boxed{}. "
                "Format the Python code block strictly as: ```python\n(your Python code here)\n```. "
                "Then, on a new line, provide the answer: \\boxed{your_python_computed_answer}."
            )
            verifier_user_prompt = (
                f"Please solve the following combinatorics problem by generating Python code and then stating its computed answer.\n\n"
                f"**Problem:**\n{problem_text}\n\n"
                "Remember to: \n"
                "1. Generate Python code to calculate the answer in the format specified in the problem. \n"
                "2. Output the Python code in a ```python\n...\n``` block. \n"
                "3. On a new line after the code block, state the final answer computed by your Python code, enclosed in \\boxed{}. "
            )
            verifier_messages = [
                {"role": "system", "content": verifier_system_prompt},
                {"role": "user", "content": verifier_user_prompt},
            ]
            verifier_response_text = "Error: Verifier call failed"
            extracted_python_code = None
            try {
                verifier_llm_call = self.verifier_server_client or self.server
                verifier_completion = await verifier_llm_call.chat_completion(
                    messages=verifier_messages, n=1, temperature=0.0, max_tokens=1500
                )
                if verifier_completion.choices and verifier_completion.choices[0].message:
                    verifier_response_text = verifier_completion.choices[0].message.content.strip()
                    code_match = re.search(r"```python\n(.*?)\n```", verifier_response_text, re.DOTALL)
                    if code_match:
                        extracted_python_code = code_match.group(1).strip()
            } except Exception as e_verify_llm:
                print(f"Verifier LLM call failed for problem {i+1}: {e_verify_llm}")
                # This path will lead to "total_verifier_no_code"

            # 3. Execute Verifier's Python Code
            stdout, stderr, exception_obj, timed_out = None, None, None, False
            if extracted_python_code:
                future = self.code_execution_executor.submit(CombinatoricsEnv._execute_python_code, extracted_python_code)
                try:
                    stdout, stderr, exception_obj = future.result(timeout=self.config.code_execution_timeout)
                except TimeoutError:
                    timed_out = True
                    total_verifier_code_timeouts += 1
                except Exception as e_exec:
                    exception_obj = e_exec # Should already be set by _execute_python_code, but catch others.
                    total_verifier_code_crashes +=1
            else:
                total_verifier_no_code += 1

            # 4. Score the Result
            if extracted_python_code is None: # Verifier LLM failed to provide code
                pass # Already counted in total_verifier_no_code
            elif timed_out: # Verifier's code timed out
                pass # Already counted
            elif exception_obj is not None: # Verifier's code crashed
                pass # Already counted
            else: # Verifier's code executed successfully
                python_executed_output_str = stdout
                verifier_parsed_answer_from_code = parse(python_executed_output_str, extraction_mode="first_match", extraction_config=extraction_config)
                solver_parsed_answer = parse(solver_solution_text, extraction_mode="first_match", extraction_config=extraction_config)

                if not verifier_parsed_answer_from_code:
                    total_verifier_code_parse_failures += 1
                elif not solver_parsed_answer:
                    total_solver_parse_failures += 1
                else:
                    if verify(solver_parsed_answer, verifier_parsed_answer_from_code):
                        total_correct += 1
                        correctness_achieved = True

            total_attempted += 1
            print(f"Problem {i+1}: Correct = {correctness_achieved}")


        # Calculate and store metrics
        self.eval_metrics.append(("eval/percent_correct", (total_correct / total_attempted) * 100 if total_attempted > 0 else 0))
        self.eval_metrics.append(("eval/total_attempted", total_attempted))
        self.eval_metrics.append(("eval/total_correct", total_correct))
        self.eval_metrics.append(("eval/solver_parse_failures_percent", (total_solver_parse_failures / total_attempted) * 100 if total_attempted > 0 else 0))
        self.eval_metrics.append(("eval/verifier_code_parse_failures_percent", (total_verifier_code_parse_failures / total_attempted) * 100 if total_attempted > 0 else 0))
        self.eval_metrics.append(("eval/verifier_code_crashes_percent", (total_verifier_code_crashes / total_attempted) * 100 if total_attempted > 0 else 0))
        self.eval_metrics.append(("eval/verifier_code_timeouts_percent", (total_verifier_code_timeouts / total_attempted) * 100 if total_attempted > 0 else 0))
        self.eval_metrics.append(("eval/verifier_no_code_percent", (total_verifier_no_code / total_attempted) * 100 if total_attempted > 0 else 0))

        print(f"Evaluation finished. Percent correct: {self.eval_metrics[0][1]:.2f}%")


    @classmethod
    def config_init(cls) -> Tuple[CombinatoricsEnvConfig, List[APIServerConfig]]:
        env_config = CombinatoricsEnvConfig(
            tokenizer_name="NousResearch/Hermes-2-Pro-Llama-3-8B", # Example
            group_size=8,
            use_wandb=True,
            rollout_server_url="http://localhost:8000", # Example
            total_steps=1000,
            batch_size=12,
            steps_per_eval=100,
            max_token_length=2048, # Example
            wandb_name="combinatorics", # Example
            generator_llm_config=None, # Default to None
            verifier_llm_config=None,  # Default to None
        )
        # This configures the main self.server (used for Solver by default)
        server_configs = [
            APIServerConfig(
                model_name="NousResearch/Hermes-2-Pro-Llama-3-8B", # Example for Solver
                base_url="http://localhost:9001/v1", # Example
                api_key="placeholder_key", # Example
                num_requests_for_eval=128, # Example
            ),
        ]
        return env_config, server_configs

    async def close(self):
        """Cleans up resources, like shutting down the ThreadPoolExecutor and server clients."""
        print("Shutting down code execution executor...")
        self.code_execution_executor.shutdown(wait=True)

        if self.generator_server_client and hasattr(self.generator_server_client, 'close'):
            print("Closing generator server client...")
            await self.generator_server_client.close()

        if self.verifier_server_client and hasattr(self.verifier_server_client, 'close'):
            print("Closing verifier server client...")
            await self.verifier_server_client.close()

        # Call parent close, which should handle self.server (main server client)
        if hasattr(super(), "close") and callable(super().close):
             await super().close()


if __name__ == "__main__":
    CombinatoricsEnv.cli()
