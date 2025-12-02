from copy import deepcopy
import json
import re
import random
import os
import yaml
import math

from typing import List, Union, Dict, Any, Callable

from .base import ToolEvalEvaluator
from .utils import register_evaluator

from tenacity import retry, stop_after_attempt, wait_exponential

import google.generativeai as genai

config_file = 'toolbench/tooleval/evaluators/tooleval_gemini/config.yaml'
CONFIG = yaml.load(open(config_file, 'r'), Loader=yaml.FullLoader)

class _DummyMessage:
    def __init__(self, content: str):
        self.content = content


class _DummyChoice:
    def __init__(self, message: _DummyMessage):
        self.message = message


class _DummyResponse:
    """
    Minimal wrapper so we can still do:
        res.choices[0].message.content
    like with OpenAI ChatCompletion.
    """
    def __init__(self, choices: List[_DummyChoice]):
        self.choices = choices


class GeminiClient:
    """
    Simple single-key Gemini client that mimics the OpenAI chat API shape
    expected by the evaluators.
    """

    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or CONFIG["api_key"]
        self.model_name = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY not set. "
                "Set GEMINI_API_KEY env var or pass api_key to GeminiClient."
            )

        genai.configure(api_key=self.api_key)

    @staticmethod
    def _messages_to_prompt(messages: List[Dict[str, str]]) -> str:
        """
        Convert OpenAI-style messages into a single text prompt.
        """
        parts = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            parts.append(f"{role.upper()}: {content}")
        return "\n\n".join(parts)

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> _DummyResponse:
        """
        Emulates OpenAI's ChatCompletion.create:
          - messages: list of {role, content}
          - kwargs: can include temperature, max_tokens, n, model, etc.
        Returns a _DummyResponse with .choices[...].message.content.
        """
        prompt = self._messages_to_prompt(messages)

        # Per-call model override
        model_name = kwargs.pop("model", None) or self.model_name

        temperature = kwargs.pop("temperature", None)
        max_tokens = kwargs.pop("max_tokens", None) or kwargs.pop(
            "max_output_tokens", None
        )
        n = kwargs.pop("n", 1)

        generation_config = {}
        if temperature is not None:
            generation_config["temperature"] = temperature
        if max_tokens is not None:
            generation_config["max_output_tokens"] = max_tokens
        if n is not None and n > 0:
            generation_config["candidate_count"] = n

        if not generation_config:
            generation_config = None

        model = genai.GenerativeModel(model_name)
        resp = model.generate_content(
            prompt,
            generation_config=generation_config,
        )

        choices: List[_DummyChoice] = []
        if resp.candidates:
            for cand in resp.candidates:
                text = ""
                for part in cand.content.parts:
                    if hasattr(part, "text"):
                        text += part.text
                choices.append(_DummyChoice(_DummyMessage(text.strip())))
        else:
            text = getattr(resp, "text", "") or ""
            if not text:
                raise RuntimeError("Gemini returned no text")
            choices.append(_DummyChoice(_DummyMessage(text.strip())))

        return _DummyResponse(choices)


@register_evaluator
class GeminiEvaluator(ToolEvalEvaluator):
    """
    Evaluator that now uses Gemini (via GeminiClient) as the judge.

    Expects the model to output JSON like:
      {"preference": 0}  or  {"preference": 1}
    in the content of each candidate.
    """

    def __init__(self,
                 cfg_path: str = None,
                 ):
        super().__init__(cfg_path)

        # Single Gemini client; key/model come from env by default
        self.client = GeminiClient()

        self.conversation_template = []
        for message in re.findall(r"<message>(.*?)</message>", self.template, re.DOTALL):
            message = {
                'role': re.findall(r"<role>(.*?)</role>", message, re.DOTALL)[0],
                'content': re.findall(r"<content>(.*?)</content>", message, re.DOTALL)[0]
            }
            self.conversation_template.append(message)

    def gemini_completions(self, task_description: Dict, answers: Dict) -> int:
        """
        Calls Gemini once (possibly with n>1) and parses 'preference' from JSON.

        Returns:
            int: chosen preference index.
        """
        conversation = deepcopy(self.conversation_template)
        for msg in conversation:
            if msg['role'] == 'user':
                msg['content'] = msg['content'].format(
                    task_description=json.dumps(task_description),
                    answers=json.dumps(answers)
                )

        res = self.client.chat(
            messages=conversation,
            **self.eval_config['completions_kwargs']
        )

        prefers = []
        for choice in res.choices:
            data = json.loads(choice.message.content)
            prefers.append(int(data['preference']))

        return random.choice(prefers)


@register_evaluator
class GeminiNormalizedEvaluator(ToolEvalEvaluator):
    """
    Normalized evaluator now using Gemini via GeminiClient.

    All "function calls" are done by:
      - Building a textual prompt with the function template + JSON schema
      - Asking Gemini to output a single JSON object
      - Parsing that JSON from message.content
    """

    def __init__(self,
                 cfg_path: str = None,
                 ):
        super().__init__(cfg_path)

        self.client = GeminiClient()

        # setting up the function templates
        self.parsed_function_templates = {}
        for function in re.findall(r"<function>(.*?)</function>", self.template, re.DOTALL):
            name = re.findall(r"<name>(.*?)</name>", function, re.DOTALL)[0]
            description = re.findall(r"<description>(.*?)</description>", function, re.DOTALL)[0]
            self.parsed_function_templates[name] = description

        # Store function schemas (we only use them for schema text in the prompt)
        self.functions = {}
        for function in self.eval_config['completions_kwargs'].get('functions', []):
            self.functions[function['name']] = function

    @retry(
        stop=stop_after_attempt(3),
        reraise=True,
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def function_call(self,
                      func_name: str,
                      func_args: Dict,
                      *,
                      return_reason: bool = False,
                      return_content: bool = False):
        """
        Simulate an OpenAI "tool call" by asking Gemini to return pure JSON.

        func_name: name of the logical function (e.g. 'check_solve_query')
        func_args: arguments to plug into the template via str.format(**func_args)

        Returns:
            dict parsed from the JSON the model returns.
        """
        completion_kwargs = deepcopy(self.eval_config['completions_kwargs'])

        func_description = deepcopy(self.functions[func_name])

        if return_reason:
            func_description['parameters']['required'].append('reason')
            func_description['parameters']['properties']['reason'] = {
                'type': 'string',
                'description': 'explain your answer.'
            }

        eval_model = os.getenv('EVAL_MODEL', None)
        if eval_model:
            completion_kwargs['model'] = eval_model

        # Remove OpenAI-specific 'functions' key if present
        completion_kwargs.pop('functions', None)

        func_template = str(self.parsed_function_templates[func_name])
        user_task = func_template.format(**func_args)

        schema_str = json.dumps(func_description.get('parameters', {}), indent=2)
        prompt = (
            "You are a JSON-only API.\n"
            "Given the following JSON schema and task, respond with a single JSON object "
            "that strictly follows the schema. Do not include backticks. Do not include any extra text.\n\n"
            f"SCHEMA:\n{schema_str}\n\n"
            f"TASK:\n{user_task}\n"
        )

        completion_kwargs['messages'] = [{
            'role': 'user',
            'content': prompt
        }]

        res = self.client.chat(
            messages=completion_kwargs['messages'],
            **{k: v for k, v in completion_kwargs.items() if k != 'messages'}
        )

        content = res.choices[0].message.content.strip()

        def strip_code_fences(text):
            # remove ```json ... ``` or ``` ... ```
            text = re.sub(r"```json\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"```", "", text)
            return text.strip()

        try:
            ret = json.loads(content)
        except json.JSONDecodeError:
            cleaned_content = strip_code_fences(content)
            try:
                ret = json.loads(cleaned_content)
            except json.JSONDecodeError:
                if func_name in ["check_answer_status", "parse_answer_status"]:
                    return {
                        "answer_status": "Unsolved",
                        "reason": ""
                    }
                elif func_name == "select_best_final_answer":
                    return {"best_answer_index": 0}
                else:
                    return {}

        # check required items
        required_args = func_description.get('parameters', {}).get('required', None)
        if required_args is not None:
            ret_args = set(ret.keys())
            for arg in required_args:
                if arg not in ret_args:
                    raise KeyError(f"Arg {arg} not found in reply!")

        if return_content:
            ret['content'] = content
        return ret

    def select_best_final_answer(self, query, final_answers: List[str]) -> int:
        hashed_ans = list(map(hash, final_answers))
        all_same = True
        for item in hashed_ans[1:]:
            if item != hashed_ans[0]:
                all_same = False
        if all_same:
            return random.choice(range(len(final_answers)))
        while True:
            selected = int(
                self.function_call(
                    'select_best_final_answer',
                    {'query': query, 'final_answers': final_answers}
                )['best_answer_index']
            )
            if 0 <= selected < len(final_answers):
                break
        return selected

    def check_solve_query(self, query, final_answer: str) -> bool:
        return bool(
            self.function_call(
                'check_solve_query',
                {'query': query, 'final_answer': final_answer}
            )['is_solved']
        )

    def compare_answer_details(self, answer: List) -> List[int]:
        parsed_answers = []

        for ans in answer:
            parsed_ans = self.function_call(
                'parse_answer_details',
                {'answer_details': ans['answer_details']}
            )
            parsed_ans['total_steps'] = ans['total_steps']
            parsed_answers.append(parsed_ans)

        # calculate score and return one with highest score
        scores = []
        for ans in parsed_answers:
            score = 0
            score += int(ans['succeed_tool_calling']) * 10
            score += int(ans['used_tool_types']) * 5
            if int(ans['total_steps']) <= 0:
                score -= int(1e5)
            else:
                score += -5 * math.log(ans['total_steps'])
            scores.append(score)
        highest_score = max(scores)
        highest_idx = [idx for idx, score in enumerate(scores) if score == highest_score]
        return random.choice(highest_idx)

    def normalized_gemini_completions(self, task_description: Dict, answers: List[Dict[Any, Any]]) -> int:

        all_empty = True
        all_nonempty = True
        is_nonempty = []
        for ans in answers:
            status = ans['final_answer'] != ''
            if status:
                all_empty = False
            else:
                all_nonempty = False
            is_nonempty.append(status)

        if all_nonempty:
            all_solved = True
            all_failed = True
            is_solved = []
            for ans in answers:
                status = self.check_solve_query(task_description['query'], ans['final_answer'])
                if status:
                    all_failed = False
                else:
                    all_solved = False
                is_solved.append(status)

            if all_solved:
                steps = [int(ans['total_steps']) for ans in answers]
                shortest_steps = min(steps)
                ans_idxs = [idx for idx, step in enumerate(steps) if step == shortest_steps]
                if len(ans_idxs) > 1:
                    return ans_idxs[self.select_best_final_answer(
                        task_description['query'],
                        [answers[idx]['final_answer'] for idx in ans_idxs]
                    )]
                else:
                    return ans_idxs[0]

            elif all_failed:
                return self.compare_answer_details(answers)
            else:
                return random.choice([index for index, solve in enumerate(is_solved) if solve])

        elif all_empty:
            return self.compare_answer_details(answers)
        else:
            return random.choice([index for index, nonempty in enumerate(is_nonempty) if nonempty])
