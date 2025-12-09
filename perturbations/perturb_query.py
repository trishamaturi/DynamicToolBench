import json
import argparse
from pathlib import Path
import random
import yaml
import google.generativeai as genai
from google.api_core import exceptions as gex
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

config_file = 'config.yml'
CONFIG = yaml.load(open(config_file, 'r'), Loader=yaml.FullLoader)

genai.configure(api_key=CONFIG['api_key'])
model_name = CONFIG.get('model', 'gemini-2.5-flash')
temperature = CONFIG.get('temperature', 0.1)

def random_seed(seed: int):
    random.seed(seed)

def maybe_add_trailing_period(text: str) -> str:
    text = text.strip()
    if not text:
        return text
    if text[-1] not in ".!?":
        return text + "."
    return text

def perturb_distractor_story(query: str) -> str:
    story = (
        "Ignore the following irrelevant story and focus ONLY on the actual task.\n\n"
        "Story: Yesterday my friend and I spent hours talking about movies, cats, and random trivia. "
        "None of this is related to what I actually need you to do.\n\n"
        "Task: "
    )
    return story + query

def perturb_case(query: str) -> str:
    mode = random.choice(["upper", "lower", "random"])
    if mode == "upper":
        return query.upper()
    if mode == "lower":
        return query.lower()
    # random case
    return "".join(
        ch.upper() if random.random() < 0.5 else ch.lower()
        for ch in query
    )

def perturb_punctuation_noise(query: str) -> str:
    # Add or remove some punctuation and newlines
    q = query.strip()

    # Maybe add excessive punctuation at the end
    if random.random() < 0.5:
        q = maybe_add_trailing_period(q) + "??"

    # Randomly insert line breaks before some commas
    q = q.replace(", ", ",\n") if random.random() < 0.5 else q

    # Maybe remove some periods
    if random.random() < 0.3:
        q = q.replace(".", "")

    return q

def call_gemini(system_prompt, prompt):
    system_prompt = {"role": "user", "parts": system_prompt}
    user_prompt = {"role": "user", "parts": prompt}
    model = genai.GenerativeModel(model_name)
    messages = [system_prompt, user_prompt]

    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = model.generate_content(
                messages,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature
                )
            )
            result = response.text
            return result
        except gex.GoogleAPIError as e:
            time.sleep(10)
            continue

def perturb_query(q: str, mode: str) -> str:
    if mode == "distractor":
        return perturb_distractor_story(q)
    elif mode == "case_mix":
        return perturb_case(q)
    elif mode == "punctuation_noise":
        return perturb_punctuation_noise(q)
    elif mode == "model_paraphrase":
        system_prompt = "You are a skilled language agent versed in paraphrasing text."
        prompt = (
            "Paraphrase the following user query while keeping the meaning identical. "
            "Do NOT change named entities or specific values.\n\n"
            f"Query: {q}\n\nParaphrase:"
        )
        response = call_gemini(system_prompt, prompt)
        return response
    else:
        return q  # no-op

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", default="model_paraphrase")
    parser.add_argument("--num_threads", default=10)
    args = parser.parse_args()

    data = json.load(open(args.input))

    futures = {}
    with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        for idx, item in enumerate(data):
            future = executor.submit(perturb_query, item["query"], args.mode)
            futures[future] = idx

        for future in tqdm(as_completed(futures), total=len(futures), desc="Perturbation Pipeline"):
            idx = futures[future]
            res = future.result()
            data[idx]["query"] = res

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(data, open(args.output, "w"), ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()