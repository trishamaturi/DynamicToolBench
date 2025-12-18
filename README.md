# Scaling LLMs Systems Project: [DynamicToolBench]
## Team Information
- **Members**:
- Trisha Maturi (tm3530)
- Pranav Pusarla (pp2903)
---
## 1. Problem Statement

Existing tool-use benchmarks assume static, perfectly aligned API schemas, while real-world APIs evolve through versioning and refactoring. This mismatch causes tool-augmented LLMs to generate requests based on outdated interface assumptions, leading to failures that current evaluations do not capture. We address the problem of evaluating LLM robustness under realistic API schema drift, independently of natural language variation, in a reproducible setting.

---
## 2. Summary of Code:

`perturbations/`: contains scripts for generating controlled query-level and api-versioning perturbations used to evaluate robustness.

`server/`: implements the server-side infrastructure for executing tool calls, including cached response lookup and LLM-based simulation of API responses when cache misses occur.

`solvable_queries/`: stores filtered query subsets that are solvable under baseline conditions and used for consistent evaluation across perturbation settings.

`toolbench/inference/`: contains the inference and execution logic for running tool-augmented LLMs, including tool selection, request generation, and iterative reasoning.

`toolbench/qa_pipeline_multithread.py`: implements the multithreaded pipeline for running large-scale tool-use experiments efficiently across multiple queries.

`tooleval/`: provides evaluation utilities for converting raw model outputs into standardized formats and computing pass-rate metrics.

`tool_response_cache/`: stores precomputed and cached tool responses downloaded from HuggingFace to ensure reproducible and deterministic API execution.

---

## 3. Final Results Summary
API Versioning Solvable Pass Rate:

| Modifications  | G1_category | G1_instruction | G1_tool | G2_category | G2_instruction | G3_instruction |
|---------------|-------------|----------------|---------|-------------|----------------|----------------|
| Baseline       | 24.8 ± 1.9  | 21.1 ± 0.4     | 17.1 ± 0.9 | 20.3 ± 2.4  | 13.3 ± 0.8     | 14.2 ± 0.8     |
| API Versioning | 23.0 ± 1.8  | 20.9 ± 1.3     | 15.7 ± 2.0 | 13.9 ± 0.9  | 11.4 ± 1.3     | 3.3 ± 0.0      |

Query Robustness Solvable Pass Rate:

| Modifications     | G1_category | G1_instruction | G1_tool | G2_category | G2_instruction | G3_instruction |
|-------------------|-------------|----------------|---------|-------------|----------------|----------------|
| Baseline          | 24.8 ± 1.9  | 21.1 ± 0.4     | 17.1 ± 0.9 | 20.3 ± 2.4  | 13.3 ± 0.8     | 14.2 ± 0.8     |
| Model Paraphrase  | 25.4 ± 0.6  | 20.6 ± 0.9     | 17.5 ± 1.8 | 19.4 ± 0.7  | 14.0 ± 2.7     | 12.6 ± 2.8     |
| Punctuation Noise | 25.9 ± 0.8  | 19.2 ± 2.4     | 16.9 ± 0.3 | 15.3 ± 0.7  | 11.1 ± 1.6     | 13.1 ± 1.3     |
| Case Mix          | 25.2 ± 1.1  | 17.7 ± 1.2     | 19.6 ± 1.0 | 15.9 ± 0.4  | 12.4 ± 0.8     | 3.8 ± 0.8      |
| Distractor Story  | 25.1 ± 1.6  | 19.2 ± 1.6     | 19.8 ± 2.0 | 21.5 ± 2.0  | 13.7 ± 1.2     | 9.3 ± 0.8      |

---
## 4. Reproducibility Instructions
### A. Pre Requisites
1. Download toolset from [Huggingface](https://huggingface.co/datasets/stabletoolbench/Cache/blob/main/server_cache.zip)
2. Move the folders (tools/ and tool_response_cache/)  to the repository

3. Create a `config.yml` file (copy from `example_config.yml`):
```bash
cp example_config.yml config.yml
```

4. Edit `config.yml` and set your `GEMINI_API_KEY`:
```env
GEMINI_API_KEY=your_api_key_here
```

5. Create a `toolbench/tooleval/evaluators/{model_evaluator}/config.yml` file (copy from `example_config.yml`):
```bash
cd toolbench/tooleval/evaluators/{model_evaluator}
cp example_config.yml config.yml
```

6. Edit `toolbench/tooleval/evaluators/{model_evaluator}/config.yaml` and set your `api-key`:
```env
api_key=your_api_key_here
```

### B. Inference
7. Create API Versioning
```bash
python pertubations/create_new_api_version.py
```
(or use api_versioning_v2.py and api_versioning_v3.py for more granular changes)

8. Create Query Pertubations
```bash
python pertubations/perturb_query.py --input [input_file] --output [output_file]
```

8. Start the virtual api server
```bash
python server/main.py
```

9. Open a new terminal window and run the pipeline on the solvable queries (G1_instruction, G1_category, G1_tool, G2_instruction, G2_category, G3_category)
```bash
python -m toolbench.inference.qa_pipeline_multithread \
  --tool_root_dir=tools/toolenv2404_filtered \
  --backbone_model=gemini_function \
  --max_observation_length=1024 \
  --method=CoT@1 \
  --input_query_file=solvable_queries/test_instruction/G1_category.json \
  --output_answer_file=data/answer/virtual_gemini_cot/G1_category \
  --num_thread=10
```

---
### C. Evaluation
To evaluate the trained model:
10. Convert answer results
```bash
python toolbench/tooleval/convert_to_answer_format.py \
  --answer_dir="data/answer/virtual_gemini_cot/G1_category" \
  --method=CoT@1 
  --output="data/model_predictions_converted/virtual_gemini_cot/G1_category.json"
```
11. Evaluate answers
``` bash
python toolbench/tooleval/eval_pass_rate.py \
 --converted_answer_path="data/model_predictions_converted" \
 --save_path="data/pass_rate_results/virtual_gemini_cot" \
 --reference_model=virtual_gemini_cot \
 --test_ids="solvable_queries/test_query_ids" \
 --max_eval_threads=35 --evaluate_times=3 --test_set=G1_category
```
---