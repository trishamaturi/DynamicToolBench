## How to Run

1. Clone the repository:
```bash
git clone <repository-url>
```

2. Download toolset from [Huggingface](https://huggingface.co/datasets/stabletoolbench/ToolEnv2404/tree/main) 
3. Move it under a folder called tools/

4. Create a `config.yml` file (copy from `example_config.yml`):
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

7. Start the virtual api server
```
python server/main.py
```

8. Open a new terminal window
9. Run the pipeline on the solvable queries
```
python toolbench/inference/qa_pipeline_multithread.py --tool_root_dir=tools/toolenv2404_filtered --backbone_model=gemini_function --max_observation_length=1024 --method=CoT@1 --input_query_file=solvable_queries/test_instruction/G1_category.json --output_answer_file=data/answer/virtual_gemini_cot --num_thread=1
```
