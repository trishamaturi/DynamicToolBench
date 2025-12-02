import os
import time
import json
import traceback
import yaml

import google.generativeai as genai
from tenacity import retry, wait_random_exponential, stop_after_attempt
from termcolor import colored

def clean_schema(obj):
    if isinstance(obj, dict):
        # First clean children (so recursive fixes happen before parent logic)
        cleaned = {
            k: clean_schema(v)
            for k, v in obj.items()
            if k not in [
                "optional",
                "nullable",
                "oneOf",
                "anyOf",
                "x-apitype",
                "example_value",
            ]
        }

        # Clean up "required" fields
        if cleaned.get("type") == "object":
            props = cleaned.get("properties", {}) or {}
            if "required" in cleaned:
                req = cleaned["required"]
                if isinstance(req, list):
                    new_req = [
                        r for r in req
                        if isinstance(r, str) and r.strip() and r in props
                    ]
                    if new_req:
                        cleaned["required"] = new_req
                    else:
                        cleaned.pop("required", None)
                else:
                    cleaned.pop("required", None)

        return cleaned
    elif isinstance(obj, list):
        return [clean_schema(v) for v in obj]
    else:
        return obj

@retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
def gemini_chat_request(
    api_key,
    messages,
    tools=None,
    tool_choice=None,   # currently unused; can be wired into tool_config
    model="gemini-1.5-pro",
    stop=None,
    process_id=0,
    **args,
):
    """
    Rough analogue of chat_completion_request, but using Gemini.
    - `messages` are OpenAI-style: [{"role": "user"/"assistant"/"system", "content": "..."}]
    - `tools` are OpenAI-style tool schemas (same as you pass to ChatGPTFunction).
    """

    use_messages = []
    for message in messages:
        if not ("valid" in message.keys() and message["valid"] is False):
            use_messages.append(message)

    for message in use_messages:
        if "function_call" in message:
            message.pop("function_call")

    # Configure client
    genai.configure(api_key=api_key)
    gemini_model = genai.GenerativeModel(model)

    contents = []
    for m in messages:
        role = m.get("role", "user")
        text = m.get("content", "")

        if not isinstance(text, str):
            text = json.dumps(text, ensure_ascii=False)

        if role == "assistant":
            tool_calls = m.get("tool_calls", "")
            if tool_calls:
                tool_name = tool_calls[0]["function"].get("name", "")
                if tool_name:
                    args_dict = json.loads(tool_calls[0]["function"].get("arguments", "{}"))
                    text = {"function_call": {"name": tool_name, "args": args_dict}}

            contents.append({"role": "model", "parts": [text]})
        else:
            # treat system/user/tool all as user for simplicity
            if role == "tool":
                text = f"Results of calling tool: {text}"
            contents.append({"role": "user", "parts": [text]})

    # Convert OpenAI-style tools → Gemini tools (function declarations)
    # OpenAI tools look like:
    # {"type":"function","function":{"name":...,"description":...,"parameters":{...}}}
    function_declarations = []

    if tools:
        # Tools are OpenAI formatted so need to be cleaned for Gemini
        tools = clean_schema(tools)
        for t in tools:
            if t.get("type") != "function":
                continue
            fn = t["function"]
            fn_decl = {
                "name": fn["name"],
                "description": fn.get("description", ""),
                # Gemini function declarations accept JSON Schema-like "parameters"
                "parameters": fn.get("parameters", {}),
            }
            function_declarations.append(fn_decl)

    gemini_tools = None
    if function_declarations:
        # In current gemini client, tools is a dict with "function_declarations"
        gemini_tools = [{"function_declarations": function_declarations}]

    # Tool config: you could emulate tool_choice, but for now we do "auto"
    tool_config = None
    if gemini_tools:
        tool_config = {
            "function_calling_config": {
                "mode": "AUTO",
            }
        }

    # Stop sequences are not 1-1 with OpenAI; we ignore for now or map to safety settings.
    # You can add stop sequences under generation_config if needed:
    generation_config = {
        "max_output_tokens": args.get("max_tokens", 1024),
    }
    if stop:
        # Gemini uses "stop_sequences"
        generation_config["stop_sequences"] = stop if isinstance(stop, list) else [stop]

    try:
        response = gemini_model.generate_content(
            contents=contents,
            tools=gemini_tools,
            tool_config=tool_config,
            generation_config=generation_config,
        )

        candidate = response.candidates[0]
        parts = candidate.content.parts

        final_text = ""
        tool_calls = []

        for p in parts:
            # function call part
            fc = getattr(p, "function_call", None)
            if fc is not None:
                # fc.name, fc.args (dict-like)
                tool_calls.append(
                    {
                        "id": None,  # Gemini doesn't give ID; STB usually doesn't require it
                        "type": "function",
                        "function": {
                            "name": fc.name,
                            "arguments": json.dumps(dict(fc.args or {})),
                        },
                    }
                )
            # normal text part
            if getattr(p, "text", None):
                final_text += p.text

        message = {
            "role": "assistant",
            "content": final_text,
        }
        if tool_calls:
            message["tool_calls"] = tool_calls

        # Usage (Gemini has usage field)
        total_tokens = 0
        if getattr(response, "usage_metadata", None):
            meta = response.usage_metadata
            total_tokens = (
                (meta.prompt_token_count or 0)
                + (meta.candidates_token_count or 0)
            )

        result_dict = {
            "choices": [{"message": message}],
            "usage": {"total_tokens": total_tokens},
        }
        return result_dict

    except Exception as e:
        print("Unable to generate Gemini response")
        traceback.print_exc()
        return {"error": str(e), "usage": {"total_tokens": 0}}


# ---------- High-level wrapper (GeminiFunction) ----------

class GeminiFunction:
    def __init__(self, model="gemini-1.5-pro", gemini_key=""):
        self.model = model
        self.conversation_history = []
        self.gemini_key = gemini_key
        self.time = time.time()
        self.TRY_TIME = 6

    def add_message(self, message):
        self.conversation_history.append(message)

    def change_messages(self, messages):
        self.conversation_history = messages

    def display_conversation(self, detailed=False):
        role_to_color = {
            "system": "red",
            "user": "green",
            "assistant": "blue",
            "function": "magenta",
            "tool": "magenta",
        }
        print("before_print" + "*" * 50)
        for message in self.conversation_history:
            role = message.get("role", "user")
            print_obj = f"{role}: {message.get('content', '')} "
            if "function_call" in message:
                print_obj += f"function_call: {message['function_call']}"
            if "tool_calls" in message:
                print_obj += f"tool_calls: {message['tool_calls']}"
                print_obj += f"number of tool calls: {len(message['tool_calls'])}"
            if detailed:
                print_obj += f"function_call: {message.get('function_call')}"
                print_obj += f"tool_calls: {message.get('tool_calls')}"
                print_obj += f"function_call_id: {message.get('function_call_id')}"
            print(colored(print_obj, role_to_color.get(role, "white")))
        print("end_print" + "*" * 50)

    def parse(self, tools, process_id, key_pos=None, **args):
        """
        Same shape as ChatGPTFunction.parse:
          - uses self.conversation_history
          - calls gemini_chat_request(...)
          - returns (message, error_code, total_tokens)
        """
        self.time = time.time()
        conversation_history = self.conversation_history

        for attempt in range(self.TRY_TIME):
            if attempt != 0:
                time.sleep(15)

            if tools is not None and tools != []:
                response = gemini_chat_request(
                    self.gemini_key,
                    conversation_history,
                    tools=tools,
                    process_id=process_id,
                    model=self.model,
                    **args,
                )
            else:
                response = gemini_chat_request(
                    self.gemini_key,
                    conversation_history,
                    process_id=process_id,
                    model=self.model,
                    **args,
                )

            try:
                total_tokens = response.get("usage", {}).get("total_tokens", 0)
                message = response["choices"][0]["message"]

                if process_id == 0:
                    print(f"[process({process_id})]total tokens: {total_tokens}")

                return message, 0, total_tokens
            except BaseException as e:
                print(f"[process({process_id})]Parsing Exception: {repr(e)}. Try again.")
                traceback.print_exc()
                if response is not None:
                    print(f"[process({process_id})]Gemini return: {response}")

        # after TRY_TIME failures
        return {"role": "assistant", "content": str(response)}, -1, 0


# ---------- Example usage (mirroring your main) ----------

def get_current_weather(location, unit="fahrenheit"):
    """Get the current weather in a given location"""
    if "tokyo" in location.lower():
        return json.dumps({"location": "Tokyo", "temperature": "10", "unit": unit})
    elif "san francisco" in location.lower():
        return json.dumps({"location": "San Francisco", "temperature": "72", "unit": unit})
    elif "paris" in location.lower():
        return json.dumps({"location": "Paris", "temperature": "22", "unit": unit})
    else:
        return json.dumps({"location": location, "temperature": "unknown"})


# if __name__ == "__main__":
#     api_key = ""
#     gemini_model = "gemini-2.5-flash"
#     llm = GeminiFunction(gemini_key=api_key, model=gemini_model)
#
#     messages = [{"role": "user", "content": "What's the weather like in San Francisco, Tokyo, and Paris?"}]
#
#     tools = [
#         {
#             "type": "function",
#             "function": {
#                 "name": "get_current_weather",
#                 "description": "Get the current weather in a given location",
#                 "parameters": {
#                     "type": "object",
#                     "properties": {
#                         "location": {
#                             "type": "string",
#                             "description": "The city and state, e.g. San Francisco, CA",
#                         },
#                         "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
#                     },
#                     "required": ["location"],
#                 },
#             },
#         }
#     ]
#
#     llm.change_messages(messages)
#     output, error_code, token_usage = llm.parse(tools=tools, process_id=0)
#     print(output)