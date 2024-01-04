import json
import time
from typing import List, Dict, Union, Any

import requests
from fastchat.model.model_adapter import get_conversation_template
# import TimeoutException
from requests.exceptions import Timeout, ConnectionError

from ..agent import AgentClient
from ...typings import AgentNetworkException
from .http_agent import HTTPAgent


class Prompter:
    @staticmethod
    def get_prompter(prompter: Union[str, None, Dict[str, Any]]):
        name = None
        args = {}
        if isinstance(prompter, str):
            name = prompter
        elif isinstance(prompter, dict):
            name = prompter["name"]
            args = prompter["args"]
        # check if prompter_name is a method and its variable
        if not name:
            return None
        if hasattr(Prompter, name) and callable(getattr(Prompter, name)):
            return getattr(Prompter, name)(**args)

    @staticmethod
    def claude():
        def _prompter(messages: List[Dict[str, str]]):
            prompt = ""
            role_dict = {
                "user": "Human",
                "agent": "Assistant",
            }
            for item in messages:
                prompt += f"{role_dict[item['role']]}: {item['content']}\n\n"
            prompt += "Assistant:"
            return {"prompt": prompt}

        return _prompter

    @staticmethod
    def openchat_v3_1():
        def _prompter(messages: List[Dict[str, str]]):
            prompt = "Assistant is GPT4<|end_of_turn|>"
            role_dict = {
                "user": "User: {content}<|end_of_turn|>",
                "agent": "Assistant: {content}<|end_of_turn|>",
            }
            for item in messages:
                prompt += role_dict[item["role"]].format(content=item["content"])
            prompt += "Assistant:"
            return {"prompt": prompt}

        return _prompter

    @staticmethod
    def openchat_v3_2():
        def _prompter(messages: List[Dict[str, str]]):
            prompt = ""
            role_dict = {
                "user": "GPT4 User: {content}<|end_of_turn|>\n",
                "agent": "GPT4 Assistant: {content}<|end_of_turn|>\n",
            }
            for item in messages:
                prompt += role_dict[item["role"]].format(content=item["content"])
            prompt += "GPT4 Assistant:"
            return {"prompt": prompt}

        return _prompter

    @staticmethod
    def prompt_string(
        prefix: str = "",
        suffix: str = "AGENT:",
        user_format: str = "USER: {content}\n\n",
        agent_format: str = "AGENT: {content}\n\n",
        prompt_key: str = "prompt",
    ):
        def prompter(messages: List[Dict[str, str]]):
            nonlocal prefix, suffix, user_format, agent_format, prompt_key
            prompt = prefix
            for item in messages:
                if item["role"] == "user":
                    prompt += user_format.format(content=item["content"])
                else:
                    prompt += agent_format.format(content=item["content"])
            prompt += suffix
            return {prompt_key: prompt}

        return prompter


# def review_generate_os(history, controller_address, worker_address, \
#                     model_name, temperature, max_new_tokens, top_p, prompter, args):
#     s = history[-1]["content"]
#     oai_history = [{"role": "system", "content": "You have to act like a reviewer of an agent that is solving a task in a Linux OS. The agent responds via Think-Act pair. You have to find problems in the agent's most recent Think-Act pair (if any).\nIf there is no problem just print '[NO PROBLEM]' otherwise point out the problem in detail."}] + list(history[6:])
#     fs_agent = FastChatAgent(model_name, controller_address, worker_address, \
#                              temperature, max_new_tokens, top_p, prompter, args)
#     oai_agent = HTTPAgent(url="https://api.openai.com/v1/chat/completions",
#                          body={"model": "gpt-4-0613", "temperature": 0, "max_tokens": 512},
#                          headers={"Content-Type": "application/json", "Authorization": "Bearer sk-"},
#                          return_format="{response[choices][0][message][content]}",
#                          prompter={"name": "system_role_content_dict", "args": {"agent_role": "assistant"}})
    
#     for _ in range(10):
#         oai_resp = "Review: " + oai_agent.inference(oai_history)
#         oai_history.append({"role": "agent", "content": oai_resp})
#         s += f"\n{oai_resp}"
#         if "NO PROBLEM" in oai_resp:
#             break
#         history.append({"role": "user", "content": f"{oai_resp}\nRewrite the corrected 'Act' based on the review."})
#         fs_resp = 'Act:' + fs_agent.inference(history).split('Act:')[-1]
#         history.append({"role": "agent", "content": fs_resp})
#         oai_history.append({"role": "user", "content": fs_resp})
#         s += f"\n{fs_resp}"
#     print("s =", s)
#     print("-"*100)
#     return s


def review_generate_os(history, controller_address, worker_address, \
                    model_name, temperature, max_new_tokens, top_p, prompter, args):
    s = history[-1]["content"]
    _history = history 
    fs_agent = FastChatAgent(model_name, controller_address, worker_address, \
                             temperature, max_new_tokens, top_p, prompter, args)
    oai_agent = HTTPAgent(url="https://api.openai.com/v1/chat/completions",
                         body={"model": "gpt-4-0613", "temperature": 0, "max_tokens": 512},
                         headers={"Content-Type": "application/json", "Authorization": "Bearer sk-"},
                         return_format="{response[choices][0][message][content]}",
                         prompter={"name": "role_content_dict", "args": {"agent_role": "assistant"}})
    for _ in range(10):
        _history.append({"role": "user", "content": "Review the agent's most recent response/Act and find problems (if any). Correctness of an 'Act' is defined by doing the correct operation to solve the task or producing the final output/answer as per the instructions provided.\nIf there is no problem with the agent's message just respond '[The 'Act' seems correct]' ONLY. Otherwise just hint a solution along with pointing out the problem in detail. DO NOT provide the correct code in the review"})
        oai_resp = "Review: " + oai_agent.inference(_history)
        _history.append({"role": "agent", "content": oai_resp})
        s += f"\n{oai_resp}"
        if "The 'Act' seems correct" in oai_resp:
            break
        _history.append({"role": "user", "content": "Rewrite the corrected 'Act' based on the review"})
        fs_resp = "Act:" + fs_agent.inference(_history).split("Act:")[-1]
        _history.append({"role": "agent", "content": fs_resp})
        s += f"\n{fs_resp}"
    print("s =", s)
    print("-"*100)
    return s
        
        
class FastChatAgent(AgentClient):
    """This agent is a test agent, which does nothing. (return empty string for each action)"""

    def __init__(
        self,
        model_name,
        controller_address=None,
        worker_address=None,
        temperature=0,
        max_new_tokens=32,
        top_p=0,
        prompter=None,
        args=None,
        **kwargs,
    ) -> None:
        if controller_address is None and worker_address is None:
            raise ValueError(
                "Either controller_address or worker_address must be specified."
            )
        self.controller_address = controller_address
        self.worker_address = worker_address
        self.model_name = model_name
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.top_p = top_p
        if isinstance(prompter, dict):
            self.prompter = Prompter.get_prompter(prompter)
        else:
            self.prompter = prompter
        self.args = args or {}
        super().__init__(**kwargs)

    def inference(self, history: List[dict]) -> str:
        if self.worker_address:
            worker_addr = self.worker_address
        else:
            controller_addr = self.controller_address
            worker_addr = controller_addr
        if worker_addr == "":
            raise ValueError
        gen_params = {
            "model": self.model_name,
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens,
            "echo": False,
            "top_p": self.top_p,
            **self.args,
        }
        if self.prompter:
            prompt = self.prompter(history)
            gen_params.update(prompt)
        else:
            conv = get_conversation_template(self.model_name)
            for history_item in history:
                role = history_item["role"]
                content = history_item["content"]
                if role == "user":
                    conv.append_message(conv.roles[0], content)
                elif role == "agent":
                    conv.append_message(conv.roles[1], content)
                else:
                    raise ValueError(f"Unknown role: {role}")
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            gen_params.update(
                {
                    "prompt": prompt,
                    "stop": conv.stop_str,
                    "stop_token_ids": conv.stop_token_ids,
                }
            )
        headers = {"User-Agent": "FastChat Client"}
        for _ in range(3):
            try:
                response = requests.post(
                    controller_addr + "/worker_generate_stream",
                    headers=headers,
                    json=gen_params,
                    stream=True,
                    timeout=120,
                )
                text = ""
                for line in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
                    if line:
                        data = json.loads(line)
                        if data["error_code"] != 0:
                            raise AgentNetworkException(data["text"])
                        text = data["text"]
                if "Think:" in text and "Act:" in text:
                    history.append({"role": "agent", "content": text})
                    text = review_generate_os(history, self.controller_address, self.worker_address, \
                                    self.model_name, self.temperature, self.max_new_tokens, \
                                    self.top_p, self.prompter, self.args)
                return text
            # if timeout or connection error, retry
            except Timeout:
                print("Timeout, retrying...")
            except ConnectionError:
                print("Connection error, retrying...")
            time.sleep(5)
        else:
            raise Exception("Timeout after 3 retries.")
