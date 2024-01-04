import contextlib
import time
import warnings

import requests
from urllib3.exceptions import InsecureRequestWarning

from src.typings import *
from src.utils import *
from ..agent import AgentClient

old_merge_environment_settings = requests.Session.merge_environment_settings


@contextlib.contextmanager
def no_ssl_verification():
    opened_adapters = set()

    def merge_environment_settings(self, url, proxies, stream, verify, cert):
        # Verification happens only once per connection so we need to close
        # all the opened adapters once we're done. Otherwise, the effects of
        # verify=False persist beyond the end of this context manager.
        opened_adapters.add(self.get_adapter(url))

        settings = old_merge_environment_settings(self, url, proxies, stream, verify, cert)
        settings['verify'] = False

        return settings

    requests.Session.merge_environment_settings = merge_environment_settings

    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', InsecureRequestWarning)
            yield
    finally:
        requests.Session.merge_environment_settings = old_merge_environment_settings

        for adapter in opened_adapters:
            try:
                adapter.close()
            except:
                pass


class Prompter:
    @staticmethod
    def get_prompter(prompter: Union[Dict[str, Any], None]):
        # check if prompter_name is a method and its variable
        if not prompter:
            return Prompter.default()
        assert isinstance(prompter, dict)
        prompter_name = prompter.get("name", None)
        prompter_args = prompter.get("args", {})
        if hasattr(Prompter, prompter_name) and callable(
            getattr(Prompter, prompter_name)
        ):
            return getattr(Prompter, prompter_name)(**prompter_args)
        return Prompter.default()

    @staticmethod
    def default():
        return Prompter.role_content_dict()

    @staticmethod
    def batched_role_content_dict(*args, **kwargs):
        base = Prompter.role_content_dict(*args, **kwargs)

        def batched(messages):
            result = base(messages)
            return {key: [result[key]] for key in result}

        return batched

    @staticmethod
    def system_role_content_dict(
        message_key: str = "messages",
        role_key: str = "role",
        content_key: str = "content",
        user_role: str = "user",
        agent_role: str = "agent",
    ):
        def prompter(messages: List[Dict[str, str]]):
            nonlocal message_key, role_key, content_key, user_role, agent_role
            role_dict = {
                "system": "system",
                "user": user_role,
                "agent": agent_role,
            }
            prompt = []
            for item in messages:
                prompt.append(
                    {role_key: role_dict[item["role"]], content_key: item["content"]}
                )
            return {message_key: prompt}

        return prompter
    
    @staticmethod
    def role_content_dict(
        message_key: str = "messages",
        role_key: str = "role",
        content_key: str = "content",
        user_role: str = "user",
        agent_role: str = "agent",
    ):
        def prompter(messages: List[Dict[str, str]]):
            nonlocal message_key, role_key, content_key, user_role, agent_role
            role_dict = {
                "user": user_role,
                "agent": agent_role,
            }
            prompt = []
            for item in messages:
                prompt.append(
                    {role_key: role_dict[item["role"]], content_key: item["content"]}
                )
            return {message_key: prompt}

        return prompter

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
            print(prompt)
            return {prompt_key: prompt}

        return prompter

    @staticmethod
    def claude():
        return Prompter.prompt_string(
            prefix="",
            suffix="Assistant:",
            user_format="Human: {content}\n\n",
            agent_format="Assistant: {content}\n\n",
        )

    @staticmethod
    def palm():
        def prompter(messages):
            return {"instances": [
                Prompter.role_content_dict("messages", "author", "content", "user", "bot")(messages)
            ]}
        return prompter


def check_context_limit(content: str):
    content = content.lower()
    and_words = [
        ["prompt", "context", "tokens"],
        [
            "limit",
            "exceed",
            "max",
            "long",
            "much",
            "many",
            "reach",
            "over",
            "up",
            "beyond",
        ],
    ]
    rule = AndRule(
        [
            OrRule([ContainRule(word) for word in and_words[i]])
            for i in range(len(and_words))
        ]
    )
    return rule.check(content)


def review_generate_os(history, url, proxies, headers,\
                        body, return_format, prompter):
    s = history[-1]["content"]
    _history = history 
    fs_agent = HTTPAgent(url, proxies, body, \
                        headers, return_format, prompter)
    oai_agent = HTTPAgent(url="https://api.openai.com/v1/chat/completions",
                         body={"model": "gpt-4-0613", "temperature": 0, "max_tokens": 512},
                         headers={"Content-Type": "application/json", "Authorization": "Bearer sk-"},
                         return_format="{response[choices][0][message][content]}",
                         prompter={"name": "role_content_dict", "args": {"agent_role": "assistant"}})
    for _ in range(10):
        _history.append({"role": "user", "content": "Review the agent's most recent response/Act and find problems (if any). Correctness of an 'Act' is defined by doing the correct operation to solve the task or producing the final output/answer as per the instructions provided.\nIf there is no problem with the agent's message just respond '[The 'Act' seems correct]' ONLY. Otherwise just hint a solution along with pointing out the problem in detail. DO NOT provide the code in the review."})
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


class HTTPAgent(AgentClient):
    def __init__(
        self,
        url,
        proxies=None,
        body=None,
        headers=None,
        return_format="{response}",
        prompter=None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.url = url
        self.proxies = proxies or {}
        self.headers = headers or {}
        self.body = body or {}
        self.return_format = return_format
        if isinstance(prompter, dict):
            self.prompter = Prompter.get_prompter(prompter)
        else:
            self.prompter = prompter
        if not self.url:
            raise Exception("Please set 'url' parameter")

    def _handle_history(self, history: List[dict]) -> Dict[str, Any]:
        return self.prompter(history)

    def inference(self, history: List[dict]) -> str:
        for _ in range(3):
            try:
                body = self.body.copy()
                body.update(self._handle_history(history))
                with no_ssl_verification():
                    resp = requests.post(
                        self.url, json=body, headers=self.headers, proxies=self.proxies, timeout=120
                    )
                # print("resp.status_code, resp.text =", resp.status_code, resp.text)
                if resp.status_code != 200:
                    # print(resp.text)
                    if check_context_limit(resp.text):
                        raise AgentContextLimitException(resp.text)
                    else:
                        raise Exception(
                            f"Invalid status code {resp.status_code}:\n\n{resp.text}"
                        )
            except AgentClientException as e:
                raise e
            except Exception as e:
                print("Warning: ", e)
                pass
            else:
                resp = resp.json()
                text = resp['choices'][0]['message']['content']
                if "Think:" in text and "Act:" in text:
                    history.append({"role": "agent", "content": text})
                    resp['choices'][0]['message']['content'] = review_generate_os(history, self.url, self.proxies, self.body,\
                                                                                  self.headers, self.return_format, self.prompter)
                return self.return_format.format(response=resp)
            time.sleep(_ + 2)
        raise Exception("Failed.")
