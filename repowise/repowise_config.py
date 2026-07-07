from copy import deepcopy
import json
class RepoWiseConfig:
    def __init__(self):
        self.index_retries = 3
        try:    
            with open("repowise/configs.json","r") as configs:
                self._configs = json.load(configs)
        except Exception as e:
            print(f"Error in getting config file : {e}")
            self._configs = {}
        self._current = "ollama_qwen_1_5b"

    def set_current(self, name: str):
        if name not in self._configs:
            raise ValueError(f"Unknown configuration: {name}")
        self._current = name

    def get_current(self) -> dict:
        return deepcopy(self._configs[self._current])

    def get_retries(self):
        return self.index_retries or 3

    def get(self, name: str) -> dict:
        if name not in self._configs:
            raise ValueError(f"Unknown configuration: {name}")
        return deepcopy(self._configs[name])

