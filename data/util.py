from typing import List, Union, Optional, Dict, Any
import torch
from collections import UserDict


class BatchEncoding(UserDict):
    def __init__(self, data: Optional[Dict[str, Any]] = None, return_tp="pt"):
        super().__init__(data)
        self.return_tp = return_tp

    def __getitem__(self, item: str) -> Union[str, torch.Tensor]:
        if isinstance(item, str):
            value = self.data[item]
            if item.startswith("text"):
                return value
            else:
                if self.return_tp == "pt":
                    return torch.Tensor(value)
                else:
                    return value
        elif isinstance(item, slice):
            return BatchEncoding({key: self.data[key][item] for key in self.data.keys()})
        else:
            raise KeyError(
                "Invalid key. Only two types of key are available: "
                "(1) string, and (2) slices for data subsetting."
            )

    def to(self, device: Union[str, "torch.device"]) -> "BatchEncoding":
        for k, v in self.data.items():
            if not k.startswith("text"):
                self.data[k] = v.to(device=device)
        return self

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()
