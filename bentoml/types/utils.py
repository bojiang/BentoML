import collections


class LazyDict(collections.UserDict):
    def __getitem__(self, key: str) -> type:
        return self.data[key]()
