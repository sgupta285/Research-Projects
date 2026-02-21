from __future__ import annotations
from dataclasses import dataclass, field

@dataclass
class CacheStats:
    hits:int=0
    misses:int=0
    def record(self, hit: bool):
        self.hits += int(hit)
        self.misses += int(not hit)
    @property
    def hit_rate(self)->float:
        t=self.hits+self.misses
        return 0.0 if t==0 else self.hits/t

@dataclass
class SimpleCache:
    store:dict=field(default_factory=dict)
    stats:CacheStats=field(default_factory=CacheStats)
    def get(self,key):
        if key in self.store:
            self.stats.record(True); return True, self.store[key]
        self.stats.record(False); return False, None
    def set(self,key,val): self.store[key]=val
