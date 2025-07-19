import random
from typing import List, Dict, Set, Optional, Any

class BaseAgent:
    """Base class for all agents in the simulation."""
    def __init__(self, unique_id: int, model: Optional[Any]):
        self.id: int = unique_id
        self.model = model
        self.type: str = "Base"
        self.wealth: float = max(0, random.gauss(100, 30))
        self.happiness: float = 1.0
        self.dissent: float = 0.0
        self.ideology: float = random.uniform(-1, 1)  # -1: left, 1: right
        self.trust_media: float = random.uniform(0, 1)
        self.social_links: Set[int] = set()
        self.memory: List[Dict] = []  # List of events/experiences
        self.influence: float = random.uniform(0, 1)
        self.role: str = "citizen"
        self.log: List[str] = []

    def step(self):
        pass

    def remember(self, event: Dict):
        self.memory.append(event)
        if len(self.memory) > 100:
            self.memory.pop(0)

    def log_event(self, msg: str):
        self.log.append(msg)
        if len(self.log) > 200:
            self.log.pop(0)

class Citizen(BaseAgent):
    def __init__(self, unique_id: int, model: Optional[Any]):
        super().__init__(unique_id, model)
        self.type = "Citizen"
        self.role = "citizen"
        self.votes: List[float] = []
        self.mobility: float = random.uniform(0, 1)  # Social mobility

    def step(self):
        shock = random.gauss(0, 5)
        self.wealth += shock
        if self.wealth < 80:
            self.dissent += 0.1
        else:
            self.dissent = max(0, self.dissent - 0.05)
        self.ideology += random.gauss(0, 0.01)
        self.ideology = max(-1, min(1, self.ideology))
        if self.model and hasattr(self.model, 'media') and self.model.media:
            media_bias = self.model.media.bias
            self.ideology += (media_bias - self.ideology) * self.trust_media * 0.05
        if self.model and hasattr(self.model, 'agents'):
            for friend_id in self.social_links:
                friend = self.model.agents[friend_id]
                self.ideology += (friend.ideology - self.ideology) * 0.01
        self.happiness = max(0, 1.0 - self.dissent)
        if self.model and hasattr(self.model, 'year') and hasattr(self.model, 'constitution') and self.model.year % self.model.constitution.get('election_interval', 4) == 0:
            self.votes.append(self.ideology)
        # Learning: adapt trust in media based on past events
        if self.memory and random.random() < 0.1:
            last_event = self.memory[-1]
            if 'fake_news' in last_event:
                self.trust_media *= 0.95
        # Social mobility: chance to become politician
        if self.mobility > 0.95 and random.random() < 0.01 and self.model and hasattr(self.model, 'promote_to_politician'):
            self.model.promote_to_politician(self)
        self.log_event(f"Year {self.model.year if self.model and hasattr(self.model, 'year') else '?'}: wealth={self.wealth:.2f}, dissent={self.dissent:.2f}, ideology={self.ideology:.2f}")

class Politician(BaseAgent):
    def __init__(self, unique_id: int, model: Optional[Any]):
        super().__init__(unique_id, model)
        self.type = "Politician"
        self.role = "politician"
        self.popularity: float = random.uniform(0, 1)
        self.corruption: float = random.uniform(0, 0.2)

    def step(self):
        if self.model and hasattr(self.model, 'citizens') and self.model.citizens:
            avg_ideology = sum(a.ideology for a in self.model.citizens) / len(self.model.citizens)
        else:
            avg_ideology = 0.0
        self.ideology += (avg_ideology - self.ideology) * 0.05
        if random.random() < 0.01:
            self.corruption += random.uniform(0, 0.1)
        self.corruption = min(self.corruption, 1.0)
        self.popularity = 1.0 - self.corruption - abs(self.ideology - avg_ideology)
        # Learning: reduce corruption if caught
        if self.memory and any('scandal' in e for e in self.memory):
            self.corruption *= 0.9
        self.log_event(f"Year {self.model.year if self.model and hasattr(self.model, 'year') else '?'}: corruption={self.corruption:.2f}, popularity={self.popularity:.2f}")

class Judge(BaseAgent):
    def __init__(self, unique_id: int, model: Optional[Any]):
        super().__init__(unique_id, model)
        self.type = "Judge"
        self.role = "judge"
        self.independence: float = random.uniform(0.7, 1.0)

    def step(self):
        if random.random() < 0.02:
            if self.independence > 0.85 and self.model:
                self.model.laws_overturned += 1
        self.log_event(f"Year {self.model.year if self.model else '?'}: independence={self.independence:.2f}")

class Journalist(BaseAgent):
    def __init__(self, unique_id: int, model: Optional[Any]):
        super().__init__(unique_id, model)
        self.type = "Journalist"
        self.role = "journalist"
        self.bias: float = random.uniform(-0.5, 0.5)

    def step(self):
        if self.model and hasattr(self.model, 'media') and self.model.media:
            self.model.media.bias += self.bias * 0.01
            self.model.media.bias = max(-1, min(1, self.model.media.bias))
        # Learning: adjust bias if public unrest is high
        if self.model and hasattr(self.model, 'history') and self.model.history and self.model.history['avg_dissent'] and self.model.history['avg_dissent'][-1] > 0.7:
            self.bias *= 0.95
        self.log_event(f"Year {self.model.year if self.model else '?'}: bias={self.bias:.2f}")

class Media:
    def __init__(self, model: Optional[Any]):
        self.model = model
        self.bias: float = random.uniform(-0.2, 0.2)
        self.fake_news_rate: float = random.uniform(0, 0.1)
        self.log: List[str] = []

    def broadcast(self, message: str):
        self.log.append(message)
        if len(self.log) > 200:
            self.log.pop(0)

def create_custom_agent_class(name: str, base: type, attrs: dict):
    """Dynamically create a new agent class with custom attributes and methods."""
    return type(name, (base,), attrs) 