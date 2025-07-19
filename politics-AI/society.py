import random
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import List, Dict, Optional
from agents import BaseAgent, Citizen, Politician, Judge, Journalist, Media

class SocietyModel:
    """Main simulation model for the virtual society."""
    def __init__(self, N_citizens=100, N_politicians=5, N_judges=3, N_journalists=3, constitution_text=None, constitution_params=None, years=20, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.N_citizens = N_citizens
        self.N_politicians = N_politicians
        self.N_judges = N_judges
        self.N_journalists = N_journalists
        self.constitution_text = constitution_text or ""
        self.constitution = self.parse_constitution(constitution_text, constitution_params)
        self.years = years
        self.year = 0
        self.laws_overturned = 0
        self.media = Media(self)
        self.agents: Dict[int, BaseAgent] = {}
        self.citizens: List[Citizen] = []
        self.politicians: List[Politician] = []
        self.judges: List[Judge] = []
        self.journalists: List[Journalist] = []
        self.history = defaultdict(list)
        self.event_log: List[str] = []
        self._init_agents()
        self._init_social_network()

    def parse_constitution(self, text, params):
        d = {
            'election_interval': 4,
            'ubi': 'universal basic income' in text.lower(),
            'recall': 'recall' in text.lower(),
            'proportional': 'proportional' in text.lower(),
            'no_supreme_court': 'no supreme court' in text.lower(),
        }
        if params:
            d.update(params)
        return d

    def _init_agents(self):
        uid = 0
        for _ in range(self.N_citizens):
            c = Citizen(uid, self)
            self.agents[uid] = c
            self.citizens.append(c)
            uid += 1
        for _ in range(self.N_politicians):
            p = Politician(uid, self)
            self.agents[uid] = p
            self.politicians.append(p)
            uid += 1
        for _ in range(self.N_judges):
            j = Judge(uid, self)
            self.agents[uid] = j
            self.judges.append(j)
            uid += 1
        for _ in range(self.N_journalists):
            j = Journalist(uid, self)
            self.agents[uid] = j
            self.journalists.append(j)
            uid += 1

    def _init_social_network(self):
        for c in self.citizens:
            friends = random.sample(self.citizens, k=min(5, len(self.citizens)-1))
            c.social_links = set(f.id for f in friends if f.id != c.id)

    def promote_to_politician(self, citizen: Citizen):
        self.citizens.remove(citizen)
        new_pol = Politician(citizen.id, self)
        self.agents[citizen.id] = new_pol
        self.politicians.append(new_pol)
        self.event_log.append(f"Year {self.year}: Citizen {citizen.id} promoted to Politician.")

    def step(self):
        self.year += 1
        self._event_system()
        for agent in self.agents.values():
            agent.step()
        self._collect_metrics()

    def run(self):
        for _ in range(self.years):
            self.step()

    def _event_system(self):
        # Economic shock
        if random.random() < 0.2:
            for c in self.citizens:
                c.wealth += random.gauss(-20, 10)
                c.remember({'type': 'economic_shock', 'amount': c.wealth})
            self.event_log.append(f"Year {self.year}: Economic shock occurred.")
        # Political scandal
        if random.random() < 0.1:
            for p in self.politicians:
                p.corruption += random.uniform(0, 0.2)
                p.corruption = min(p.corruption, 1.0)
                p.remember({'type': 'scandal'})
            self.event_log.append(f"Year {self.year}: Political scandal broke out.")
        # Fake news
        if random.random() < self.media.fake_news_rate:
            self.media.bias += random.uniform(-0.2, 0.2)
            self.media.bias = max(-1, min(1, self.media.bias))
            for c in self.citizens:
                c.remember({'type': 'fake_news'})
            self.event_log.append(f"Year {self.year}: Fake news event.")
        # Environmental event
        if random.random() < 0.05:
            for c in self.citizens:
                c.wealth -= random.gauss(10, 5)
                c.remember({'type': 'environmental', 'amount': -10})
            self.event_log.append(f"Year {self.year}: Environmental disaster.")
        # Random event
        if random.random() < 0.03:
            self.event_log.append(f"Year {self.year}: Random event occurred.")

    def inject_event(self, event_type: str, **kwargs):
        """Inject a what-if event into the simulation."""
        if event_type == 'revolution':
            for c in self.citizens:
                c.dissent += 0.5
            self.event_log.append(f"Year {self.year}: Revolution event injected. Dissent increased.")
        elif event_type == 'new_law':
            law = kwargs.get('law', 'Unknown Law')
            for c in self.citizens:
                c.memory.append({'type': 'law', 'law': law})
            self.event_log.append(f"Year {self.year}: New law injected: {law}.")
        elif event_type == 'economic_boom':
            for c in self.citizens:
                c.wealth += 50
            self.event_log.append(f"Year {self.year}: Economic boom injected. Wealth increased.")
        elif event_type == 'pandemic':
            for c in self.citizens:
                c.happiness -= 0.2
                c.dissent += 0.2
            self.event_log.append(f"Year {self.year}: Pandemic event injected. Happiness down, dissent up.")
        # Add more event types as needed

    def _collect_metrics(self):
        wealths = [c.wealth for c in self.citizens]
        gini = self._gini(wealths)
        avg_dissent = np.mean([c.dissent for c in self.citizens])
        avg_happiness = np.mean([c.happiness for c in self.citizens])
        ideologies = [c.ideology for c in self.citizens]
        polarization = np.std(ideologies)
        avg_corruption = np.mean([p.corruption for p in self.politicians]) if self.politicians else 0
        press_freedom = 1.0 - abs(self.media.bias)
        protests = int(avg_dissent > 0.5)
        self.history['year'].append(self.year)
        self.history['gini'].append(gini)
        self.history['avg_dissent'].append(avg_dissent)
        self.history['avg_happiness'].append(avg_happiness)
        self.history['polarization'].append(polarization)
        self.history['avg_corruption'].append(avg_corruption)
        self.history['press_freedom'].append(press_freedom)
        self.history['protests'].append(protests)
        self.history['laws_overturned'].append(self.laws_overturned)

    def _gini(self, values):
        sorted_vals = sorted(values)
        n = len(values)
        cumvals = np.cumsum(sorted_vals)
        gini = (n + 1 - 2 * np.sum(cumvals) / cumvals[-1]) / n if cumvals[-1] > 0 else 0
        return round(gini, 3)

    def get_metrics_df(self):
        return pd.DataFrame(self.history)

    def seed_from_dataframe(self, df):
        """Seed agents from a DataFrame (e.g., uploaded scenario)."""
        for i, row in df.iterrows():
            if i < len(self.citizens):
                c = self.citizens[i]
                for attr in ['wealth', 'ideology', 'happiness', 'dissent']:
                    if attr in row:
                        setattr(c, attr, float(row[attr])) 