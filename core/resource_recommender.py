"""
GOD's EYE — Resource Recommender
Maps detected events to resource deployment suggestions.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Recommendation:
    action: str
    units: str
    priority: str  # immediate, high, standard
    icon: str = ""


class ResourceRecommender:
    RULES = {
        "Fire/Smoke Emergency": Recommendation(
            action="Alert fire brigade immediately. Initiate zone evacuation.",
            units="1 Fire unit + 1 Ambulance",
            priority="immediate",
            icon="🔥"
        ),
        "Person Fall Detected": Recommendation(
            action="Dispatch medical assistance to the area.",
            units="1 Ambulance",
            priority="high",
            icon="🚑"
        ),
        "Vehicle Incident": Recommendation(
            action="Deploy traffic police and medical team.",
            units="1 Traffic unit + 1 Ambulance",
            priority="high",
            icon="🚔"
        ),
        "High Crowd Risk": Recommendation(
            action="Deploy police units for crowd management.",
            units="2-3 Police units",
            priority="high",
            icon="👮"
        ),
        "Elevated Crowd Activity": Recommendation(
            action="Increase monitoring. Alert patrol units on standby.",
            units="1 Patrol unit on standby",
            priority="standard",
            icon="📡"
        ),
        "Anomalous Behavior": Recommendation(
            action="Focus camera. Dispatch reconnaissance patrol.",
            units="1 Patrol unit",
            priority="standard",
            icon="⚠️"
        ),
        "Predictive Pre-Alert": Recommendation(
            action="Pre-position response units. Increase surveillance coverage.",
            units="1-2 Units pre-positioned",
            priority="standard",
            icon="🔮"
        ),
    }

    def recommend(self, event) -> Optional[Recommendation]:
        return self.RULES.get(event.event_type)
