from dataclasses import dataclass
from enum import Enum


CASE_FILE = "cases/glover37.json"


class Action(str, Enum):
    SET_GENERATOR_P = "set_generator_p"
    DISCONNECT = "disconnect"


@dataclass(frozen=True)
class Change:
    action: Action
    element: str
    value: float | None = None


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    changes: tuple[Change, ...]


SCENARIOS = {
    "A1": Scenario(
        name="A1",
        description=(
            "Generator 28 dispatched to 0 MW while remaining "
            "connected and synchronized."
        ),
        changes=(
            Change(
                action=Action.SET_GENERATOR_P,
                element="generator.g28-1",
                value=0.0,
            ),
        ),
    ),

    "A2": Scenario(
        name="A2",
        description="Generator 28 disconnected.",
        changes=(
            Change(
                action=Action.DISCONNECT,
                element="generator.g28-1",
            ),
        ),
    ),

    "A3": Scenario(
        name="A3",
        description="Generator 14 disconnected.",
        changes=(
            Change(
                action=Action.DISCONNECT,
                element="generator.g14-1",
            ),
        ),
    ),

    "B1": Scenario(
        name="B1",
        description="Line 14-34 circuit 1 disconnected.",
        changes=(
            Change(
                action=Action.DISCONNECT,
                element="line.line 14-34 ckt 1",
            ),
        ),
    ),

    "B2": Scenario(
        name="B2",
        description="Line 21-48 circuit 1 disconnected.",
        changes=(
            Change(
                action=Action.DISCONNECT,
                element="line.line 21-48 ckt 1",
            ),
        ),
    ),

    "B3": Scenario(
        name="B3",
        description="Transformer 28-29 circuit 1 disconnected.",
        changes=(
            Change(
                action=Action.DISCONNECT,
                element="transformer.transformer 28-29 ckt 1",
            ),
        ),
    ),
}