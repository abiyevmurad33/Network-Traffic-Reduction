from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class Obs:
    """
    One per-tic observation used by predictors and evaluators.

    action_mask is the action executed *during* this tic (i.e., between tic and tic+1).
    """

    tic: int
    x: float
    y: float
    z: float
    yaw: float
    pitch: float
    action_mask: int


class Predictor(ABC):
    """
    Stateful predictor that advances one tic at a time.

    - reset(): clears internal state (use on correction updates)
    - observe(obs): inject a known (authoritative) state at obs.tic
    - step(action_mask): predict next obs (tic+1) using internal state and the action at current tic
    """

    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def observe(self, obs: Obs) -> None: ...

    @abstractmethod
    def step(self, action_mask: int) -> Obs: ...
