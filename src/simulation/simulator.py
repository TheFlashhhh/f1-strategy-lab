"""F1 race simulator for testing strategies."""

import logging
from dataclasses import dataclass
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class RaceState:
    """Current state of the race."""

    lap: int
    position: int
    fuel_level: float
    tire_age: int
    tire_compound: str


class RaceSimulator:
    """Simulate F1 races with different strategies."""

    def __init__(self, total_laps: int = 70):
        """
        Initialize race simulator.

        Args:
            total_laps: Total number of laps in the race.
        """
        self.total_laps = total_laps
        self.current_state = RaceState(
            lap=0, position=1, fuel_level=100.0, tire_age=0, tire_compound="medium"
        )

    def simulate_lap(self) -> float:
        """
        Simulate a single lap.

        Returns:
            Lap time in seconds.
        """
        lap_time = self._calculate_lap_time()
        self.current_state.lap += 1
        self._update_tire_age()
        self._consume_fuel()

        return lap_time

    def simulate_race(self) -> List[float]:
        """
        Simulate a complete race.

        Returns:
            List of lap times.
        """
        lap_times = []

        while self.current_state.lap < self.total_laps:
            lap_time = self.simulate_lap()
            lap_times.append(lap_time)

        logger.info(f"Simulated race: {len(lap_times)} laps")
        return lap_times

    def pit_stop(self, new_compound: str) -> None:
        """
        Perform a pit stop.

        Args:
            new_compound: New tire compound (soft, medium, hard).
        """
        self.current_state.tire_age = 0
        self.current_state.tire_compound = new_compound
        logger.info(f"Pit stop executed: switched to {new_compound} tires")

    def _calculate_lap_time(self) -> float:
        """Calculate lap time based on current conditions."""
        base_time = 90.0  # Base lap time in seconds
        return base_time

    def _update_tire_age(self) -> None:
        """Update tire age and degradation."""
        self.current_state.tire_age += 1

    def _consume_fuel(self) -> None:
        """Consume fuel during the lap."""
        fuel_per_lap = 1.0
        self.current_state.fuel_level -= fuel_per_lap
