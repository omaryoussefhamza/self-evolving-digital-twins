"""Stage and condition identities for each communication round."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Step:
    scenario: str
    condition: str
    shift: float = 0.0  # used by the synthetic distribution only


def schedule(rounds: int) -> list[Step]:
    if rounds < 8:
        raise ValueError("Use at least eight rounds for two or more rounds per stage.")
    length, extra = divmod(rounds, 4)
    stable, sudden, gradual, returning = [length + (i < extra) for i in range(4)]
    return (
        [Step("stable", "original", 0.0)] * stable
        + [Step("sudden", "shifted", 1.0)] * sudden
        + [
            Step("gradual", f"gradual_{i}", 1.0 - 0.5 * i / gradual)
            for i in range(1, gradual + 1)
        ]
        + [Step("return", "original", 0.0)] * returning
    )
