import statistics

def to_params(state) -> dict:
    # Depth: driven by bass energy (already a 0-1 ratio, no scaling needed)
    depth = max(0.0, min(state.bass_energy, 1.0))

    # Time between each beat and the next, used to measure how regular the beat is
    gaps = [state.beat_times[i+1] - state.beat_times[i] for i in range(len(state.beat_times) - 1)]
    if len(gaps) >= 2:
        strength = max(0.0, min(1.0 - statistics.stdev(gaps) / statistics.mean(gaps), 1.0))
    else:
        strength = 0.5  # not enough beats to measure consistency, use a neutral fallback

    # Edge: driven by beat strength (how consistent the beat spacing is)
    edge = strength

    # Consistency: inverse of energy, so calmer music holds frames steadier
    consistency = max(0.0, min(1.0 - state.energy, 1.0))

    # Creativity: driven by arousal, scaled to the 0-19 t_index range, then inverted
    creativity_intent = max(0, min(int(state.arousal * 19), 19))
    t_index = 19 - creativity_intent

    return {
        # controlnets is positional, so all three are sent every cycle even if only one value changed
        "controlnets": [
            {"conditioning_scale": depth},        # index 0: Depth
            {"conditioning_scale": edge},         # index 1: Edge
            {"conditioning_scale": consistency},  # index 2: Consistency
        ],
        "t_index_list": str(t_index),
    }