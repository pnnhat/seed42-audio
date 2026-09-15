def to_params(state) -> dict:
    # Depth: driven by brightness, scaled from typical spectral centroid range down to 0-1
    depth = max(0.0, min(state.brightness / 3000, 1.0))

    # Edge: driven by tempo, scaled from typical BPM range (60-200) down to 0-1
    edge = max(0.0, min(state.tempo / 200, 1.0))

    # Consistency: inverse of energy, so calmer music holds frames steadier
    consistency = max(0.0, min(1.0 - state.energy, 1.0))

    # Creativity: driven by arousal, scaled to the 0-19 t_index range, then inverted
    # (UI shows Creativity, but the API takes t_index, where high creativity = low t_index)
    creativity_intent = max(0, min(int(state.arousal * 19), 19))
    t_index = 19 - creativity_intent

    return {
        "controlnets": [
            {"conditioning_scale": depth},        # index 0: Depth
            {"conditioning_scale": edge},          # index 1: Edge
            {"conditioning_scale": consistency},   # index 2: Consistency
        ],
        "t_index_list": [t_index],
    }
