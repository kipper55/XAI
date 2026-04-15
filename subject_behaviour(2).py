

# subject_behaviour.py
import numpy as np

# --- BEHAVIORAL CONSTANTS (Refined via SPS Research) ---
FEATURE_PROBABILITIES = {
    "linear":    0.25, "field":    0.14, "structure": 0.13, "road":   0.13,
    "drainage":  0.12, "water":    0.08, "woodland":  0.07, "rock":   0.04,
    "scrub":     0.03, "brush":    0.01
}

assert abs(sum(FEATURE_PROBABILITIES.values()) - 1.0) < 0.01, (
    f"FEATURE_PROBABILITIES weights sum to {sum(FEATURE_PROBABILITIES.values()):.3f}, expected ~1.0"
)

class LPBExplainability:
    """
    Multi-Criteria Evaluation (MCE) for Slope-Based Probabilistic Search (SPS).
    Based on 'Terrain-Informed UAV Path Planning for Mountain Search' (2026).

    Weights: 50% Terrain/Slope, 30% Distance/Persistence, 20% Features.
    """

    @staticmethod
    def calculate_slope_penalty(elevation_grid, lkp_elevation=None, resolution=30):
        """
        Exponential decay on steepness with optional gravity bias.
        Called by region_processing.py to precompute and save slope_p.npy.
        Physical grounding: lost subjects avoid high-energy (steep) paths.
        """
        dy, dx     = np.gradient(elevation_grid, resolution, resolution)
        slope_deg  = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
        prob_slope = np.exp(-0.18 * slope_deg)

        if lkp_elevation is not None:
            gravity_factor = np.where(elevation_grid < lkp_elevation, 1.2, 0.8)
            prob_slope    *= gravity_factor

        return prob_slope / np.max(prob_slope)

    @staticmethod
    def get_combined_poc(slope_p, dist_p, feature_p_map):
        """
        Weighted combination of the three probability layers.
        Called by subject_optimise.py's score_field() as the single
        source of truth for MCE logic.
        """
        combined = (slope_p * 0.5) + (dist_p * 0.3) + (feature_p_map * 0.2)
        return combined / np.sum(combined)