


# subject_optimise.py
import os
import numpy as np
import rioxarray
from scipy.optimize import minimize
from subject_behaviour import LPBExplainability

# --- CONFIG ---
DATASET_DIR   = r"C:\Users\antvi\PYTHON_PROJECTS\SAREnv\sarenv_dataset\ben_nevis"
TIF_PATH      = os.path.join(DATASET_DIR, "ben_nevis_geo.tif")
PATH_SLOPE_P  = os.path.join(DATASET_DIR, "slope_p.npy")
PATH_DIST_P   = os.path.join(DATASET_DIR, "dist_p.npy")
PATH_HEATMAP  = os.path.join(DATASET_DIR, "heatmap.npy")

# ---------------------------------------------------------
# 1. ENVIRONMENT LOADING
# ---------------------------------------------------------
def load_environment():
    # Files guaranteed by region_processing.py (Step 1)
    rds         = rioxarray.open_rasterio(TIF_PATH)
    elevation   = rds.values[0] if rds.values.ndim == 3 else rds.values

    # LKP at center
    center_y, center_x = elevation.shape[0] // 2, elevation.shape[1] // 2
    lkp_elev    = elevation[center_y, center_x]

    # Load pre-computed layers from region_processing
    slope_p     = np.load(PATH_SLOPE_P)
    dist_p      = np.load(PATH_DIST_P)
    heatmap     = np.load(PATH_HEATMAP)
    feature_map = heatmap / heatmap.max()

    return {
        "rds":         rds,
        "elevation":   elevation,
        "slope_p":     slope_p,
        "dist_p":      dist_p,
        "feature_map": feature_map,
    }

# ---------------------------------------------------------
# 2. BASELINE MODEL & SURVIVOR GENERATION
# ---------------------------------------------------------
def build_baseline_poc(env, w_slope=0.5, w_dist=0.3, w_feat=0.2):
    final_poc = (
        env["slope_p"]     * w_slope +
        env["dist_p"]      * w_dist  +
        env["feature_map"] * w_feat
    )
    final_poc = np.clip(final_poc, 1e-12, None)
    final_poc /= final_poc.sum()
    return final_poc

def generate_survivors(env, num_survivors=1_000_000):
    final_poc      = build_baseline_poc(env)
    flat_poc       = final_poc.ravel()
    sample_indices = np.random.choice(len(flat_poc), size=num_survivors, p=flat_poc)
    incidents      = [{"end": int(idx)} for idx in sample_indices]
    return incidents, final_poc

# ---------------------------------------------------------
# 3. PARAMETRIC MODEL
# ---------------------------------------------------------
def vec_to_theta(vec, names):
    return {name: vec[i] for i, name in enumerate(names)}

def theta_to_vec(theta, names):
    return np.array([theta[name] for name in names])

def score_field(theta, env):
    scaled_slope   = env["slope_p"]     * (theta["slope"] / 0.5)
    scaled_dist    = env["dist_p"]      * (theta["dist"]  / 0.3)
    scaled_feature = env["feature_map"] * (theta["feat"]  / 0.2)
    return LPBExplainability.get_combined_poc(scaled_slope, scaled_dist, scaled_feature)

# ---------------------------------------------------------
# 4. NEGATIVE LOG-LIKELIHOOD
# ---------------------------------------------------------
def nll(theta_vec, incidents, feature_names, env):
    theta     = vec_to_theta(theta_vec, feature_names)
    S         = score_field(theta, env).ravel()
    S_max     = S.max()
    exp_S     = np.exp(S - S_max)
    log_Z     = np.log(exp_S.sum()) + S_max
    log_probs = S - log_Z
    return -sum(log_probs[inc["end"]] for inc in incidents)

def optimise_weights(incidents, env, initial_theta):
    feature_names = list(initial_theta.keys())
    theta_vec0    = theta_to_vec(initial_theta, feature_names)
    result        = minimize(
        nll,
        theta_vec0,
        args=(incidents, feature_names, env),
        method="L-BFGS-B"
    )
    return vec_to_theta(result.x, feature_names), result

# ---------------------------------------------------------
# 5. MAIN
# ---------------------------------------------------------
def main():
    env                     = load_environment()
    incidents, baseline_poc = generate_survivors(env, num_survivors=1_000_000)

    initial_theta = {"slope": 0.5, "dist": 0.3, "feat": 0.2}
    feature_names = list(initial_theta.keys())

    baseline_nll        = nll(theta_to_vec(initial_theta, feature_names),
                              incidents, feature_names, env)
    theta_hat, result   = optimise_weights(incidents, env, initial_theta)
    final_nll           = result.fun

    avg_log_prob_baseline = -baseline_nll / len(incidents)
    avg_log_prob_final    = -final_nll    / len(incidents)

    print("\n" + "═" * 70)
    print("   SUBJECT OPTIMISATION: BEN NEVIS SPS SURROGATE MODEL")
    print("═" * 70)
    print(f"  Survivors used:           {len(incidents):,}")
    print(f"  Baseline NLL:             {baseline_nll:.3f}")
    print(f"  Optimised NLL:            {final_nll:.3f}")
    print(f"  ΔNLL (baseline - final):  {baseline_nll - final_nll:.3f}")
    print()
    print(f"  Avg log P (baseline):     {avg_log_prob_baseline:.4f}")
    print(f"  Avg log P (optimised):    {avg_log_prob_final:.4f}")
    print()
    print("  Optimised weights:")
    for k, v in theta_hat.items():
        print(f"    {k:5s}: {v:.4f}")
    print(f"  Converged: {result.success}")
    print("═" * 70 + "\n")

    return theta_hat

if __name__ == "__main__":
    main()