


# main.py
import sys
import traceback

def run_pipeline():
    print("=" * 70)
    print("   BEN NEVIS SAR PIPELINE")
    print("=" * 70)

    # ── Step 1: Region Processing ──────────────────────────────────────────
    print("\n[STEP 1/4] Region Processing — Terrain & Feature Extraction")
    print("-" * 70)
    try:
        from region_processing import export_final_labeled_ben_nevis
        result = export_final_labeled_ben_nevis()
        if not result:
            raise RuntimeError("region_processing returned no success signal.")
        print("[STEP 1] ✓ Complete\n")
    except Exception as e:
        print(f"[STEP 1] ✗ Failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    # ── Step 2: Subject Behaviour ──────────────────────────────────────────
    print("\n[STEP 2/4] Subject Behaviour — SPS Logic Validation")
    print("-" * 70)
    try:
        from subject_behaviour import LPBExplainability, FEATURE_PROBABILITIES
        import numpy as np

        print(f"  Feature types loaded:  {len(FEATURE_PROBABILITIES)}")
        print(f"  Weight sum check:      {sum(FEATURE_PROBABILITIES.values()):.2f} (should be ~1.0)")

        dummy_elev = np.random.uniform(600, 1344, (10, 10))
        logic      = LPBExplainability()
        slope_p    = logic.calculate_slope_penalty(dummy_elev, lkp_elevation=900)
        poc        = logic.get_combined_poc(slope_p, slope_p, slope_p)
        print(f"  Slope penalty range:   [{slope_p.min():.3f}, {slope_p.max():.3f}]")
        print(f"  POC sum check:         {poc.sum():.6f} (should be ~1.0)")
        print("[STEP 2] ✓ Complete\n")
    except Exception as e:
        print(f"[STEP 2] ✗ Failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    # ── Step 3: Subject Optimisation ──────────────────────────────────────
    print("\n[STEP 3/4] Subject Optimisation — Weight Fitting via L-BFGS-B")
    print("-" * 70)
    try:
        from subject_optimise import main as optimise_main
        theta_hat = optimise_main()
        if theta_hat is None:
            raise RuntimeError("optimise_main returned no weights.")
        print("[STEP 3] ✓ Complete\n")
    except Exception as e:
        print(f"[STEP 3] ✗ Failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    # ── Step 4: Outputs ────────────────────────────────────────────────────
    print("\n[STEP 4/4] Outputs — Explainability Report, CSV & GUI")
    print("-" * 70)
    try:
        from outputs import main as outputs_main
        outputs_main(theta_hat=theta_hat)
        print("[STEP 4] ✓ Complete\n")
    except Exception as e:
        print(f"[STEP 4] ✗ Failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    print("=" * 70)
    print("   PIPELINE COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    run_pipeline()
