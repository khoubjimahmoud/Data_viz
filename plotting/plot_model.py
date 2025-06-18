# plotting/plot_model.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import re

def plot_model_pph_last12_df(
    df: pd.DataFrame,
    last_12_months: list,
    sel_models: list
):
    """
    For each model in sel_models, over last_12_months:
      • Compute problems_count = Σ [ (PPH% / 100) × Sample ] for Body + Interior.
      • Compute sample_count   = Sample from Body noises (or Interior if Body missing).
      • sum_problems += problems_count; sum_sample += sample_count.
    Combined PPH = (sum_problems / sum_sample)*100.
    Drop models with PPH == 0, then barplot sorted ascending.
    Returns (PNG_buffer, df_res).
    """
    if not sel_models:
        return None

    key_model = "Vehicle Model-Body-Trim"
    key_brand = "Brand"
    key_noise = "Noise Typology"

    records = []
    for mdl in sel_models:
        subset = df[df[key_model] == mdl]

        # Determine the “final brand” exactly as before
        brands_for_this_model = (
            subset.loc[
                (subset[key_brand] != "Total (qualified)"),
                key_brand
            ]
            .dropna()
            .unique()
            .tolist()
        )
        if set(brands_for_this_model) == {"Opel", "Vauxhall"}:
            final_brand = "Opel/Vauxhall"
        elif brands_for_this_model:
            final_brand = brands_for_this_model[0]
        else:
            final_brand = "Unknown"

        sum_problems = 0.0
        sum_sample   = 0.0

        # --- corrected loop ---
        for month in last_12_months:
            pr_col = f"{month} PPH"
            sa_col = f"{month} Sample"
            if pr_col not in subset.columns or sa_col not in subset.columns:
                continue

            # Body noises
            mask_body = (
                (subset[key_brand] == "Total (qualified)") &
                (subset[key_noise] == "Body noises")
            )
            if mask_body.any():
                raw_pb = subset.loc[mask_body, pr_col].replace({'*': np.nan, '-': np.nan})
                raw_sb = subset.loc[mask_body, sa_col].replace({'*': np.nan, '-': np.nan})
                P_body = pd.to_numeric(raw_pb.iat[0], errors="coerce")
                S_body = pd.to_numeric(raw_sb.iat[0], errors="coerce")
                if not np.isnan(P_body) and not np.isnan(S_body) and S_body > 0:
                    prob_body = (P_body / 100.0) * S_body
                else:
                    prob_body = None
                    S_body    = None
            else:
                prob_body = None
                S_body    = None

            # Interior noises
            mask_int = (
                (subset[key_brand] == "Total (qualified)") &
                (subset[key_noise] == "Interior noises")
            )
            if mask_int.any():
                raw_pi = subset.loc[mask_int, pr_col].replace({'*': np.nan, '-': np.nan})
                raw_si = subset.loc[mask_int, sa_col].replace({'*': np.nan, '-': np.nan})
                P_int = pd.to_numeric(raw_pi.iat[0], errors="coerce")
                S_int = pd.to_numeric(raw_si.iat[0], errors="coerce")
                if not np.isnan(P_int) and not np.isnan(S_int) and S_int > 0:
                    prob_int = (P_int / 100.0) * S_int
                else:
                    prob_int = None
                    S_int    = None
            else:
                prob_int = None
                S_int    = None

            # combine counts
            probs = [(prob_body or 0.0), (prob_int or 0.0)]
            # sample: prefer Body if present, else Interior
            samp = S_body if S_body is not None else S_int
            if samp is None:
                continue

            # accumulate
            sum_problems += sum(probs)
            sum_sample   += samp

        # compute final PPH
        comb_pph = (sum_problems / sum_sample) * 100.0 if sum_sample > 0 else 0.0

        records.append({
            "Model": mdl,
            "Brand": final_brand,
            "Problems": sum_problems,
            "Sample": sum_sample,
            "PPH": comb_pph
        })

    # Build DataFrame and drop zeros
    df_res = pd.DataFrame(records)
    df_res = df_res[df_res["PPH"] != 0.0]
    if df_res.empty:
        return None
    df_res = df_res.sort_values("PPH", ascending=True).reset_index(drop=True)

    # Plotting (identical style to before)
    plt.figure(figsize=(14, 8))
    ax = plt.gca()
    bars = sns.barplot(
        data=df_res, x="Model", y="PPH", hue="Brand",
        dodge=False, palette="tab20", ax=ax
    )
    # annotate
    for patch in bars.patches:
        h = patch.get_height()
        if h > 0:
            x = patch.get_x() + patch.get_width() / 2
            ax.text(x, h + df_res["PPH"].max()*0.01, f"{h:.2f}",
                    ha="center", va="bottom", fontsize=8, rotation=90)

    ax.set_xlabel("Vehicle Model")
    ax.set_ylabel("PPH")
    ax.set_title("Combined Body + Interior PPH Over Last 12 Months")
    ax.set_xticklabels(df_res["Model"], rotation=90, fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", title="Brand", fontsize=9, title_fontsize=10)
    ax.set_ylim(0, df_res["PPH"].max() * 1.15)

    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)
    plt.close()

    return buf, df_res
