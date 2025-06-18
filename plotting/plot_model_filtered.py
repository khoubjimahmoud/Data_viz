# plotting/plot_model_filtered.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

def plot_model_pph_filtered_df(
    df: pd.DataFrame,
    months_sel: list,
    sel_models: list,
    sel_noises: list
):
    """
    Like plot_model_pph_last12_df but applies a noise + period filter:
      1) For each model & month:
         - problems_month = sum of df[f"{month} Problems"] across sel_noises 
         - sample_month   = df[f"{month} Sample"] from the first noise in sel_noises
      2) sum_problems = Σ_problems_month
         sum_sample   = Σ_sample_month
      3) combined PPH = (sum_problems / sum_sample) * 100
      4) drop models with PPH == 0, sort ascending
      5) barplot as in Plot 2
    """
    if not sel_models:
        return None

    key_model = "Vehicle Model-Body-Trim"
    key_brand = "Brand"
    key_noise = "Noise Typology"

    records = []
    first_noise = sel_noises[0]

    for mdl in sel_models:
        subset = df[df[key_model] == mdl]
        # determine brand exactly as in plot_model.py
        brands = (
            subset
            .loc[
                subset[key_brand] != "Total (qualified)",  # row filter
                key_brand                                   # column selector
            ]
            .dropna()
            .unique()
            .tolist()
        )

        if set(brands) == {"Opel", "Vauxhall"}:
            final_brand = "Opel/Vauxhall"
        elif brands:
            final_brand = brands[0]
        else:
            final_brand = "Unknown"

        sum_problems = 0.0
        sum_sample   = 0.0

        for month in months_sel:
            pr_col = f"{month} Problems"
            sa_col = f"{month} Sample"
            if pr_col not in subset.columns or sa_col not in subset.columns:
                continue

            # 1a) sum Problems over all sel_noises
            tot_p = 0.0
            for nt in sel_noises:
                mask = (
                    (subset[key_brand] == "Total (qualified)") &
                    (subset[key_noise] == nt)
                )
                if not mask.any():
                    continue
                raw_pr = subset.loc[mask, pr_col].replace({'*':np.nan,'-':np.nan})
                p = pd.to_numeric(raw_pr.iat[0], errors="coerce")
                if pd.isna(p):
                    continue
                tot_p += p

            # 1b) fetch Sample from the first noise only
            mask0 = (
                (subset[key_brand] == "Total (qualified)") &
                (subset[key_noise] == first_noise)
            )
            if not mask0.any():
                continue
            raw_sa0 = subset.loc[mask0, sa_col].replace({'*':np.nan,'-':np.nan})
            s0 = pd.to_numeric(raw_sa0.iat[0], errors="coerce")
            if pd.isna(s0) or s0 <= 0:
                continue

            # accumulate
            sum_problems += tot_p
            sum_sample   += s0

        # compute PPH
        if sum_sample > 0:
            comb_pph = (sum_problems / sum_sample) * 100.0
        else:
            comb_pph = 0.0

        records.append({
            "Model": mdl,
            "Brand": final_brand,
            "Problems": sum_problems,
            "Sample":   sum_sample,
            "PPH":      comb_pph
        })

    df_res = pd.DataFrame(records)
    df_res = df_res[df_res["PPH"] != 0.0]
    if df_res.empty:
        return None
    df_res = df_res.sort_values("PPH", ascending=True).reset_index(drop=True)

    # plotting (same style as plot_model_pph_last12_df)
    plt.figure(figsize=(14, 8))
    ax = plt.gca()
    bars = sns.barplot(
        data=df_res,
        x="Model", y="PPH", hue="Brand",
        dodge=False, palette="tab20", ax=ax
    )
    # annotate
    for patch in bars.patches:
        h = patch.get_height()
        if h > 0:
            x = patch.get_x() + patch.get_width()/2
            ax.text(x, h + df_res["PPH"].max()*0.01, f"{h:.2f}",
                    ha="center", va="bottom", fontsize=8, rotation=90)

    ax.set_xlabel("Vehicle Model")
    ax.set_ylabel("PPH")
    ax.set_title("Combined PPH by Model (Filtered by Noise & Period)")
    ax.set_xticklabels(df_res["Model"], rotation=90, fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", title="Brand", fontsize=9, title_fontsize=10)
    ax.set_ylim(0, df_res["PPH"].max()*1.15)
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)
    plt.close()
    return buf, df_res
