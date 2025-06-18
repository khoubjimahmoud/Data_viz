# plotting/plot_brand.py

import pandas as pd
import numpy as np
from io import BytesIO
import matplotlib.pyplot as plt

def compute_raw_pph(df, months_str, brands):
    """
    Build a DataFrame of raw PPH values for each brand × month.
    If brand == "Opel + Vauxhall", combine Opel & Vauxhall.
    Returns a DataFrame (index=months_str, columns=brands).
    """
    key_model, key_brand, key_noise = df.columns[:3]
    tbl = pd.DataFrame(index=months_str, columns=brands, dtype=float)

    for b in brands:
        for m in months_str:
            col_pph  = f"{m} PPH"
            col_samp = f"{m} Sample"

            # If this PPH/Sample column doesn’t exist, skip
            if col_pph not in df.columns or col_samp not in df.columns:
                tbl.at[m, b] = np.nan
                continue

            if b != "Opel + Vauxhall":
                # 1) Body noises for brand b
                mask_body = (
                    (df[key_brand] == "Total (qualified)")
                    & (df[key_noise] == "Body noises")
                    & (df[key_model] == b)
                )
                if mask_body.any():
                    raw_pb = df.loc[mask_body, col_pph].replace({'*': np.nan, '-': np.nan})
                    raw_sb = df.loc[mask_body, col_samp].replace({'*': np.nan, '-': np.nan})
                    P_body = pd.to_numeric(raw_pb.iat[0], errors="coerce")
                    S_body = pd.to_numeric(raw_sb.iat[0], errors="coerce")
                else:
                    P_body = S_body = np.nan

                # 2) Interior noises for brand b
                mask_intr = (
                    (df[key_brand] == "Total (qualified)")
                    & (df[key_noise] == "Interior noises")
                    & (df[key_model] == b)
                )
                if mask_intr.any():
                    raw_pi = df.loc[mask_intr, col_pph].replace({'*': np.nan, '-': np.nan})
                    raw_si = df.loc[mask_intr, col_samp].replace({'*': np.nan, '-': np.nan})
                    P_intr = pd.to_numeric(raw_pi.iat[0], errors="coerce")
                    S_intr = pd.to_numeric(raw_si.iat[0], errors="coerce")
                else:
                    P_intr = S_intr = np.nan

                # Convert PPH → Problems (counts)
                prob_body = (P_body / 100.0) * S_body if pd.notna(P_body) and pd.notna(S_body) else np.nan
                prob_intr = (P_intr / 100.0) * S_intr if pd.notna(P_intr) and pd.notna(S_intr) else np.nan

                if pd.notna(prob_body) and pd.notna(prob_intr):
                    total_prob = prob_body + prob_intr
                    total_samp = ((S_body if pd.notna(S_body) else 0.0)
                                  + (S_intr if pd.notna(S_intr) else 0.0))
                elif pd.notna(prob_body):
                    total_prob = prob_body
                    total_samp = S_body
                elif pd.notna(prob_intr):
                    total_prob = prob_intr
                    total_samp = S_intr
                else:
                    total_prob = total_samp = np.nan

                if pd.notna(total_prob) and pd.notna(total_samp) and total_samp > 0:
                    tbl.at[m, b] = (total_prob / total_samp) * 100.0
                else:
                    tbl.at[m, b] = np.nan

            else:
                # Combine Opel + Vauxhall
                def fetch_P_S(brand_name, noise_type):
                    mask = (
                        (df[key_brand] == "Total (qualified)")
                        & (df[key_noise] == noise_type)
                        & (df[key_model] == brand_name)
                    )
                    if mask.any():
                        raw_P = df.loc[mask, col_pph].replace({'*': np.nan, '-': np.nan})
                        raw_S = df.loc[mask, col_samp].replace({'*': np.nan, '-': np.nan})
                        return (
                            pd.to_numeric(raw_P.iat[0], errors="coerce"),
                            pd.to_numeric(raw_S.iat[0], errors="coerce")
                        )
                    return np.nan, np.nan

                # Opel body/interior
                P_body_op, S_body_op = fetch_P_S("Opel", "Body noises")
                P_int_op,  S_int_op  = fetch_P_S("Opel", "Interior noises")
                prob_body_op = (
                    (P_body_op / 100.0) * S_body_op
                    if pd.notna(P_body_op) and pd.notna(S_body_op) else np.nan
                )
                prob_int_op = (
                    (P_int_op / 100.0) * S_int_op
                    if pd.notna(P_int_op) and pd.notna(S_int_op) else np.nan
                )

                if pd.notna(prob_body_op) and pd.notna(prob_int_op):
                    total_prob_op = prob_body_op + prob_int_op
                    total_samp_op = ((S_body_op if pd.notna(S_body_op) else 0.0)
                                     + (S_int_op if pd.notna(S_int_op) else 0.0))
                elif pd.notna(prob_body_op):
                    total_prob_op = prob_body_op
                    total_samp_op = S_body_op
                elif pd.notna(prob_int_op):
                    total_prob_op = prob_int_op
                    total_samp_op = S_int_op
                else:
                    total_prob_op = total_samp_op = np.nan

                # Vauxhall body/interior
                P_body_vx, S_body_vx = fetch_P_S("Vauxhall", "Body noises")
                P_int_vx,  S_int_vx  = fetch_P_S("Vauxhall", "Interior noises")
                prob_body_vx = (
                    (P_body_vx / 100.0) * S_body_vx
                    if pd.notna(P_body_vx) and pd.notna(S_body_vx) else np.nan
                )
                prob_int_vx = (
                    (P_int_vx / 100.0) * S_int_vx
                    if pd.notna(P_int_vx) and pd.notna(S_int_vx) else np.nan
                )

                if pd.notna(prob_body_vx) and pd.notna(prob_int_vx):
                    total_prob_vx = prob_body_vx + prob_int_vx
                    total_samp_vx = ((S_body_vx if pd.notna(S_body_vx) else 0.0)
                                     + (S_int_vx if pd.notna(S_int_vx) else 0.0))
                elif pd.notna(prob_body_vx):
                    total_prob_vx = prob_body_vx
                    total_samp_vx = S_body_vx
                elif pd.notna(prob_int_vx):
                    total_prob_vx = prob_int_vx
                    total_samp_vx = S_int_vx
                else:
                    total_prob_vx = total_samp_vx = np.nan

                if (
                    pd.notna(total_prob_op) and pd.notna(total_prob_vx)
                    and pd.notna(total_samp_op) and pd.notna(total_samp_vx)
                    and (total_samp_op + total_samp_vx) > 0
                ):
                    combined_prob = total_prob_op + total_prob_vx
                    combined_samp = total_samp_op + total_samp_vx
                    tbl.at[m, b] = (combined_prob / combined_samp) * 100.0
                else:
                    tbl.at[m, b] = np.nan

    return tbl


def plot_pph_df(tbl, title):
    """
    Plot each column of `tbl` (indexed by month strings) over time,
    annotate the last‐month numeric value AND brand name,
    label the y‐axis as "PPH," and increase font sizes for clarity.
    """
    x = pd.to_datetime(tbl.index, format="%b-%Y", errors="coerce")

    fig, ax = plt.subplots(figsize=(16, 10))

    # Loop through brands to draw lines + annotate last value + brand name
    for b in tbl.columns:
        y = tbl[b].astype(float)

        # Draw line + markers
        line, = ax.plot(x, y, marker="o", label=b)

        # Label the final non‐NaN point with "value + brand name"
        non_na = y.dropna()
        if not non_na.empty:
            last_idx = non_na.index[-1]     # e.g. "Jun-2024"
            last_val = non_na.iat[-1]       # numeric PPH
            x_last   = pd.to_datetime(last_idx, format="%b-%Y", errors="coerce")
            x_text   = x_last + pd.Timedelta(days=6)

            # Draw the combined annotation: "<value>  <brand>"
            ax.text(
                x_text,
                last_val,
                f"{last_val:.2f}  {b}",    # numeric and brand name
                va="bottom",
                ha="left",
                fontsize=14,               # larger font for annotation
                color=line.get_color()
            )

    # Format axes & labels
    ax.set_xticks(x)
    ax.set_xticklabels(tbl.index, rotation=45, fontsize=12)   # larger x‐tick font
    ax.set_ylabel("PPH", fontsize=16)                          # y‐axis label with bigger font
    ax.set_title(title, fontsize=18)                           # larger title
    ax.grid(True)

    # (Legend has been deliberately removed to hide the color‐code box)
    # If you ever want to re‐enable a legend, uncomment the next line:
    # ax.legend(loc="upper left", fontsize=12, frameon=True)

    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf
