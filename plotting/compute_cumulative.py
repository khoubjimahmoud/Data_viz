# plotting/compute_cumulative.py

import pandas as pd
import numpy as np

def compute_cumulative_pph(df, months_str, brands):
    """
    For each brand b (including "Opel + Vauxhall"), build a DataFrame with:
      [ Month | Problems | Sample | AccumulativeMeanPPH ].
    """
    key_model, key_brand, key_noise = df.columns[:3]
    out_tables = {}

    for b in brands:
        cum_p, cum_s = 0.0, 0.0
        rows = []

        for m in months_str:
            pph_col  = f"{m} PPH"
            samp_col = f"{m} Sample"

            # Helper: fetch (P, S) for given brand_name and noise_type
            def fetch_P_S(brand_name, noise_type):
                mask = (
                    (df[key_brand] == "Total (qualified)") &
                    (df[key_noise] == noise_type) &
                    (df[key_model] == brand_name)
                )
                if mask.any() and (pph_col in df.columns) and (samp_col in df.columns):
                    raw_P = df.loc[mask, pph_col].replace({'*': np.nan, '-': np.nan})
                    raw_S = df.loc[mask, samp_col].replace({'*': np.nan, '-': np.nan})
                    if not raw_P.empty:
                        P_val = pd.to_numeric(raw_P.iat[0], errors="coerce")
                        S_val = pd.to_numeric(raw_S.iat[0], errors="coerce")
                        return P_val, S_val
                return np.nan, np.nan

            if b != "Opel + Vauxhall":
                # 1) Brand-specific Body & Interior
                P_body, S_body = fetch_P_S(b, "Body noises")
                P_int,  S_int  = fetch_P_S(b, "Interior noises")

                # compute “Problems”
                prob_body = (P_body/100.0)*S_body if pd.notna(P_body) and pd.notna(S_body) else np.nan
                prob_int  = (P_int/100.0)*S_int if pd.notna(P_int) and pd.notna(S_int) else np.nan

                if pd.notna(prob_body) and pd.notna(prob_int):
                    total_prob = prob_body + prob_int
                    if S_body > 0:
                        total_samp = S_body
                    else:
                        total_samp = S_int
                elif pd.notna(prob_body):
                    total_prob = prob_body
                    total_samp = S_body
                elif pd.notna(prob_int):
                    total_prob = prob_int
                    total_samp = S_int
                else:
                    total_prob = total_samp = np.nan

            else:
                # 2) Opel
                P_body_op, S_body_op = fetch_P_S("Opel", "Body noises")
                P_int_op,  S_int_op  = fetch_P_S("Opel", "Interior noises")
                prob_body_op = (P_body_op/100.0)*S_body_op if pd.notna(P_body_op) and pd.notna(S_body_op) else np.nan
                prob_int_op  = (P_int_op/100.0)*S_int_op if pd.notna(P_int_op) and pd.notna(S_int_op) else np.nan

                if pd.notna(prob_body_op) and pd.notna(prob_int_op):
                    total_prob_op = prob_body_op + prob_int_op
                    if S_body_op > 0:
                        total_samp_op = S_body_op
                    else:
                        total_samp_op = S_int_op
                elif pd.notna(prob_body_op):
                    total_prob_op = prob_body_op
                    total_samp_op = S_body_op
                elif pd.notna(prob_int_op):
                    total_prob_op = prob_int_op
                    total_samp_op = S_int_op
                else:
                    total_prob_op = total_samp_op = np.nan

                # 3) Vauxhall
                P_body_vx, S_body_vx = fetch_P_S("Vauxhall", "Body noises")
                P_int_vx,  S_int_vx  = fetch_P_S("Vauxhall", "Interior noises")
                prob_body_vx = (P_body_vx/100.0)*S_body_vx if pd.notna(P_body_vx) and pd.notna(S_body_vx) else np.nan
                prob_int_vx  = (P_int_vx/100.0)*S_int_vx if pd.notna(P_int_vx) and pd.notna(S_int_vx) else np.nan

                if pd.notna(prob_body_vx) and pd.notna(prob_int_vx):
                    total_prob_vx = prob_body_vx + prob_int_vx
                    if S_body_vx > 0 :
                        total_samp_vx = S_body_vx
                    else:
                        total_samp_vx = S_int_vx
                elif pd.notna(prob_body_vx):
                    total_prob_vx = prob_body_vx
                    total_samp_vx = S_body_vx
                elif pd.notna(prob_int_vx):
                    total_prob_vx = prob_int_vx
                    total_samp_vx = S_int_vx
                else:
                    total_prob_vx = total_samp_vx = np.nan

                # 4) Combine Opel + Vauxhall
                if pd.notna(total_prob_op) and pd.notna(total_prob_vx) and \
                   pd.notna(total_samp_op) and pd.notna(total_samp_vx) and \
                   (total_samp_op + total_samp_vx) > 0:
                    total_prob = total_prob_op + total_prob_vx
                    total_samp  = total_samp_op + total_samp_vx
                else:
                    total_prob = total_samp = np.nan

            # 5) Accumulate running sums
            if pd.notna(total_prob) and pd.notna(total_samp) and total_samp > 0:
                cum_p += total_prob
                cum_s += total_samp
                acc_mean = (cum_p / cum_s) * 100.0
            else:
                acc_mean = np.nan

            rows.append({
                "Month": m,
                "Problems": total_prob,
                "Sample": total_samp,
                "AccumulativeMeanPPH": acc_mean
            })

        out_tables[b] = pd.DataFrame(rows)

    return out_tables
