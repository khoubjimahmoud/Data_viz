# plotting/plot_brand_filtered.py

import pandas as pd
import numpy as np
from io import BytesIO
from plotting.plot_brand import plot_pph_df

def compute_filtered_raw_pph(df, months_str, sel_brands, sel_noises):
    tbl = pd.DataFrame(index=months_str, columns=sel_brands, dtype=float)
    for m in months_str:
        for b in sel_brands:
            tot_p = tot_s = 0.0
            bad = False
            for nt in sel_noises:
                sub = df[
                    (df['Brand']=='Total (qualified)') &
                    (df['Vehicle Model-Body-Trim']==b) &
                    (df['Noise Typology']==nt)
                ]
                if sub.empty:
                    bad = True; break
                P = pd.to_numeric(sub[f"{m} Problems"].iat[0], errors='coerce')
                S = pd.to_numeric(sub[f"{m} Sample"].iat[0],   errors='coerce')
                if pd.isna(P) or pd.isna(S) or S==0:
                    bad = True; break
                tot_p += P
                tot_s += S
            tbl.at[m, b] = (tot_p/tot_s*100) if not bad else np.nan

    # Opel+Vauxhall combine
    opl = []
    for m in months_str:
        tot_p = tot_s = 0.0
        bad = False
        for nm in ('Opel','Vauxhall'):
            for nt in sel_noises:
                sub = df[
                    (df['Brand']=='Total (qualified)') &
                    (df['Vehicle Model-Body-Trim']==nm) &
                    (df['Noise Typology']==nt)
                ]
                if sub.empty:
                    bad=True; break
                P = pd.to_numeric(sub[f"{m} Problems"].iat[0], errors='coerce')
                S = pd.to_numeric(sub[f"{m} Sample"].iat[0],   errors='coerce')
                if pd.isna(P) or pd.isna(S) or S==0:
                    bad=True; break
                tot_p += P
                tot_s += S
            if bad: break
        opl.append((tot_p/tot_s*100) if not bad else np.nan)
    tbl['Opel + Vauxhall'] = opl

    return tbl

def compute_filtered_cumulative_pph(df, months_str, sel_brands, sel_noises):
    key_model, key_brand, key_noise = df.columns[:3]
    out = {}
    first_noise = sel_noises[0]

    for b in sel_brands:
        cum_p = cum_s = 0.0
        rows = []
        for m in months_str:
            # collect Problems & Samples
            probs = []
            samp_first = np.nan
            bad = False
            for nt in sel_noises:
                mask = (
                  (df[key_brand]=='Total (qualified)') &
                  (df[key_noise]==nt) &
                  (df[key_model]==b)
                )
                if not mask.any():
                    bad = True; break
                raw_P = df.loc[mask, f"{m} PPH"].replace({'*':np.nan,'-':np.nan})
                raw_S = df.loc[mask, f"{m} Sample"].replace({'*':np.nan,'-':np.nan})
                P = pd.to_numeric(raw_P.iat[0], errors='coerce')
                S = pd.to_numeric(raw_S.iat[0], errors='coerce')
                if pd.isna(P) or pd.isna(S) or S==0:
                    bad=True; break
                # convert % → counts
                tot_prob = P/100.0 * S
                probs.append(tot_prob)
                if nt == first_noise:
                    samp_first = S

            if bad or pd.isna(samp_first):
                rows.append({"Month":m, "Problems":np.nan, "Sample":np.nan, "AccumulativeMeanPPH":np.nan})
            else:
                total_prob = sum(probs)
                cum_p += total_prob
                cum_s += samp_first
                acc = (cum_p/cum_s)*100 if cum_s>0 else np.nan
                rows.append({"Month":m, "Problems":total_prob, "Sample":samp_first, "AccumulativeMeanPPH":acc})

        out[b] = pd.DataFrame(rows)

    return out

def plot_pph_values_filtered_df(df, months_str, sel_brands, sel_noises):
    # build raw & cumulative tables
    raw_tbl = compute_filtered_raw_pph(df, months_str, sel_brands, sel_noises)
    cum_tables = compute_filtered_cumulative_pph(df, months_str, sel_brands, sel_noises)

    # assemble cumulative matrix
    cum_tbl = pd.DataFrame(index=months_str, columns=sel_brands, dtype=float)
    for b in sel_brands:
        for _, r in cum_tables[b].iterrows():
            cum_tbl.at[r["Month"], b] = r["AccumulativeMeanPPH"]

    # plot with identical styling
    buf_raw = plot_pph_df(raw_tbl, "Filtered Raw PPH by Brand")
    buf_cum = plot_pph_df(cum_tbl, "Filtered Cumulative Mean PPH by Brand")
    return buf_raw, buf_cum
