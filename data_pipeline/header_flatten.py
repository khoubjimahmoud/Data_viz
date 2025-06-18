import pandas as pd
import numpy as np

def process_excel(file):
    """
    1) Locate the two header rows (month names & sublabels)
    2) Flatten into single-row cols: [Model, Brand, Noise] + "{Mon-YYYY} {Suffix}"
    3) Return (df, months_str)
    """
    raw = pd.read_excel(file, header=None, engine="openpyxl")

    # find subheader row by “Sample” in col4
    submask = raw[3].astype(str).str.strip().eq("Sample")
    if not submask.any():
        raise ValueError("Couldn’t find 'Sample' in column 4")
    sub_i = submask.idxmax()
    mon_i = sub_i - 1

    hdr_mon = raw.iloc[mon_i].ffill().astype(str)
    hdr_sub = raw.iloc[sub_i].astype(str)
    data_r  = raw.iloc[sub_i+1:].reset_index(drop=True)

    valid = {"Sample","Vehicles","Problems","PPH"}
    keep, cols = [], []
    for i, lbl in enumerate(hdr_sub):
        if i < 3:
            keep.append(i); cols.append(lbl.strip())
        else:
            suf = lbl.strip()
            if suf not in valid:
                nxt = str(raw.iloc[sub_i+1, i]).strip()
                if nxt in valid:
                    suf = nxt
                else:
                    continue
            mon = hdr_mon[i].strip()
            keep.append(i); cols.append(f"{mon} {suf}")

    df = data_r.iloc[:, keep].copy()
    df.columns = cols

    # extract & sort months
    months = [c.rsplit(" ",1)[0] for c in cols[3:]]
    mon_dt = pd.to_datetime(months, format="%b-%Y", errors="coerce")
    months_str = [
        d.strftime("%b-%Y")
        for d in sorted(set(mon_dt.dropna()))
    ]

    # ensure uniform four suffix cols per month
    for m in months_str:
        for suf in valid:
            col = f"{m} {suf}"
            if col not in df.columns:
                df[col] = np.nan

    return df, months_str

