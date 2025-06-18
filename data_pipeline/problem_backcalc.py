import pandas as pd
import numpy as np

def fill_missing_problems(df, gray_mask, months_str):
    """
    For each month in months_str:
      Problems = (PPH/100) * Sample
    but only where PPH>0, Sample exists, and PPH cell isn’t gray.
    """
    for m in months_str:
        sc, pc, qc = f"{m} Sample", f"{m} PPH", f"{m} Problems"

        df[sc] = (df[sc]
                  .replace({'*':np.nan,'-':np.nan})
                  .pipe(pd.to_numeric, errors="coerce"))

        if pc in df.columns:
            raw_pph = df[pc].replace({'*':np.nan,'-':np.nan})
            pnum    = pd.to_numeric(raw_pph, errors="coerce")
        else:
            pnum = pd.Series(np.nan, index=df.index)

        valid = (pnum > 0) & (~df[sc].isna())
        if pc in gray_mask.columns:
            valid &= ~gray_mask[pc]

        df.loc[valid, qc] = (pnum[valid]/100.0) * df.loc[valid, sc]

    return df
