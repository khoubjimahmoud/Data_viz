from io import BytesIO
import pandas as pd

def to_excel(df):
    """Sink DataFrame to in‐memory Excel bytes."""
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
        df.to_excel(w, index=False, sheet_name="Processed Data")
    return buf.getvalue()
