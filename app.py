# File: app.py

import streamlit as st
import pandas as pd
import re
from io import BytesIO
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from data_pipeline.io_helpers       import read_with_style
from data_pipeline.header_flatten   import process_excel
from data_pipeline.problem_backcalc import fill_missing_problems
from data_pipeline.excel_export     import to_excel

from plotting.plot_brand            import compute_raw_pph, plot_pph_df
from plotting.plot_model            import plot_model_pph_last12_df
from plotting.plot_brand_filtered import plot_pph_values_filtered_df
from plotting.compute_cumulative    import compute_cumulative_pph
from plotting.plot_model_filtered import plot_model_pph_filtered_df

st.title("Excel File Processor")

uploaded = st.file_uploader("Upload an Excel file", type=["xlsx"])
if not uploaded:
    st.stop()

data_bytes = uploaded.read()

# ─── Data I/O & Processing ──────────────────────────────────────────────────

# 1) Read raw data + gray‐mask
raw_df, gray_mask = read_with_style(BytesIO(data_bytes))

# 2) Flatten headers and extract months_str
df, months_str = process_excel(BytesIO(data_bytes))

# 3) Back‐calculate any missing “Problems” columns
df = fill_missing_problems(df, gray_mask, months_str)

# Ensure all column names are strings
df.columns = df.columns.map(str)

# ─── Display & Download Processed Data ──────────────────────────────────────

st.subheader("Processed Data (with back‐calculated Problems)")

# build an ordered list of columns: for each month, show Sample, PPH, then Problems
cols = []
for m in months_str:
    for suf in ("Sample", "PPH", "Problems"):
        col = f"{m} {suf}"
        if col in df.columns:
            cols.append(col)

# now display only those, so the back‐calculated Problems are right there
st.dataframe(df[cols].astype(str))

out = to_excel(df)
st.download_button(
    "Download Processed Excel",
    out,
    "Processed_with_Problems.xlsx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# ─── Plot 1: Raw PPH & Cumulative Mean PPH by Brand ──────────────────────────

with st.expander("▶️ Plot 1: Global Body noises by Brand & Cumulative Mean"):
    all_brands = [
        b for b in df["Brand"].unique()
        if pd.notna(b) and b not in ["Total (qualified)", "Brand"]
    ]
    if "Opel + Vauxhall" not in all_brands:
        all_brands.append("Opel + Vauxhall")

    sel_br1 = st.multiselect("Select Brands", all_brands, default=all_brands)

    last12 = months_str[-12:]
    st.markdown(f"🔢 Displaying brands over months **{last12[0]}** → **{last12[-1]}**")
    
    raw_tbl  = compute_raw_pph(df, last12, sel_br1)

    cum_tables = compute_cumulative_pph(df, last12, sel_br1)
    cum_tbl = pd.DataFrame(index=last12, columns=sel_br1, dtype=float)
    for b in sel_br1:
        dfb = cum_tables[b]
        for _, row in dfb.iterrows():
            cum_tbl.at[row["Month"], b] = row["AccumulativeMeanPPH"]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Raw PPH by Brand**")
        buf1 = plot_pph_df(raw_tbl, "Raw PPH by Brand")
        st.image(buf1, use_container_width=True, caption="Raw PPH by Brand")

    with col2:
        st.markdown("**Cumulative Mean PPH by Brand**")
        buf_cum = plot_pph_df(cum_tbl, "Cumulative Mean PPH by Brand")
        st.image(buf_cum, use_container_width=True, caption="Cumulative Mean PPH by Brand")

    st.subheader("🔍 Cumulative Mean PPH Details")
    for b in sel_br1:
        st.markdown(f"**{b}**")
        dfb = cum_tables[b].copy()
        dfb["Problems"] = dfb["Problems"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
        dfb["Sample"]   = dfb["Sample"].map(lambda x: f"{x:.0f}" if pd.notna(x) else "")
        dfb["AccumulativeMeanPPH"] = dfb["AccumulativeMeanPPH"].map(
            lambda x: f"{x:.2f}" if pd.notna(x) else ""
        )
        st.dataframe(dfb)

# ─── Plot 2: Combined Body + Interior PPH Over Last 12 Months ─────────────────

with st.expander("▶️ Plot 2: Combined Body + Interior PPH Over Last 12 Months"):
    key_model = "Vehicle Model-Body-Trim"

    # 2a) Identify every “Vehicle Model‐Body‐Trim” that contains at least one digit
    all_models = (
        df[key_model]
          .dropna()
          .astype(str)
          .unique()
          .tolist()
    )
    valid_models = [m for m in all_models if re.search(r"\d", m)]
    valid_models.sort()

    # 2b) Compute a model→brand mapping (excluding “Total (qualified)”)
    model_to_brand = {}
    for mdl in valid_models:
        brands_for_mdl = (
            df.loc[
                (df[key_model] == mdl) &
                (df["Brand"] != "Total (qualified)"),
                "Brand"
            ]
            .dropna()
            .unique()
            .tolist()
        )
        if set(brands_for_mdl) == {"Opel", "Vauxhall"}:
            model_to_brand[mdl] = "Opel/Vauxhall"
        elif len(brands_for_mdl) >= 1:
            model_to_brand[mdl] = brands_for_mdl[0]
        else:
            model_to_brand[mdl] = "Unknown"

    # 2c) Collect the distinct set of real brands among valid_models
    all_real_brands = sorted(set(model_to_brand[m] for m in valid_models))

    # 2d) Render a checkbox for each brand so the user can toggle on/off
    st.markdown("**Filter by Brand →**")
    cols_per_row = 4
    brand_checkers = {}
    row = None
    for i, b in enumerate(all_real_brands):
        if i % cols_per_row == 0:
            row = st.columns(cols_per_row)
        col_idx = i % cols_per_row
        brand_checkers[b] = row[col_idx].checkbox(b, value=True)

    selected_brands_for_model = [
        b for b, sel in brand_checkers.items() if sel
    ]

    # 2e) Narrow down valid_models to those whose brand is checked
    models_after_brand_filter = [
        m for m in valid_models
        if model_to_brand[m] in selected_brands_for_model
    ]

    # 2f) Let the user further pick individual models
    st.markdown("**Then select specific models (optional)**")
    sel_models = st.multiselect(
        "Choose Models – only from the brands you checked above",
        models_after_brand_filter,
        default=models_after_brand_filter
    )

    # 2g) Extract the final 12 months
    last12 = months_str[-12:]
    st.markdown(f"🔢 Using months: **{last12[0]}** – **{last12[-1]}**")

    # 2h) Call our cleaned‐up plotting routine
    result = plot_model_pph_last12_df(df, last12, sel_models=sel_models)
    if result is not None:
        buf2, df_res = result

        # 2i) Display the chart
        st.image(
            buf2,
            use_container_width=True,
            caption="Last 12-Month Combined Body + Interior PPH by Model"
        )

# ─── Plot 3: Filtered PPH by Brand & Period ───────────────────────────────────
with st.expander("▶️ Plot 3: Filtered PPH by Brand & Period"):
    noises = [
        'Body noises','Brakes noises','Chassis noises','Engine noises',
        'HVAC noises','Interior noises','Other noises','Wind noises'
    ]
    sel_noises = st.multiselect("Select Noise Types", noises, default=noises)

    start = st.selectbox("From", months_str, index=0)
    end   = st.selectbox("To",   months_str, index=len(months_str)-1)
    i0, i1 = months_str.index(start), months_str.index(end)
    months_sel = months_str[i0:i1+1]

    # now returns two buffers
    buf_raw, buf_cum = plot_pph_values_filtered_df(
        df, months_sel, sel_br1, sel_noises
    )

    st.image(
        buf_raw,
        use_container_width=True,
        caption=f"Filtered Raw PPH for {sel_noises} from {start} to {end}"
    )
    st.image(
        buf_cum,
        use_container_width=True,
        caption=f"Filtered Cumulative Mean PPH for {sel_noises} from {start} to {end}"
    )

# ─── Plot 4: Filtered PPH by Model & Period ───────────────────────────────────


with st.expander("▶️ Plot 4: Combined PPH by Model & Noise/Period Filter"):
    noises = [
        'Body noises','Brakes noises','Chassis noises','Engine noises',
        'HVAC noises','Interior noises','Other noises','Wind noises'
    ]
    # add a key here
    sel_noises = st.multiselect(
        "Select Noise Types",
        noises,
        default=noises,
        key="plot4_sel_noises"
    )

    # give unique keys to these selectboxes
    start = st.selectbox(
        "From",
        months_str,
        index=0,
        key="plot4_start_month"
    )
    end = st.selectbox(
        "To",
        months_str,
        index=len(months_str)-1,
        key="plot4_end_month"
    )
    i0, i1 = months_str.index(start), months_str.index(end)
    months_sel = months_str[i0:i1+1]

    # model multiselect also needs its own key
    sel_models = st.multiselect(
        "Select Models",
        valid_models,
        default=valid_models,
        key="plot4_sel_models"
    )

    result = plot_model_pph_filtered_df(df, months_sel, sel_models, sel_noises)
    if result is not None:
        buf4, df_model_res = result
        st.image(
            buf4,
            use_container_width=True,
            caption=f"Combined PPH for {sel_noises} from {start} to {end}"
        )
        st.dataframe(df_model_res)

# ─── Summary PDF ─────────────────────────────────────────────────────────────

summary = BytesIO()
with PdfPages(summary) as pdf:
    for img in (buf1, buf2, buf_raw, buf_cum, buf4):
        if img:
            fig = plt.figure(figsize=(15,8))
            plt.imshow(plt.imread(img))
            plt.axis("off")
            pdf.savefig(fig)
            plt.close(fig)
summary.seek(0)
st.download_button("Download Summary PDF", summary, "Summary.pdf", "application/pdf")
