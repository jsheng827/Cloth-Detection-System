import os
import sys
import base64
import streamlit as st
import pandas as pd
from datetime import datetime, date, timedelta

# Make sure we can import from the existing `py/` modules
# pages/2_Audit_Log.py -> frontend/ -> project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
PY_DIR = os.path.join(BASE_DIR, "py")
if PY_DIR not in sys.path:
    sys.path.append(PY_DIR)

from db import evaluations, violations  # type: ignore

st.set_page_config(page_title="Audit Log", layout="wide")

st.markdown(
    """
    <style>
        .audit-root {
            background-color: #0f0f0f;
            color: #f5f5f5;
            padding: 1rem 2rem;
        }
        .filter-card {
            background: #1e1e1e;
            border-radius: 20px;
            padding: 1.25rem;
            border: 1px solid #2c2c2c;
            margin-bottom: 1rem;
        }
        .table-card {
            background: #181818;
            border-radius: 16px;
            padding: 1.5rem;
            border: 1px solid #272727;
        }
        .detail-card {
            background: #1b1b1b;
            border-radius: 12px;
            padding: 1.25rem;
            border: 1px solid #303030;
        }
        body {
            background-color: #0f0f0f;
            color: #f5f5f5;
        }
        .stApp {
            background-color: #0f0f0f;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Audit Log")
st.caption("Filter detections by range, status, and sort order.")

search = st.text_input("Search by evaluation id or status")

cursor = evaluations.find().sort("datetime", -1)

records = []
for doc in cursor:
    eval_id = doc.get("evaluation_id") or str(doc.get("_id"))
    violation = violations.find_one({"evaluation_id": eval_id})
    has_violation = bool(violation)
    detail_cell = "..." if has_violation else (doc.get("details") or "-")
    records.append(
        {
            "Evaluation ID": eval_id,
            "Clothing Category": doc.get("clothing_category", "-"),
            "Evaluation Status": doc.get("status", "-"),
            "Evaluation Datetime": doc.get("datetime"),
            "Has Violation": "Yes" if has_violation else "No",
            "Snapshot": violation.get("image") if violation else None,
            "Violation Type": violation.get("violation_type") if violation else "-",
            "Violation Description": violation.get("description") if violation else "",
        }
    )

df = pd.DataFrame(records)

if not df.empty:
    # Prepare a datetime column for filtering.
    def to_dt(value):
        if not value:
            return None
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            try:
                return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                return None

    df["DateObj"] = df["Evaluation Datetime"].apply(to_dt)

default_from = date.today() - timedelta(days=7)
default_to = date.today()
status_options = ["Appropriate", "Not Appropriate"]

col_filter, col_table = st.columns([1.1, 2.5])

with col_filter:
    st.markdown('<div class="filter-card">', unsafe_allow_html=True)
    st.subheader("Filter by")
    from_date = st.date_input("From", value=default_from)
    to_date = st.date_input("To", value=default_to)
    selected_status = st.multiselect(
        "Evaluation Status",
        options=status_options,
        default=status_options,
    )
    sort_choice = st.selectbox(
        "Sort by",
        options=["Newest to Oldest", "Oldest to Newest"],
        index=0,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with col_table:
    if not df.empty:
        mask = pd.Series(True, index=df.index)
        if search:
            query = search.lower()
            mask &= df.apply(
                lambda row: query in " ".join(map(str, row.values)).lower(), axis=1
            )

        if selected_status:
            mask &= df["Evaluation Status"].isin(selected_status)

        if df["DateObj"].notna().any():
            from_dt = datetime.combine(from_date, datetime.min.time())
            to_dt = datetime.combine(to_date, datetime.max.time())
            mask &= df["DateObj"].between(from_dt, to_dt, inclusive="both")

        df_filtered = df[mask].copy()

        ascending = sort_choice == "Oldest to Newest"
        if df_filtered["DateObj"].notna().any():
            df_filtered = df_filtered.sort_values("DateObj", ascending=ascending)

        st.markdown('<div class="table-card">', unsafe_allow_html=True)
        if df_filtered.empty:
            st.info("No evaluations for the selected filters.")
        else:
            st.dataframe(
                df_filtered[
                    [
                        "Evaluation ID",
                        "Clothing Category",
                        "Evaluation Status",
                        "Evaluation Datetime",
                    ]
                ],
                width="stretch",
                hide_index=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("No evaluations yet.")

st.markdown("### Violation List")

violations_cursor = violations.find().sort("violation_id", -1)
violation_rows = []
for doc in violations_cursor:
    vio_id = doc.get("violation_id") or doc.get("_id")
    if vio_id is not None:
        vio_id = str(vio_id)
    else:
        vio_id = "-"
    violation_rows.append(
        {
            "Violation ID": vio_id,
            "Evaluation ID": doc.get("evaluation_id"),
            "Evaluation Status": doc.get("evaluation_status", "Not Appropriate"),
            "Violation Types": doc.get("violation_type"),
            "Violation Description": doc.get("description"),
            "Violation Image": "View",
            "Snapshot": doc.get("image"),
        }
    )

if violation_rows:
    vio_df = pd.DataFrame(violation_rows)
    vio_df["Violation ID"] = vio_df["Violation ID"].astype(str)
    search_vio = st.text_input("Search violation by ID or keyword").strip()

    filtered_vio = vio_df
    if search_vio:
        q = search_vio.lower()
        mask_vio = vio_df.apply(
            lambda row: q in " ".join(map(str, row.values)).lower(), axis=1
        )
        filtered_vio = vio_df[mask_vio]

    st.dataframe(
        vio_df[
            [
                "Violation ID",
                "Evaluation ID",
                "Evaluation Status",
                "Violation Types",
                "Violation Description",
            ]
        ],
        width="stretch",
        hide_index=True,
    )

    if not filtered_vio.empty:
        # If user typed an exact ID (case-insensitive), prefer that row; otherwise take first match.
        detail_source = filtered_vio if search_vio else vio_df
        if search_vio:
            normalized_ids = detail_source["Violation ID"].str.lower()
            matches = detail_source[normalized_ids == search_vio.lower()]
            if not matches.empty:
                vio_row = matches.iloc[0]
            else:
                vio_row = detail_source.iloc[0]
        else:
            vio_row = detail_source.iloc[0]

        st.markdown('<div class="detail-card">', unsafe_allow_html=True)
        st.subheader(f"Violation {vio_row['Violation ID']}")
        st.write(f"Evaluation ID: {vio_row['Evaluation ID']}")
        st.write(f"Status: **{vio_row['Evaluation Status']}**")
        st.write(f"Type: {vio_row['Violation Types']}")
        st.write(vio_row["Violation Description"])
        if vio_row["Snapshot"]:
            img_bytes = base64.b64decode(vio_row["Snapshot"])
            st.image(img_bytes, caption="Violation image")
        else:
            st.caption("No image stored for this violation.")
        st.markdown("</div>", unsafe_allow_html=True)
else:
    st.info("No violations recorded yet.")