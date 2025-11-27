import streamlit as st

st.set_page_config(
page_title="Real-Time Cloth Detection System",
layout="wide",
page_icon="🧥",
)

st.markdown(
    """
    <style>
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #101010;
        height: 100%;
    }

    [data-testid="stAppViewContainer"] > .main {
        min-height: 100vh;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    /* Hide sidebar on landing page */
    [data-testid="stSidebar"] {
        display: none;
    }

    .hero-wrapper {
        text-align: center;
        color: #f5f5f7;
        width: 100%;
        max-width: 900px;
        margin: 0 auto;
    }

    .hero-title {
        font-size: 3rem;
        font-weight: 600;
        margin-top: 1rem;
    }

    .hero-subtitle {
        opacity: 0.8;
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }

    .start-btn {
        display: flex;
        justify-content: center;
        margin-top: 2rem;
    }

    .start-btn button {
        background-color: #1f8a3f !important;
        color: #fff !important;
        border-radius: 8px;
        padding: 0.9rem 3rem;
        font-size: 1.1rem;
        border: none;
    }

    .start-btn button:hover {
        background-color: #32a852 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.container():
    st.markdown('<div class="hero-wrapper">', unsafe_allow_html=True)
    st.image("logo/logo2.jpg", width=320)
    st.markdown(
    '<div class="hero-title">Real-Time Cloth Detection System</div>',
    unsafe_allow_html=True,
    )
    st.markdown(
    '<p class="hero-subtitle">Monitor dress code compliance straight from your browser.</p>',
    unsafe_allow_html=True,
    )
    st.markdown('<div class="start-btn">', unsafe_allow_html=True)
    if st.button("Start ▶", key="start_btn", help="Go to live dashboard"):
        st.switch_page("pages/1_Real_Time_Dashboard.py")
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)