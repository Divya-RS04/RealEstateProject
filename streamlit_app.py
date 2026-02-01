import streamlit as st
import pandas as pd
import joblib

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Real Estate Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# ------------------ CUSTOM CSS ------------------
st.markdown("""
<style>
.main {
    background: linear-gradient(135deg, #f5f7fa, #c3cfe2);
}

.block-container {
    padding-top: 2rem;
}

.card {
    background: white;
    padding: 2rem;
    border-radius: 18px;
    box-shadow: 0px 10px 25px rgba(0,0,0,0.08);
}

.result-card {
    background: linear-gradient(135deg, #43cea2, #185a9d);
    padding: 1.8rem;
    border-radius: 16px;
    color: white;
    text-align: center;
    font-size: 1.3rem;
    font-weight: 600;
}

.small-text {
    color: #6c757d;
    font-size: 0.95rem;
}
</style>
""", unsafe_allow_html=True)

# ------------------ LOAD MODEL ------------------
model = joblib.load("real_estate_price_model.pkl")

# ------------------ HEADER ------------------
st.markdown("<h1 style='text-align:center;'>🏠 Real Estate Price Prediction</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;' class='small-text'>End-to-End Machine Learning Application with a Modern UI</p>",
    unsafe_allow_html=True
)

st.markdown("<br>", unsafe_allow_html=True)

# ------------------ MAIN LAYOUT ------------------
left, right = st.columns([1.2, 1])

# ------------------ INPUT CARD ------------------
with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📋 Property Details")

    col1, col2 = st.columns(2)
    with col1:
        mrt = st.number_input("🚇 Distance to MRT (meters)", min_value=0.0, step=10.0)
        lat = st.number_input("🌍 Latitude", format="%.6f")
    with col2:
        stores = st.number_input("🏪 Convenience Stores", min_value=0, step=1)
        lon = st.number_input("🌍 Longitude", format="%.6f")

    st.markdown("<br>", unsafe_allow_html=True)

    predict_btn = st.button("🔍 Predict House Price", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ------------------ RESULT SECTION ------------------
with right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📈 Prediction Result")

    if predict_btn:
        df = pd.DataFrame([[mrt, stores, lat, lon]],
                          columns=[
                              "Distance to the nearest MRT station",
                              "Number of convenience stores",
                              "Latitude",
                              "Longitude"
                          ])

        price = model.predict(df)[0]

        st.markdown(
            f"<div class='result-card'>🏡 Estimated Price<br>{price:.2f} (unit area)</div>",
            unsafe_allow_html=True
        )
    else:
        st.info("Enter details and click **Predict House Price**")

    st.markdown("</div>", unsafe_allow_html=True)

# ------------------ FOOTER ------------------
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;' class='small-text'>Built using Streamlit & Machine Learning</p>",
    unsafe_allow_html=True
)



