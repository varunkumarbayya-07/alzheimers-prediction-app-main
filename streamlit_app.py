import os, sys, site

# ── Critical: Fix TF native DLL loading for Microsoft Store Python ────────────
try:
    _user_pkgs = site.getusersitepackages()
    if _user_pkgs not in sys.path:
        sys.path.insert(0, _user_pkgs)
    for _subdir in ["", "tensorflow", os.path.join("tensorflow", "python")]:
        _d = os.path.join(_user_pkgs, _subdir)
        if os.path.isdir(_d):
            os.environ["PATH"] = _d + os.pathsep + os.environ.get("PATH", "")
            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(_d)
                except Exception:
                    pass
except Exception:
    pass

import base64
import streamlit as st
from config import CSS, BACKGROUND, SIDE_BANNER
from streamlit_pages._home_page import home_page
from streamlit_pages._predict_alzheimer import prediction_page
from streamlit_pages._latest_news import news_page
from streamlit_pages._team_members import team_members
from streamlit_pages._chat_page import chat_bot
from streamlit_pages._model_accuracy import model_accuracy_page

# SETTING PAGE CONFIG
st.set_page_config(
    page_title="Alzheimer's Prediction System",
    page_icon=":brain:",
    layout="wide",
)

st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


def set_page_background(png_file):
    @st.cache_data()
    def get_base64_of_bin_file(bin_file):
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()

    try:
        bin_str = get_base64_of_bin_file(png_file)
        page_bg_img = f'''
            <style>
            .stApp {{
                background-image: url("data:image/png;base64,{bin_str}");
                background-size: cover;
                background-attachment: fixed;
                }}
            </style>
        '''
        st.markdown(page_bg_img, unsafe_allow_html=True)
    except FileNotFoundError:
        pass  # Background image missing; use default Streamlit background


set_page_background(BACKGROUND)

# STREAMLIT APP — Sidebar (circular image via HTML)
import base64 as _b64
try:
    with open(SIDE_BANNER, "rb") as _f:
        _img_b64 = _b64.b64encode(_f.read()).decode()
    st.sidebar.markdown(f"""
        <div style="text-align:center; padding: 10px 0 4px 0;">
            <img src="data:image/png;base64,{_img_b64}"
                 style="width:140px; height:140px; border-radius:50%;
                        object-fit:cover;
                        border: 2px solid rgba(139,92,246,0.6);
                        box-shadow: 0 0 20px rgba(139,92,246,0.5);" />
        </div>
    """, unsafe_allow_html=True)
except Exception:
    st.sidebar.markdown("<div style='text-align:center;font-size:3rem;'>🧠</div>",
                        unsafe_allow_html=True)

st.sidebar.title("Alzheimer's Prediction System")
app_mode = st.sidebar.selectbox(
    "Please navigate through the different sections of our website from here",
    ["Home", "Predict Alzheimer's", "Model Accuracy", "ChatBot", "Latest News", "Team Members"],
)

st.sidebar.write("""
# Disclaimer
The predictions provided by this system are for informational purposes only. Consult a healthcare professional for accurate diagnosis and advice.

# Contact
For inquiries, you can mail us [here](mailto:arpitsengar99@gmail.com).
""")


def main():
    if app_mode == "Home":
        home_page()
    elif app_mode == "Predict Alzheimer's":
        prediction_page()
    elif app_mode == "Model Accuracy":
        model_accuracy_page()
    elif app_mode == "ChatBot":
        chat_bot()
    elif app_mode == "Latest News":
        news_page()
    elif app_mode == "Team Members":
        team_members()


if __name__ == "__main__":
    main()
