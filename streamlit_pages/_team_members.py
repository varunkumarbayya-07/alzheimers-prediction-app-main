import streamlit as st
from config import TEAM_MEMBERS

def team_members():
    st.title("👥 Team Members")
    st.write("Meet the team behind the Alzheimer's Prediction System.")
    st.write("---")

    cols = st.columns(len(TEAM_MEMBERS))

    for col, member in zip(cols, TEAM_MEMBERS):
        with col:
            st.markdown(f"""
                <div style="
                    background: rgba(255,255,255,0.07);
                    border: 1px solid rgba(139,92,246,0.35);
                    border-radius: 14px;
                    padding: 22px 16px;
                    text-align: center;
                    backdrop-filter: blur(6px);
                    box-shadow: 0 4px 20px rgba(0,0,0,0.5);
                    height: 100%;
                ">
                    <div style="font-size:2.5rem; margin-bottom:10px;">🎓</div>
                    <h3 style="
                        font-size:1rem;
                        font-weight:700;
                        color:#e0d7ff;
                        margin-bottom:6px;
                    ">{member['name']}</h3>
                    <p style="
                        font-size:0.78rem;
                        color:#a78bfa;
                        font-weight:600;
                        margin-bottom:8px;
                    ">{member['role']}</p>
                    <p style="
                        font-size:0.72rem;
                        color:#c4b5fd;
                        margin-bottom:4px;
                        line-height:1.4;
                    ">{member.get('dept','')}</p>
                    <p style="
                        font-size:0.70rem;
                        color:#9ca3af;
                        margin-bottom:6px;
                        line-height:1.4;
                    ">{member.get('college','')}<br>{member.get('city','')}</p>
                    <a href="mailto:{member.get('email','')}" style="
                        font-size:0.68rem;
                        color:#818cf8;
                        text-decoration:none;
                        word-break:break-all;
                    ">✉ {member.get('email','')}</a>
                </div>
            """, unsafe_allow_html=True)
