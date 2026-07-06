import streamlit as st

def home_page():
    # Styled hero header — matches the original's big serif all-caps title look
    st.markdown("""
        <div style="
            background: rgba(10, 5, 30, 0.55);
            border: 1px solid rgba(139, 92, 246, 0.35);
            border-radius: 12px;
            padding: 60px 40px;
            text-align: center;
            margin-bottom: 30px;
            backdrop-filter: blur(6px);
            box-shadow: 0 8px 32px rgba(0,0,0,0.5);
        ">
            <h1 style="
                font-family: 'Georgia', 'Times New Roman', serif;
                font-size: 3.5rem;
                font-weight: 700;
                letter-spacing: 6px;
                text-transform: uppercase;
                color: #ffffff;
                text-shadow: 2px 2px 8px rgba(0,0,0,0.4);
                margin: 0;
                line-height: 1.2;
            ">
                ALZHEIMER'S<br>PREDICTION SYSTEM
            </h1>
        </div>
    """, unsafe_allow_html=True)

    st.write("""
        ## Introduction to Alzheimer's Disease
        Alzheimer's disease (AD) is a progressive neurodegenerative disease. Though best known for its role in declining memory function, symptoms also include: difficulty thinking and reasoning, making judgements and decisions, and planning and performing familiar tasks. It may also cause alterations in personality and behavior. The cause of AD is not well understood. There is thought to be a significant hereditary component. For example, a variation of the APOE gene, APOE e4, increases risk of Alzheimer's disease.

        ## Why Early Detection Matters
        Early detection of Alzheimer's disease is paramount because it offers the best chance for effective treatment and improved quality of life. Identifying the condition at its onset allows for timely interventions, which can slow its progression and enable individuals and their families to plan for the future. Early detection also facilitates access to support services and clinical trials, fostering hope for more effective therapies in the fight against this devastating disease.

        ## Purpose of the project
        The purpose of this project proposal is to develop a machine learning model for the early prediction of Alzheimer's disease. Alzheimer's disease is a devastating neurodegenerative disorder that affects millions of individuals worldwide. Early detection is crucial for better patient care and the development of potential interventions. This project aims to leverage machine learning techniques to create a predictive model that can identify individuals at risk of Alzheimer's disease based on relevant data.

        <br>

    """, unsafe_allow_html=True)

    st.caption("Finished reading? Navigate to the `Prediction Page` to make some predictions")
