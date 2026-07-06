import streamlit as st
from config import HF_EMAIL, HF_PASS, BASE_PROMPT


# ── Comprehensive offline knowledge base ──────────────────────────────────────
def _smart_response(prompt: str) -> str:
    p = prompt.lower().strip()

    # Greetings
    if any(w in p for w in ["hello", "hi", "hey", "good morning", "good evening", "howdy"]):
        return (
            "Hello! 👋 I'm your Alzheimer's AI assistant. I can help you with information about "
            "Alzheimer's disease, its symptoms, prevention strategies, treatment options, "
            "caregiving tips, and more. What would you like to know?"
        )

    # Symptoms
    if any(w in p for w in ["symptom", "sign", "what is alzheimer", "what are the", "how to know"]):
        return (
            "**Common Symptoms of Alzheimer's Disease:**\n\n"
            "🔹 **Early Stage:** Memory loss affecting daily life, difficulty planning or solving problems, "
            "confusion with time or place, trouble understanding visual images.\n\n"
            "🔹 **Middle Stage:** Increasing memory loss, difficulty recognizing family/friends, "
            "wandering, personality and behavioral changes, needing help with daily activities.\n\n"
            "🔹 **Late Stage:** Loss of ability to communicate verbally, complete dependence on others "
            "for care, vulnerability to infections.\n\n"
            "⚕️ If you notice these signs in yourself or a loved one, please consult a neurologist "
            "for proper evaluation."
        )

    # Causes / Risk factors
    if any(w in p for w in ["cause", "risk", "why", "reason", "apoe", "genetic", "gene", "heredit"]):
        return (
            "**Causes & Risk Factors of Alzheimer's Disease:**\n\n"
            "🔬 **Genetic factors:** The APOE-e4 gene variant is the strongest known genetic risk factor. "
            "Having one copy increases risk 2-3x; two copies increases it 8-12x.\n\n"
            "📅 **Age:** The greatest known risk factor. Risk doubles every 5 years after age 65.\n\n"
            "🏥 **Medical conditions:** Cardiovascular disease, diabetes, high blood pressure, and "
            "high cholesterol are all linked to higher risk.\n\n"
            "🧠 **Head trauma:** A history of significant head injuries may increase risk.\n\n"
            "👪 **Family history:** Having a parent or sibling with Alzheimer's increases your risk."
        )

    # Prevention
    if any(w in p for w in ["prevent", "avoid", "reduce risk", "protect", "lower risk", "stop"]):
        return (
            "**Ways to Reduce Your Alzheimer's Risk:**\n\n"
            "🏃 **Exercise regularly:** 150 min/week of aerobic exercise improves blood flow to the brain.\n\n"
            "🥗 **Eat a brain-healthy diet:** The Mediterranean or MIND diet has been linked to "
            "lower Alzheimer's risk.\n\n"
            "🧩 **Stay mentally active:** Reading, puzzles, learning new skills, and social engagement "
            "build cognitive reserve.\n\n"
            "😴 **Get quality sleep:** During sleep, the brain clears toxic proteins including amyloid plaques.\n\n"
            "🚭 **Avoid smoking and limit alcohol:** Both are modifiable risk factors.\n\n"
            "❤️ **Manage cardiovascular health:** Control blood pressure, cholesterol, and diabetes."
        )

    # Treatment / medication
    if any(w in p for w in ["treat", "cure", "medication", "drug", "medicine", "therapy", "lecanemab", "donepezil"]):
        return (
            "**Alzheimer's Treatment Options:**\n\n"
            "💊 **FDA-Approved Medications:**\n"
            "- *Cholinesterase inhibitors* (donepezil, rivastigmine, galantamine) — help with symptoms in early/middle stages\n"
            "- *Memantine* — helps with moderate to severe Alzheimer's\n"
            "- *Lecanemab (Leqembi)* — newer drug that slows progression in early stages by targeting amyloid plaques\n\n"
            "🧘 **Non-Drug Approaches:**\n"
            "- Cognitive stimulation therapy\n"
            "- Occupational therapy\n"
            "- Physical exercise programs\n"
            "- Music and art therapy\n\n"
            "⚠️ There is currently no cure, but early treatment can significantly slow progression."
        )

    # Stages
    if any(w in p for w in ["stage", "progression", "mild", "moderate", "severe", "early", "late"]):
        return (
            "**Stages of Alzheimer's Disease:**\n\n"
            "📌 **Stage 1 — Preclinical (No symptoms):** Brain changes occurring but no noticeable memory problems.\n\n"
            "📌 **Stage 2 — Mild Cognitive Impairment (MCI):** Slight but noticeable decline in memory or thinking. "
            "Not severe enough to interfere with daily life.\n\n"
            "📌 **Stage 3 — Mild Alzheimer's:** Mild memory loss, difficulty with complex tasks, "
            "may repeat questions. Can still live independently.\n\n"
            "📌 **Stage 4 — Moderate Alzheimer's:** Increasing memory loss, confusion, needs help "
            "with daily activities like dressing and bathing.\n\n"
            "📌 **Stage 5 — Severe Alzheimer's:** Loss of ability to communicate or care for oneself. "
            "Requires full-time care."
        )

    # Diagnosis / testing
    if any(w in p for w in ["diagnos", "test", "detect", "check", "mmse", "scan", "mri"]):
        return (
            "**Diagnosing Alzheimer's Disease:**\n\n"
            "🔍 **Clinical Assessment:** Doctors evaluate memory, thinking, and behavior through interviews "
            "and standardized tests like the MMSE (Mini-Mental State Examination).\n\n"
            "🧠 **Brain Imaging:** MRI and CT scans can rule out other causes and show brain shrinkage. "
            "PET scans can detect amyloid plaques and tau tangles.\n\n"
            "🧬 **Biomarker Tests:** Blood tests and cerebrospinal fluid (CSF) analysis can detect "
            "amyloid and tau proteins.\n\n"
            "🧬 **Genetic Testing:** APOE genotyping can indicate risk but is not used alone for diagnosis.\n\n"
            "This prediction system uses clinical data (age, MMSE, APOE genotype, etc.) to assess risk using "
            "a Logistic Regression model trained on ADNI data."
        )

    # Caregiver support
    if any(w in p for w in ["caregiv", "care for", "help family", "support", "nursing", "home care"]):
        return (
            "**Caregiving Tips for Alzheimer's Patients:**\n\n"
            "💙 **Create a safe environment:** Remove fall hazards, lock away medications, install grab bars.\n\n"
            "📅 **Establish routines:** Consistent daily schedules reduce confusion and anxiety.\n\n"
            "🗣️ **Communicate clearly:** Use simple sentences, speak slowly, and be patient.\n\n"
            "🧘 **Take care of yourself too:** Caregiver burnout is real — join a support group and take breaks.\n\n"
            "📞 **Resources:** Alzheimer's Association Helpline: 1-800-272-3900 (24/7)\n\n"
            "🏥 **Professional help:** Consider in-home care, adult day programs, or memory care facilities "
            "as the disease progresses."
        )

    # Statistics / prevalence
    if any(w in p for w in ["statistic", "how many", "prevalence", "worldwide", "global", "million"]):
        return (
            "**Alzheimer's Disease — Key Statistics:**\n\n"
            "📊 Over **55 million** people worldwide live with dementia, and Alzheimer's accounts for "
            "60-70% of cases.\n\n"
            "📊 In the US alone, approximately **6.9 million** Americans age 65+ are living with Alzheimer's.\n\n"
            "📊 Every **3 seconds**, someone in the world develops dementia.\n\n"
            "📊 Alzheimer's is the **7th leading cause of death** in the United States.\n\n"
            "📊 The global cost of dementia is estimated at over **$1.3 trillion** annually.\n\n"
            "📊 By 2050, the number of people with Alzheimer's could **triple** to 153 million worldwide."
        )

    # This prediction system
    if any(w in p for w in ["predict", "this app", "system", "model", "how does", "accuracy", "logistic"]):
        return (
            "**About This Prediction System:**\n\n"
            "🤖 This app uses a **Logistic Regression** model trained on data from the "
            "Alzheimer's Disease Neuroimaging Initiative (ADNI) dataset.\n\n"
            "📋 **Input features used:**\n"
            "- Age, Years of Education, MMSE Score\n"
            "- Race & Ethnicity\n"
            "- APOE4 Allele Type & Genotype\n"
            "- Imputed Genotype, Gender\n\n"
            "🎯 **Output classes:**\n"
            "- **CN** — Cognitively Normal\n"
            "- **LMCI** — Late Mild Cognitive Impairment\n"
            "- **AD** — Alzheimer's Disease\n\n"
            "⚠️ This tool is for educational and research purposes. It does not replace a medical diagnosis."
        )

    # Goodbye
    if any(w in p for w in ["bye", "goodbye", "thank", "thanks", "exit"]):
        return (
            "Thank you for using the Alzheimer's Prediction System! 🧠\n\n"
            "Remember: early detection saves lives. If you have concerns about yourself or a loved one, "
            "please consult a healthcare professional.\n\n"
            "Stay healthy! 💙"
        )

    # Default
    return (
        "That's an interesting question about Alzheimer's disease. Here's what I can help you with:\n\n"
        "🔹 **Symptoms** — What signs to look for\n"
        "🔹 **Causes & Risk Factors** — Genetic and lifestyle factors\n"
        "🔹 **Prevention** — How to reduce your risk\n"
        "🔹 **Treatment** — Available medications and therapies\n"
        "🔹 **Stages** — How the disease progresses\n"
        "🔹 **Diagnosis** — How doctors detect Alzheimer's\n"
        "🔹 **Caregiving** — Tips for caring for a loved one\n"
        "🔹 **Statistics** — Global prevalence data\n\n"
        "Try asking me about any of these topics!"
    )


def chat_bot():
    st.title("🤖 Alzheimer's Assistant ChatBot")
    st.write("Ask me anything about Alzheimer's disease — symptoms, prevention, treatment, and more.")
    st.write("---")

    # Initialise session state
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi! I'm your Alzheimer's AI assistant. How can I help you today? 🧠"}
        ]

    # Display all previous messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Accept new user input
    user_input = st.chat_input("Type your question here...")

    if user_input:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Keep original user text for the smart-response fallback
                original_input = user_input
                response = None

                # Try HuggingFace first if credentials are configured
                if HF_EMAIL and HF_PASS:
                    try:
                        from hugchat import hugchat
                        from hugchat.login import Login
                        sign = Login(HF_EMAIL, HF_PASS)
                        cookies = sign.login()
                        chatbot = hugchat.ChatBot(cookies=cookies.get_dict())
                        hf_query = user_input
                        if not st.session_state.get("_base_sent"):
                            st.session_state["_base_sent"] = True
                            hf_query = BASE_PROMPT + user_input  # prepend only for HF
                        response = str(chatbot.chat(hf_query)).strip("`")
                    except Exception:
                        response = None

                # Fall back to smart offline response using the ORIGINAL user text
                if not response:
                    response = _smart_response(original_input)

            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})
