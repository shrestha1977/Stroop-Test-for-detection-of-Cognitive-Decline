import streamlit as st
import random
import time
import pandas as pd
import pickle
import numpy as np

# Load ML model and scaler
model = pickle.load(open("dementia_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# --- STROOP TEST WORDS ---
colors = ["RED", "BLUE", "GREEN", "YELLOW", "PURPLE"]
actual_colors = ["red", "blue", "green", "yellow", "purple"]

def run_stroop_test(num_questions=10):
    st.write("### Stroop Test")

    correct = 0
    wrong = 0
    reaction_times = []

    start_button = st.button("Start Test")

    if start_button:
        st.write("**The test has started! Identify the COLOR of the word (not the word itself).**")
        st.write("---")

        for i in range(num_questions):
            word = random.choice(colors)
            color = random.choice(actual_colors)

            st.write(f"### Word: **: {word}**")
            st.write(f"### Color shown: (the text will appear in {color})")

            start_time = time.time()
            user_answer = st.radio(
                "Choose the color:",
                actual_colors,
                key=f"q{i}"
            )
            confirm = st.button(f"Submit Q{i}")

            if confirm:
                end_time = time.time()
                reaction_time = end_time - start_time
                reaction_times.append(reaction_time)

                if user_answer == color:
                    correct += 1
                else:
                    wrong += 1

                st.write("---")

        avg_rt = sum(reaction_times) / len(reaction_times)
        stroop_score = correct * 4 - wrong * 2

        return avg_rt, correct, wrong, stroop_score

    return None

# ----------- STREAMLIT UI ---------------
st.title("🧠 Dementia Detection Using Stroop Test (ML Powered)")
st.write("This website uses a trained ML model + Stroop Test performance to estimate early dementia risk.")

age = st.number_input("Enter your age", 40, 90)

result = run_stroop_test(num_questions=5)

if result:
    avg_rt, correct, wrong, stroop_score = result

    st.write("### Your Stroop Test Summary")
    st.write(f"**Average Reaction Time:** {avg_rt:.2f} sec")
    st.write(f"**Correct Answers:** {correct}")
    st.write(f"**Wrong Answers:** {wrong}")
    st.write(f"**Stroop Score:** {stroop_score}")

    # ML prediction
    user_data = np.array([[age, avg_rt, correct, wrong, stroop_score]])
    user_data_scaled = scaler.transform(user_data)
    pred = model.predict(user_data_scaled)[0]

    st.write("---")
    st.write("## 🧠 Dementia Risk Result")

    if pred == 1:
        st.error("⚠ High probability of Cognitive Impairment / Dementia")
    else:
        st.success("✅ Low probability of Dementia")
