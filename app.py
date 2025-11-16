import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pickle
import re

# Load the trained model
with open('trained_spam_model1.sav', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the vectorizer
with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Text cleaner to replicate VS Code behavior
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    return text.strip()

# Streamlit UI
st.set_page_config(page_title="Spam Mail Detector", layout="centered")
# Sidebar
# Sidebar
st.sidebar.title("📩 Spam Mail Detector")

st.sidebar.markdown("""
---
                    
🔍 **Model Used:**  
Multinomial Naive Bayes

📊 **Accuracy:**  
96%

🗂 **Trained on:**  
Public spam email dataset
                    
💻 **Github:**  
[Himanshu-771]()

🔗 **LinkedIn:**  
[Himanshu-LinkedIn]()
                    
✉️ **Contact:** 
himanshusharma@example.com
                    
---

👨‍💻 **Developed by:**  
Himanshu Sharma

---
""")



st.title("📩 Spam Mail Detector")
st.markdown("Enter your email content below to check if it's SPAM or NOT SPAM.")

# Text input
input_subject = st.text_input("📧 Enter the email subject here:")

input_mail = st.text_area("✉ Paste the email content here:")

# Prediction
if st.button("Check Now"):
    if input_subject.strip() == "" and input_mail.strip() == "":
        st.warning("⚠ Please enter the email subject or content.")

    else:
        try:
            combined_text = input_subject + " " + input_mail
            cleaned_input = clean_text(combined_text)


            # ✅ Show what's being predicted
            st.write("🔍 Cleaned Input:", cleaned_input)

            # Predict
            transformed_input = vectorizer.transform([cleaned_input])
            prediction = model.predict(transformed_input)[0]
            proba = model.predict_proba(transformed_input)[0]

            st.write(f"🧠 Prediction Confidence — Not Spam: {proba[0]:.2f}, Spam: {proba[1]:.2f}")

            prediction_result = "SPAM" if prediction == 1 else "NOT SPAM"
            download_content = f"""
            Prediction Result: {prediction_result}
            Not Spam Confidence: {proba[0]:.2f}
            Spam Confidence: {proba[1]:.2f}

            Cleaned Email Content:
            {cleaned_input}
            """

            # Download combined result + cleaned text
            st.download_button(
                label="📥 Download Result & Cleaned Text",
                data=download_content,
                file_name="spam_detection_result.txt",
                mime="text/plain"
            )

            # Gauge / Dial Chart for Spam Probability
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba[1]*100,
                title={'text': "Spam Probability (%)"},
                gauge={'axis': {'range': [0, 100]},
                    'bar': {'color': "red"}}
            ))

            st.write("### 🎯 Spam Probability Gauge")
            st.plotly_chart(fig)


            # ✅ Feedback collection added here
            st.write("### 📝 Feedback")
            feedback = st.radio("Was this prediction correct?", ["Yes", "No"])
            comments = st.text_input("Any comments or suggestions?")

            if st.button("Submit Feedback"):
                st.success("✅ Thank you for your feedback!")
                st.write("🔎 Your Response:")
                st.write(f"Prediction Correct: {feedback}")
                if comments:
                    st.write(f"Comments: {comments}")

            if prediction == 1:
                st.error("🚨 This is SPAM!")
            else:
                st.success("✅ This is NOT SPAM.")
        except Exception as e:
            st.error("❌ Prediction failed.")
            st.exception(e)

# BATCH EMAIL PREDICTION
st.markdown("---")
st.header("📁 Batch Email Spam Detection")

uploaded_file = st.file_uploader("Upload a CSV file with emails", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        if 'email' not in df.columns:
            st.error("❌ CSV must have a column named 'email'.")
        else:
            st.success("✅ File uploaded successfully!")

            # Clean and predict for each email
            cleaned_emails = df['email'].apply(clean_text)
            transformed_inputs = vectorizer.transform(cleaned_emails)
            predictions = model.predict(transformed_inputs)
            proba = model.predict_proba(transformed_inputs)
            
            # Prepare results dataframe
            results_df = pd.DataFrame({
                'Original Email': df['email'],
                'Cleaned Email': cleaned_emails,
                'Prediction': ['SPAM' if p == 1 else 'NOT SPAM' for p in predictions],
                'Not Spam Confidence': proba[:,0],
                'Spam Confidence': proba[:,1]
            })

            st.write("### 📊 Batch Prediction Results")
            st.dataframe(results_df)

            # Download results
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name='batch_spam_detection_results.csv',
                mime='text/csv'
            )
    except Exception as e:
        st.error("❌ Prediction failed.")
        st.exception(e)



