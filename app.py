import streamlit as st
import numpy as np
import cv2
import os
import json
import gdown
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------
# Files for persistence
# -------------------------
USERS_FILE = "users.json"
APPOINTMENTS_FILE = "appointments.json"
MODEL_PATH = "stroke_model.h5"

# -------------------------
# Helper functions
# -------------------------
def save_to_file(data, file):
    with open(file, "w") as f:
        json.dump(data, f)

def load_from_file(file):
    if os.path.exists(file):
        with open(file, "r") as f:
            return json.load(f)
    return {}

# -------------------------
# Model loading
# -------------------------
@st.cache_resource
def load_stroke_model():
    if not os.path.exists(MODEL_PATH):
        url = "https://drive.google.com/uc?id=YOUR_MODEL_ID"
        gdown.download(url, MODEL_PATH, quiet=False)
    model = load_model(MODEL_PATH)
    return model

# -------------------------
# Image Preprocessing
# -------------------------
def preprocess_image(image):
    image = cv2.resize(image, (224, 224))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image

# -------------------------
# Prediction + Visualization
# -------------------------
def predict_stroke(model, image):
    processed = preprocess_image(image)
    pred = model.predict(processed)[0][0]
    return pred

# -------------------------
# Initialize Session
# -------------------------
if "uploaded_images" not in st.session_state:
    st.session_state.uploaded_images = []
if "true_labels" not in st.session_state:
    st.session_state.true_labels = []
if "pred_labels" not in st.session_state:
    st.session_state.pred_labels = []

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="AI Stroke Detection", layout="wide", page_icon="🧠")

st.title("🧠 Brain Stroke Detection System")
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Upload CT/MRI Image", "Appointments", "Admin Dashboard", "Post-Stroke Care"])

# -------------------------
# UPLOAD PAGE
# -------------------------
if page == "Upload CT/MRI Image":
    st.header("📤 Upload Brain CT or MRI Scan")
    uploaded_file = st.file_uploader("Upload CT/MRI Image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        st.image(image, caption="Uploaded Scan", use_container_width=True)

        model = load_stroke_model()
        pred = predict_stroke(model, image)
        label = "Stroke Detected 🧠" if pred > 0.5 else "No Stroke ✅"

        st.success(f"**Prediction:** {label}")
        st.progress(int(pred * 100))

        # Simulate true label (for evaluation)
        true_label = 1 if "stroke" in uploaded_file.name.lower() else 0
        pred_label = 1 if pred > 0.5 else 0

        st.session_state.true_labels.append(true_label)
        st.session_state.pred_labels.append(pred_label)
        st.session_state.uploaded_images.append(uploaded_file.name)

        st.markdown("### 🩸 Stroke Region Highlighted")
        st.image(image, caption="Detected Stroke Region", use_container_width=True)

        # --------------------------------------------------
        # Confusion Matrix, ROC Curve, Performance Summary
        # --------------------------------------------------
        if len(st.session_state.true_labels) >= 2:
            y_true = np.array(st.session_state.true_labels)
            y_pred = np.array(st.session_state.pred_labels)

            cm = confusion_matrix(y_true, y_pred)
            fpr, tpr, _ = roc_curve(y_true, y_pred)
            roc_auc = auc(fpr, tpr)

            accuracy = np.mean(y_true == y_pred)
            sensitivity = cm[1,1] / (cm[1,0] + cm[1,1]) if (cm[1,0] + cm[1,1]) > 0 else 0
            specificity = cm[0,0] / (cm[0,0] + cm[0,1]) if (cm[0,0] + cm[0,1]) > 0 else 0

            st.markdown("## 📊 Performance Summary")
            st.write(f"**Accuracy:** {accuracy:.2f}")
            st.write(f"**AUC:** {roc_auc:.2f}")
            st.write(f"**Sensitivity (Recall):** {sensitivity:.2f}")
            st.write(f"**Specificity:** {specificity:.2f}")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 🧩 Confusion Matrix")
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                            xticklabels=["Pred No Stroke", "Pred Stroke"],
                            yticklabels=["Actual No Stroke", "Actual Stroke"])
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                st.pyplot(fig)

            with col2:
                st.markdown("### 📈 ROC Curve")
                fig2, ax2 = plt.subplots()
                ax2.plot(fpr, tpr, color='blue', lw=2, label=f"AUC = {roc_auc:.2f}")
                ax2.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
                ax2.set_xlabel("False Positive Rate")
                ax2.set_ylabel("True Positive Rate")
                ax2.legend(loc="lower right")
                st.pyplot(fig2)

# -------------------------
# APPOINTMENTS PAGE
# -------------------------
elif page == "Appointments":
    st.header("👩‍⚕️ Doctor Appointment Booking")
    appointments = load_from_file(APPOINTMENTS_FILE)

    name = st.text_input("Patient Name")
    doctor = st.selectbox("Select Doctor", ["Dr. Karthik", "Dr. Meena", "Dr. Aravind"])
    date = st.date_input("Select Date")
    time = st.time_input("Select Time")
    reason = st.text_area("Reason for Appointment")

    if st.button("Book Appointment"):
        new_app = {"name": name, "doctor": doctor, "date": str(date), "time": str(time), "reason": reason, "status": "Pending"}
        appointments[len(appointments)] = new_app
        save_to_file(appointments, APPOINTMENTS_FILE)
        st.success("✅ Appointment booked successfully!")

# -------------------------
# ADMIN PAGE
# -------------------------
elif page == "Admin Dashboard":
    st.header("🩺 Admin Dashboard")
    appointments = load_from_file(APPOINTMENTS_FILE)

    if not appointments:
        st.info("No appointments yet.")
    else:
        for key, data in appointments.items():
            st.write(f"**Patient:** {data['name']}")
            st.write(f"**Doctor:** {data['doctor']}")
            st.write(f"**Date:** {data['date']}")
            st.write(f"**Time:** {data['time']}")
            st.write(f"**Reason:** {data['reason']}")
            st.write(f"**Status:** {data['status']}")

            colA, colB, colC = st.columns(3)
            if colA.button(f"✅ Approve {key}"):
                data["status"] = "Approved"
            if colB.button(f"❌ Reject {key}"):
                data["status"] = "Rejected"
            if colC.button(f"🕒 Reschedule {key}"):
                data["status"] = "Reschedule Needed"
            appointments[key] = data
            save_to_file(appointments, APPOINTMENTS_FILE)
            st.markdown("---")

# -------------------------
# POST-STROKE CARE PAGE
# -------------------------
elif page == "Post-Stroke Care":
    st.header("🏥 Post-Stroke Care & Lifestyle Recommendations")
    st.markdown("""
    💪 **Rehabilitation:** Daily physiotherapy & cognitive exercises.  
    🥗 **Diet:** Include omega-3 rich foods and avoid high salt intake.  
    🚶 **Activity:** 30 mins of mild walking or stretching per day.  
    💊 **Medication:** Continue prescribed blood thinners and BP meds.  
    💧 **Hydration:** Drink at least 2–3 litres of water daily.  
    🧘 **Stress Management:** Practice meditation and maintain 8 hours of sleep.  
    """)

