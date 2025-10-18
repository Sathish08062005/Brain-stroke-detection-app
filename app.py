import streamlit as st
import numpy as np
import cv2
import json
import requests
import os
import gdown
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# -------------------------
# Users file for persistence
# -------------------------
USERS_FILE = "users.json"

def save_users_to_file():
    with open(USERS_FILE, "w") as f:
        json.dump(st.session_state.users, f, indent=2)

# -------------------------
# Page Config
# -------------------------
st.set_page_config(page_title="🧠 Stroke Detection App", layout="centered")

# -------------------------
# App Branding
# -------------------------
st.markdown("# 🧠 NeuroNexusAI ")

# -------------------------
# Load trained classification model
# -------------------------
MODEL_PATH = "stroke_model.h5"
DRIVE_FILE_ID = "12Azoft-5R2x8uDTMr2wkTQIHT-c2274z"  
DRIVE_URL = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"

if not os.path.exists(MODEL_PATH):
    with st.spinner("⬇ Downloading stroke model... please wait ⏳"):
        gdown.download(DRIVE_URL, MODEL_PATH, quiet=False)

@st.cache_resource(show_spinner=False)
def load_stroke_model():
    return load_model(MODEL_PATH)

model = load_stroke_model()

# -------------------------
# Preprocess image for classification
# -------------------------
def preprocess_image(image):
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def classify_image(image):
    processed = preprocess_image(image)
    prediction = model.predict(processed)[0][0]
    stroke_prob = float(prediction)
    no_stroke_prob = 1 - stroke_prob
    return stroke_prob, no_stroke_prob

def highlight_stroke_regions(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blurred, 180, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(image)
    for cnt in contours:
        if cv2.contourArea(cnt) > 100:
            cv2.drawContours(mask, [cnt], -1, (0, 0, 255), -1)
    highlighted = cv2.addWeighted(image, 0.7, mask, 0.3, 0)
    return highlighted

# -------------------------
# Auth state
# -------------------------
def ensure_state():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "role" not in st.session_state:
        st.session_state.role = None
    if "username" not in st.session_state:
        st.session_state.username = None
    if "users" not in st.session_state:
        if os.path.exists(USERS_FILE):
            try:
                with open(USERS_FILE, "r") as f:
                    st.session_state.users = json.load(f)
            except:
                st.session_state.users = {"Sathish": {"password": "Praveenasathish", "role": "admin"}}
        else:
            st.session_state.users = {"Sathish": {"password": "Praveenasathish", "role": "admin"}}
    if "settings" not in st.session_state:
        st.session_state.settings = {
            "BOT_TOKEN": "8427091249:AAHZpuUI9A6xjA6boADh-nuO7SyYqMygMTY",
            "CHAT_ID": "6250672742",
        }
    if "report_log" not in st.session_state:
        st.session_state.report_log = []
    if "appointments" not in st.session_state:
        st.session_state.appointments = []

ensure_state()

# -------------------------
# Auth functions
# -------------------------
def login(username, password):
    users = st.session_state.users
    if username in users and users[username]["password"] == password:
        st.session_state.logged_in = True
        st.session_state.username = username
        st.session_state.role = users[username]["role"]
        return True
    return False

def logout():
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.role = None

def add_user(new_username, new_password, role="user"):
    if not new_username or not new_password:
        return False, "Username and password are required."
    if new_username in st.session_state.users:
        return False, "Username already exists."
    st.session_state.users[new_username] = {"password": new_password, "role": role}
    save_users_to_file()
    return True, f"User '{new_username}' created."

def delete_user(username):
    if username == "Sathish":
        return False, "Cannot delete the default admin."
    if username not in st.session_state.users:
        return False, "User not found."
    del st.session_state.users[username]
    save_users_to_file()
    return True, f"User '{username}' deleted."

def reset_password(username, new_password):
    if username not in st.session_state.users:
        return False, "User not found."
    st.session_state.users[username]["password"] = new_password
    save_users_to_file()
    return True, f"Password reset for '{username}'."

def export_users_json():
    return json.dumps(st.session_state.users, indent=2)

# -------------------------
# UI: Login
# -------------------------
def render_login():
    st.title("🔐 Login Portal")
    username = st.text_input("Username", key="login_username")
    password = st.text_input("Password", type="password", key="login_password")
    if st.button("Login", use_container_width=True, key="login_btn"):
        if login(username, password):
            st.success("Login successful ✅")
            st.rerun()
        else:
            st.error("❌ Invalid Username or Password")

# -------------------------
# Admin Dashboard
# -------------------------
def render_admin_dashboard():
    st.title("🛡 Admin Dashboard")
    st.write(f"Welcome, {st.session_state.username} (admin)")
    if st.button("🚪 Logout", key="admin_logout_btn"):
        logout()
        st.rerun()

    with st.expander("🩺 View Doctor Appointments"):
        if not st.session_state.appointments:
            st.info("No appointment requests yet.")
        else:
            for idx, appt in enumerate(st.session_state.appointments):
                st.write(f"**{appt['patient_name']}** | {appt['doctor']} | {appt['date']} {appt['time']}")
                st.write(f"Status: {appt['status']}")
                if st.button(f"✅ Approve {idx}", key=f"approve_{idx}"):
                    appt["status"] = "Approved"
                if st.button(f"❌ Reject {idx}", key=f"reject_{idx}"):
                    appt["status"] = "Rejected"
                st.divider()

# -------------------------
# Stroke App Main UI (With Inline Appointment)
# -------------------------
def render_user_app():
    st.title("🧠 Stroke Detection from CT/MRI Scans")
    uploaded_file = st.file_uploader("📤 Upload CT/MRI Image", type=["jpg", "png", "jpeg"], key="upload_scan")

    if uploaded_file is not None:
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
        st.image(image, caption="Uploaded Scan", use_column_width=True)
        stroke_prob, no_stroke_prob = classify_image(image)
        stroke_percent = stroke_prob * 100
        no_stroke_percent = no_stroke_prob * 100
        st.write(f"🩸 Stroke: {stroke_percent:.2f}% | ✅ No Stroke: {no_stroke_percent:.2f}%")

    st.write("---")
    st.subheader("🩺 Book Doctor Appointment (Inline – No Rerun)")

    with st.form("appointment_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            patient_name = st.text_input("Patient Name", key="form_patient_name")
            mobile = st.text_input("Mobile Number", key="form_mobile")
        with col2:
            date = st.date_input("Appointment Date", key="form_date")
            time = st.time_input("Preferred Time", key="form_time")
        doctor = st.selectbox(
            "Select Doctor",
            ["Dr. Ramesh (Neurologist)", "Dr. Priya (Radiologist)", "Dr. Kumar (Specialist)"],
            key="form_doctor",
        )
        submitted = st.form_submit_button("📩 Send Appointment Request")
        if submitted:
            appt = {
                "patient_name": patient_name,
                "mobile": mobile,
                "date": str(date),
                "time": str(time),
                "doctor": doctor,
                "status": "Pending",
                "requested_by": st.session_state.username,
            }
            st.session_state.appointments.append(appt)
            st.success("✅ Appointment request sent successfully!")

    if st.button("🚪 Logout", key="user_logout_btn"):
        logout()
        st.rerun()

# -------------------------
# Main Routing
# -------------------------
if not st.session_state.logged_in:
    render_login()
else:
    if st.session_state.role == "admin":
        render_admin_dashboard()
    else:
        render_user_app()
