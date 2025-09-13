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
# Users & Appointments file for persistence
# -------------------------
USERS_FILE = "users.json"
APPOINTMENTS_FILE = "appointments.json"

def save_users_to_file():
    with open(USERS_FILE, "w") as f:
        json.dump(st.session_state.users, f, indent=2)

def save_appointments_to_file():
    with open(APPOINTMENTS_FILE, "w") as f:
        json.dump(st.session_state.appointments, f, indent=2)

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
                st.session_state.users = {
                    "Sathish": {"password": "Praveenasathish", "role": "admin"}
                }
        else:
            st.session_state.users = {
                "Sathish": {"password": "Praveenasathish", "role": "admin"}
            }
    if "appointments" not in st.session_state:
        if os.path.exists(APPOINTMENTS_FILE):
            try:
                with open(APPOINTMENTS_FILE, "r") as f:
                    st.session_state.appointments = json.load(f)
            except:
                st.session_state.appointments = []
        else:
            st.session_state.appointments = []
    if "settings" not in st.session_state:
        st.session_state.settings = {
            "BOT_TOKEN": "",
            "CHAT_ID": "",
        }
    if "report_log" not in st.session_state:
        st.session_state.report_log = []

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

# -------------------------
# Appointment Functions
# -------------------------
def add_appointment(data):
    st.session_state.appointments.append(data)
    save_appointments_to_file()

def update_appointment(index, new_data):
    st.session_state.appointments[index].update(new_data)
    save_appointments_to_file()

# -------------------------
# UI: Login
# -------------------------
def render_login():
    st.title("🔐 Login Portal")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login", use_container_width=True):
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

    with st.sidebar:
        if st.button("🚪 Logout"):
            logout()
            st.rerun()

    tabs = st.tabs(["👤 Manage Users", "📝 Reports", "📅 Appointments"])

    with tabs[2]:
        st.subheader("📅 Appointment Requests")
        if st.session_state.appointments:
            for i, appt in enumerate(st.session_state.appointments):
                st.write(f"**Patient:** {appt['patient_name']} | Doctor: {appt['doctor']} | Date: {appt['date']} | Time: {appt['time']} | Status: {appt['status']}")
                new_date = st.date_input(f"Change Date ({appt['patient_name']})", key=f"date_{i}")
                new_time = st.text_input(f"Change Time ({appt['patient_name']})", value=appt['time'], key=f"time_{i}")
                new_doctor = st.text_input(f"Change Doctor ({appt['patient_name']})", value=appt['doctor'], key=f"doc_{i}")
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button(f"✅ Approve {appt['patient_name']}", key=f"approve_{i}"):
                        update_appointment(i, {"status": "Approved", "date": str(new_date), "time": new_time, "doctor": new_doctor})
                        st.success("Appointment Approved")
                with col2:
                    if st.button(f"❌ Reject {appt['patient_name']}", key=f"reject_{i}"):
                        update_appointment(i, {"status": "Rejected"})
                        st.error("Appointment Rejected")
                with col3:
                    if st.button(f"🔄 Update {appt['patient_name']}", key=f"update_{i}"):
                        update_appointment(i, {"date": str(new_date), "time": new_time, "doctor": new_doctor})
                        st.info("Appointment Updated")
        else:
            st.info("No appointment requests yet.")

# -------------------------
# Stroke App Main UI
# -------------------------
def render_user_app():
    st.title("🧠 Stroke Detection & Appointment Booking")

    st.subheader("📅 Book Appointment")
    with st.form("appointment_form"):
        patient_name = st.text_input("Patient Name")
        patient_age = st.number_input("Age", min_value=1, max_value=120)
        patient_mobile = st.text_input("Mobile Number")
        reason = st.text_area("Reason for Appointment")
        reminder = st.text_area("Reminder Notes")
        doctor = st.selectbox("Select Doctor", ["Dr. Kumar", "Dr. Priya", "Dr. John"])
        date = st.date_input("Select Date")
        time = st.text_input("Select Time (e.g., 10:30 AM)")
        submitted = st.form_submit_button("📤 Submit Request")
        if submitted:
            add_appointment({
                "patient_name": patient_name,
                "age": patient_age,
                "mobile": patient_mobile,
                "reason": reason,
                "reminder": reminder,
                "doctor": doctor,
                "date": str(date),
                "time": time,
                "status": "Pending",
                "requested_by": st.session_state.username
            })
            st.success("✅ Appointment request submitted. Await admin approval.")

    st.write("---")
    st.subheader("📋 My Appointment Requests")
    my_appts = [a for a in st.session_state.appointments if a["requested_by"] == st.session_state.username]
    if my_appts:
        for appt in my_appts:
            st.write(f"Doctor: {appt['doctor']} | Date: {appt['date']} | Time: {appt['time']} | Status: {appt['status']}")
    else:
        st.info("No requests yet.")

    with st.sidebar:
        st.write(f"Logged in as: {st.session_state.username}")
        if st.button("🚪 Logout"):
            logout()
            st.rerun()

# -------------------------
# Main
# -------------------------
if not st.session_state.logged_in:
    render_login()
else:
    if st.session_state.role == "admin":
        render_admin_dashboard()
    else:
        render_user_app()
