import streamlit as st
import numpy as np
import cv2
import json
import requests
import os
import gdown
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from datetime import datetime

# -------------------------
# Users & Appointments file
# -------------------------
USERS_FILE = "users.json"
APPOINTMENTS_FILE = "appointments.json"

def save_users_to_file():
    with open(USERS_FILE, "w") as f:
        json.dump(st.session_state.users, f, indent=2)

def save_appointments_to_file():
    with open(APPOINTMENTS_FILE, "w") as f:
        json.dump(st.session_state.appointments, f, indent=2)

def load_appointments_from_file():
    if os.path.exists(APPOINTMENTS_FILE):
        try:
            with open(APPOINTMENTS_FILE, "r") as f:
                return json.load(f)
        except:
            return []
    return []

# -------------------------
# Page Config
# -------------------------
st.set_page_config(page_title="🧠 Stroke Detection App", layout="centered")

# -------------------------
# App Branding
# -------------------------
st.markdown("# 🧠 NeuroNexusAI")

# -------------------------
# Load trained classification model
# -------------------------
MODEL_PATH = "stroke_model.h5"
DRIVE_FILE_ID = "12Azoft-5R2x8uDTMr2wkTQIHT-c2274z"  # replace with your file ID
DRIVE_URL = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"

if not os.path.exists(MODEL_PATH):
    with st.spinner("⬇ Downloading stroke model... please wait ⏳"):
        gdown.download(DRIVE_URL, MODEL_PATH, quiet=False)

@st.cache_resource(show_spinner=False)
def load_stroke_model():
    return load_model(MODEL_PATH)

model = load_stroke_model()

# -------------------------
# Preprocess image
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
    if "settings" not in st.session_state:
        st.session_state.settings = {
            "BOT_TOKEN": "",
            "CHAT_ID": "",
        }
    if "report_log" not in st.session_state:
        st.session_state.report_log = []
    if "appointments" not in st.session_state:
        st.session_state.appointments = load_appointments_from_file()

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

def import_users_json(file_bytes):
    try:
        data = json.loads(file_bytes.decode("utf-8"))
        for k, v in data.items():
            if not isinstance(v, dict) or "password" not in v or "role" not in v:
                return False, "Invalid users JSON schema."
        st.session_state.users = data
        save_users_to_file()
        return True, "Users imported."
    except Exception as e:
        return False, f"Import failed: {e}"

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
# Appointment Booking (User)
# -------------------------
def render_appointment_booking():
    st.subheader("📅 Book an Appointment")
    patient_name = st.text_input("Patient Name", value=st.session_state.username)
    patient_age = st.number_input("Patient Age", min_value=1, max_value=120, value=30)
    patient_mobile = st.text_input("Mobile Number")
    reason = st.text_area("Reason for Appointment")
    reminder = st.text_input("Reminder Note (optional)")

    doctor = st.selectbox("Select Doctor", ["Dr. Smith (Neurologist)", "Dr. Kumar (Stroke Specialist)", "Dr. Lee (Neurosurgeon)"])
    date = st.date_input("Select Date")
    time = st.time_input("Select Time")

    if st.button("📨 Send Appointment Request"):
        req = {
            "id": len(st.session_state.appointments) + 1,
            "patient_name": patient_name,
            "age": patient_age,
            "mobile": patient_mobile,
            "reason": reason,
            "reminder": reminder,
            "doctor": doctor,
            "date": str(date),
            "time": str(time),
            "status": "Pending",
        }
        st.session_state.appointments.append(req)
        save_appointments_to_file()
        st.success("✅ Appointment request sent successfully!")

# -------------------------
# Admin Dashboard
# -------------------------
def render_admin_dashboard():
    st.title("🛡 Admin Dashboard")
    st.write(f"Welcome, {st.session_state.username} (admin)")

    st.sidebar.header("⚙ Admin Actions")
    if st.button("🚪 Logout"):
        logout()
        st.rerun()

    tabs = st.tabs(["👤 Users", "📨 Telegram", "📅 Appointments", "📝 Reports"])

    with tabs[0]:
        st.subheader("Manage Users")
        for uname, meta in st.session_state.users.items():
            st.write(f"👤 {uname} | Role: {meta['role']}")

    with tabs[1]:
        st.subheader("Telegram Settings")
        st.write("Bot Token / Chat ID already configured.")

    with tabs[2]:
        st.subheader("📅 Appointment Requests")
        if not st.session_state.appointments:
            st.info("No appointment requests yet.")
        for appt in st.session_state.appointments:
            st.write("---")
            st.write(f"👤 {appt['patient_name']} | Age: {appt['age']} | Mobile: {appt['mobile']}")
            st.write(f"🧑‍⚕️ Doctor: {appt['doctor']}")
            st.write(f"📅 Date: {appt['date']} | ⏰ Time: {appt['time']}")
            st.write(f"📌 Reason: {appt['reason']}")
            st.write(f"📝 Reminder: {appt['reminder']}")
            st.write(f"📊 Status: {appt['status']}")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                if st.button(f"✅ Approve {appt['id']}", key=f"approve_{appt['id']}"):
                    appt["status"] = "Approved"
                    save_appointments_to_file()
                    st.success("Approved ✅")
            with col2:
                if st.button(f"❌ Reject {appt['id']}", key=f"reject_{appt['id']}"):
                    appt["status"] = "Rejected"
                    save_appointments_to_file()
                    st.error("Rejected ❌")
            with col3:
                new_date = st.date_input("Change Date", key=f"date_{appt['id']}")
                new_time = st.time_input("Change Time", key=f"time_{appt['id']}")
                if st.button(f"Update Date/Time {appt['id']}", key=f"dt_{appt['id']}"):
                    appt["date"] = str(new_date)
                    appt["time"] = str(new_time)
                    appt["status"] = "Rescheduled"
                    save_appointments_to_file()
                    st.info("Date/Time updated 🕒")
            with col4:
                new_doc = st.selectbox("Change Doctor", ["Dr. Smith", "Dr. Kumar", "Dr. Lee"], key=f"doc_{appt['id']}")
                if st.button(f"Update Doctor {appt['id']}", key=f"docbtn_{appt['id']}"):
                    appt["doctor"] = new_doc
                    appt["status"] = "Doctor Changed"
                    save_appointments_to_file()
                    st.info("Doctor updated 🧑‍⚕️")

    with tabs[3]:
        st.subheader("📝 Recent Reports")
        if st.session_state.report_log:
            for r in st.session_state.report_log[-5:]:
                st.write(r)
        else:
            st.caption("No reports yet.")

# -------------------------
# Stroke Detection UI
# -------------------------
def render_user_app():
    st.title("🧠 Stroke Detection")
    st.write("Upload CT/MRI scan and book an appointment if needed.")

    render_appointment_booking()

    uploaded_file = st.file_uploader("📤 Upload CT/MRI Image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        st.image(image, caption="🖼 Uploaded Scan", use_column_width=True)

        stroke_prob, no_stroke_prob = classify_image(image)
        stroke_percent, no_stroke_percent = stroke_prob * 100, no_stroke_prob * 100

        st.write(f"🩸 Stroke Probability: {stroke_percent:.2f}%")
        st.write(f"✅ No Stroke Probability: {no_stroke_percent:.2f}%")

# -------------------------
# App Router
# -------------------------
if not st.session_state.logged_in:
    render_login()
else:
    if st.session_state.role == "admin":
        render_admin_dashboard()
    else:
        render_user_app()
