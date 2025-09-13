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
# Users file for persistence
# -------------------------
USERS_FILE = "users.json"
APPOINTMENTS_FILE = "appointments.json"  # new for appointments

def save_users_to_file():
    with open(USERS_FILE, "w") as f:
        json.dump(st.session_state.users, f, indent=2)

def load_appointments():
    if os.path.exists(APPOINTMENTS_FILE):
        with open(APPOINTMENTS_FILE, "r") as f:
            return json.load(f)
    return []

def save_appointments(data):
    with open(APPOINTMENTS_FILE, "w") as f:
        json.dump(data, f, indent=4)

# -------------------------
# Page Config
# -------------------------
st.set_page_config(page_title="🧠 Stroke Detection App", layout="centered")

# -------------------------
# App Branding
# -------------------------
st.markdown("# 🧠 NeuroNexusAI", unsafe_allow_html=True)

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
# Preprocess + Classify
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
        st.session_state.settings = {"BOT_TOKEN": "", "CHAT_ID": ""}
    if "report_log" not in st.session_state:
        st.session_state.report_log = []
    if "appointments" not in st.session_state:
        st.session_state.appointments = load_appointments()

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
# Appointment Functions
# -------------------------
def render_appointment_booking():
    st.subheader("📅 Book Appointment")
    doctors = ["Dr. Arjun (Neurologist)", "Dr. Priya (General Physician)", "Dr. Karthik (Stroke Specialist)"]
    doctor = st.selectbox("Choose Doctor", doctors)
    date = st.date_input("Select Date")
    time = st.time_input("Select Time")
    age = st.number_input("Patient Age", min_value=1, max_value=120, step=1)
    mobile = st.text_input("Mobile Number")
    reason = st.text_area("Reason for Appointment")
    reminder = st.text_area("Reminder Notes")

    if st.button("Book Appointment"):
        new_appt = {
            "id": len(st.session_state.appointments) + 1,
            "doctor": doctor,
            "date": str(date),
            "time": str(time),
            "name": st.session_state.username,
            "age": age,
            "mobile": mobile,
            "reason": reason,
            "reminder": reminder,
            "status": "Pending",
            "updated_by_admin": False,
        }
        st.session_state.appointments.append(new_appt)
        save_appointments(st.session_state.appointments)
        st.success("✅ Appointment booked and sent to Admin!")

def render_my_appointments(user):
    st.subheader("📋 My Appointments")
    my_appts = [a for a in st.session_state.appointments if a["name"] == user]
    if not my_appts:
        st.info("No appointments yet.")
    for appt in my_appts:
        with st.expander(f"{appt['doctor']} | {appt['date']} {appt['time']} | Status: {appt['status']}"):
            st.write(appt)

def render_admin_appointments():
    st.subheader("🛠 Manage Appointments")
    if not st.session_state.appointments:
        st.info("No appointments yet.")
    for appt in st.session_state.appointments:
        with st.expander(f"{appt['name']} | {appt['doctor']} | {appt['date']} {appt['time']} | Status: {appt['status']}"):
            new_doc = st.selectbox("Change Doctor", ["--No Change--"] + ["Dr. Arjun (Neurologist)", "Dr. Priya (General Physician)", "Dr. Karthik (Stroke Specialist)"], key=f"doc_{appt['id']}")
            new_date = st.date_input("Change Date", datetime.strptime(appt["date"], "%Y-%m-%d").date(), key=f"date_{appt['id']}")
            new_time = st.time_input("Change Time", datetime.strptime(appt["time"], "%H:%M:%S").time(), key=f"time_{appt['id']}")
            if st.button("✅ Approve", key=f"approve_{appt['id']}"):
                appt["status"] = "Approved"
                if new_doc != "--No Change--": appt["doctor"] = new_doc
                appt["date"], appt["time"] = str(new_date), str(new_time)
                appt["updated_by_admin"] = True
                save_appointments(st.session_state.appointments)
            if st.button("❌ Reject", key=f"reject_{appt['id']}"):
                appt["status"] = "Rejected"
                appt["updated_by_admin"] = True
                save_appointments(st.session_state.appointments)

# -------------------------
# UI: Login
# -------------------------
def render_login():
    st.title("🔐 Login Portal")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login", use_container_width=True):
        if login(username, password):
            st.success("✅ Login successful")
            st.rerun()
        else:
            st.error("❌ Invalid Username or Password")

# -------------------------
# Admin Dashboard
# -------------------------
def render_admin_dashboard():
    st.title("🛡 Admin Dashboard")
    st.write(f"Welcome, {st.session_state.username} (admin)")
    if st.button("🚪 Logout"):
        logout()
        st.rerun()

    tabs = st.tabs(["👤 Create User", "🧑‍🤝‍🧑 Manage Users", "📤 Export/Import", "📨 Telegram", "📅 Appointments"])
    with tabs[4]:
        render_admin_appointments()

# -------------------------
# Stroke App + Appointment UI
# -------------------------
def render_user_app():
    st.title("🧠 Stroke Detection from CT/MRI Scans")
    st.write("Upload a brain scan image and check stroke probability.")

    # ---- Stroke detection UI ----
    uploaded_file = st.file_uploader("📤 Upload CT/MRI Image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        st.image(image, caption="🖼 Uploaded Scan", use_column_width=True)

        stroke_prob, no_stroke_prob = classify_image(image)
        st.write(f"🩸 Stroke: {stroke_prob*100:.2f}% | ✅ No Stroke: {no_stroke_prob*100:.2f}%")

    # ---- Appointment Booking ----
    st.divider()
    render_appointment_booking()
    render_my_appointments(st.session_state.username)

    if st.button("🚪 Logout"):
        logout()
        st.rerun()

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
