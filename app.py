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
APPOINTMENTS_FILE = "data/appointments.json"

def save_users_to_file():
    with open(USERS_FILE, "w") as f:
        json.dump(st.session_state.users, f, indent=2)

def load_appointments():
    if not os.path.exists(APPOINTMENTS_FILE):
        return {"requests": [], "approved": []}
    with open(APPOINTMENTS_FILE, "r") as f:
        return json.load(f)

def save_appointments(data):
    os.makedirs(os.path.dirname(APPOINTMENTS_FILE), exist_ok=True)
    with open(APPOINTMENTS_FILE, "w") as f:
        json.dump(data, f, indent=2)

# -------------------------
# Page Config
# -------------------------
st.set_page_config(page_title="🧠 Stroke Detection App", layout="centered")

# -------------------------
# App Branding
# -------------------------
st.markdown(
    """ 
#  🧠 NeuroNexusAI 
 """,
    unsafe_allow_html=True,
)

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
    if "settings" not in st.session_state:
        st.session_state.settings = {
            "BOT_TOKEN": "8427091249:AAHZpuUI9A6xjA6boADh-nuO7SyYqMygMTY",
            "CHAT_ID": "6250672742",
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
# Doctor Appointment Features (Admin)
# -------------------------
def render_appointments_tab():
    st.subheader("📅 Doctor Appointment Requests")
    data = load_appointments()
    
    st.write("📝 Pending Requests:")
    for i, req in enumerate(data["requests"]):
        st.write(f"👤 {req['patient_name']} | {req['date']} {req['time']} | Doctor: {req['doctor']}")
        col1, col2, col3, col4 = st.columns([1,1,2,2])
        with col1:
            if st.button(f"✅ Approve #{i+1}", key=f"approve_{i}"):
                data["approved"].append(req)
                del data["requests"][i]
                save_appointments(data)
                st.success("Appointment approved!")
                st.experimental_rerun()
        with col2:
            if st.button(f"❌ Reject #{i+1}", key=f"reject_{i}"):
                del data["requests"][i]
                save_appointments(data)
                st.success("Appointment rejected!")
                st.experimental_rerun()
        with col3:
            new_date = st.date_input(f"Change Date #{i+1}", value=datetime.strptime(req['date'], "%Y-%m-%d"))
            new_time = st.time_input(f"Change Time #{i+1}", value=datetime.strptime(req['time'], "%H:%M").time())
            if st.button(f"📝 Update Date/Time #{i+1}", key=f"update_time_{i}"):
                req['date'] = str(new_date)
                req['time'] = str(new_time)
                save_appointments(data)
                st.success("Date & Time updated!")
                st.experimental_rerun()
        with col4:
            new_doctor = st.text_input(f"Change Doctor #{i+1}", value=req['doctor'], key=f"doctor_{i}")
            if st.button(f"📝 Update Doctor #{i+1}", key=f"update_doctor_{i}"):
                req['doctor'] = new_doctor
                save_appointments(data)
                st.success("Doctor updated!")
                st.experimental_rerun()

    st.write("---")
    st.write("📋 Approved Appointments:")
    for i, appt in enumerate(data["approved"]):
        st.write(f"✅ {appt['patient_name']} | {appt['date']} {appt['time']} | Doctor: {appt['doctor']}")

# -------------------------
# Admin Dashboard
# -------------------------
def render_admin_dashboard():
    st.title("🛡 Admin Dashboard")
    st.write(f"Welcome, {st.session_state.username} (admin)")

    with st.sidebar:
        st.header("⚙ Admin Actions")
        if st.button("🚪 Logout"):
            logout()
            st.rerun()

    tabs = st.tabs(["👤 Create User", "🧑‍🤝‍🧑 Manage Users", "📤 Export/Import", "📨 Telegram Settings", "📅 Doctor Appointments"])

    # Existing tabs remain unchanged
    with tabs[4]:
        render_appointments_tab()

# -------------------------
# Login & App Router
# -------------------------
if not st.session_state.logged_in:
    # Your existing render_login code here
    pass
else:
    if st.session_state.role == "admin":
        render_admin_dashboard()
    else:
        # Your existing render_user_app code here
        pass
