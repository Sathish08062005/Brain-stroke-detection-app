import streamlit as st
import numpy as np
import cv2
import json
import requests
import os
import gdown
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from io import BytesIO

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
st.markdown("# 🧠 NeuroNexusAI")

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
    return np.expand_dims(image, axis=0)

def classify_image(image):
    processed = preprocess_image(image)
    prediction = model.predict(processed)[0][0]
    return float(prediction), float(1 - prediction)

def highlight_stroke_regions(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blurred, 180, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(image)
    for cnt in contours:
        if cv2.contourArea(cnt) > 100:
            cv2.drawContours(mask, [cnt], -1, (0, 0, 255), -1)
    return cv2.addWeighted(image, 0.7, mask, 0.3, 0)

# -------------------------
# Auth + State Init
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
        st.session_state.appointments = []
    if "appt_temp" not in st.session_state:
        st.session_state.appt_temp = {
            "name": "John Doe", "mobile": "9876543210", "age": 45,
            "date": None, "time": None, "doctor": None
        }

ensure_state()

# -------------------------
# Telegram Send Function
# -------------------------
def send_report_to_telegram(message, image=None):
    BOT_TOKEN = st.session_state.settings.get("BOT_TOKEN", "")
    CHAT_ID = st.session_state.settings.get("CHAT_ID", "")

    if not BOT_TOKEN or not CHAT_ID:
        return False, "Telegram settings missing."

    try:
        url_msg = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        res = requests.post(url_msg, data={"chat_id": CHAT_ID, "text": message})
        
        if res.status_code != 200:
            return False, f"Message send failed: {res.text}"

        if image is not None:
            _, buffer = cv2.imencode(".jpg", image)
            img_bytes = BytesIO(buffer.tobytes())
            url_img = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
            res = requests.post(url_img, data={"chat_id": CHAT_ID}, files={"photo": img_bytes})
            if res.status_code != 200:
                return False, f"Image send failed: {res.text}"

        return True, "Report sent successfully."
    except Exception as e:
        return False, f"Error: {e}"

# -------------------------
# User Stroke Detection
# -------------------------
def render_user_app():
    st.title("🧠 Stroke Detection from CT/MRI Scans")

    col1, col2 = st.columns(2)
    with col1:
        patient_name = st.text_input("Patient Name", value="John Doe")
        patient_age = st.number_input("Patient Age", 1, 120, 45)
        patient_gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    with col2:
        patient_id = st.text_input("Patient ID / Hospital No.", value="P12345")
        patient_contact = st.text_input("Patient Contact Number", value="9876543210")
        patient_address = st.text_area("Patient Address", value="Chennai, India")

    uploaded_file = st.file_uploader("📤 Upload CT/MRI Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        st.image(image, caption="🖼 Uploaded Scan", use_column_width=True)

        stroke_prob, no_stroke_prob = classify_image(image)
        stroke_percent, no_stroke_percent = stroke_prob * 100, no_stroke_prob * 100

        st.write(f"🩸 Stroke Probability: {stroke_percent:.2f}%")
        st.write(f"✅ No Stroke Probability: {no_stroke_percent:.2f}%")

        marked_image = highlight_stroke_regions(image) if stroke_prob > 0.5 else image

        if st.button("💾 Save & Send to Telegram"):
            message = (
                f"🧾 Stroke Report\n\n"
                f"👤 {patient_name} | {patient_age} yrs | {patient_gender}\n"
                f"🆔 {patient_id} | 📞 {patient_contact}\n"
                f"🏠 {patient_address}\n\n"
                f"🩸 Stroke: {stroke_percent:.2f}%\n"
                f"✅ No Stroke: {no_stroke_percent:.2f}%"
            )
            ok, msg = send_report_to_telegram(message, marked_image)
            if ok:
                st.success(msg)
                st.session_state.report_log.append({
                    "patient_name": patient_name,
                    "stroke_percent": stroke_percent,
                    "no_stroke_percent": no_stroke_percent,
                    "by": st.session_state.username or "unknown"
                })
            else:
                st.error(msg)

# -------------------------
# User Appointment Booking
# -------------------------
def render_appointment_booking():
    st.subheader("📅 Book Appointment")

    with st.form("appointment_form"):
        name = st.text_input("Patient Name", value=st.session_state.appt_temp["name"])
        mobile = st.text_input("Mobile Number", value=st.session_state.appt_temp["mobile"])
        age = st.number_input("Age", 1, 120, st.session_state.appt_temp["age"])
        date = st.date_input("Preferred Date")
        time = st.time_input("Preferred Time")
        doctor = st.text_input("Doctor's Name")

        submit = st.form_submit_button("✅ Confirm Appointment")

        if submit:
            appt = {
                "name": name,
                "mobile": mobile,
                "age": age,
                "date": str(date),
                "time": str(time),
                "doctor": doctor,
                "booked_by": st.session_state.username or "unknown"
            }
            st.session_state.appointments.append(appt)

            # send to admin telegram
            msg = (
                f"📅 New Appointment\n\n"
                f"👤 {name} ({age} yrs)\n"
                f"📞 {mobile}\n"
                f"🩺 Doctor: {doctor}\n"
                f"📅 {date} ⏰ {time}"
            )
            ok, resp = send_report_to_telegram(msg)
            if ok:
                st.success("Appointment booked & sent to admin ✅")
            else:
                st.error(f"Failed: {resp}")

# -------------------------
# Admin Dashboard
# -------------------------
def render_admin_dashboard():
    st.title("🛡 Admin Dashboard")

    st.subheader("📋 Appointment Requests")
    if not st.session_state.appointments:
        st.info("No appointments booked yet.")
    else:
        for appt in st.session_state.appointments:
            st.write(
                f"👤 {appt['name']} ({appt['age']} yrs)\n"
                f"📞 {appt['mobile']}\n"
                f"🩺 Doctor: {appt['doctor']}\n"
                f"📅 {appt['date']} ⏰ {appt['time']}\n"
                f"Booked by: {appt['booked_by']}"
            )
            st.divider()

    st.subheader("🧾 Stroke Reports Log")
    if not st.session_state.report_log:
        st.info("No stroke reports yet.")
    else:
        for report in st.session_state.report_log:
            st.write(
                f"👤 {report['patient_name']}\n"
                f"🩸 Stroke: {report['stroke_percent']:.2f}% | ✅ No Stroke: {report['no_stroke_percent']:.2f}%\n"
                f"Submitted by: {report['by']}"
            )
            st.divider()

# -------------------------
# Main Routing
# -------------------------
if not st.session_state.logged_in:
    st.write("🔐 Please login first (Admin/User).")
else:
    if st.session_state.role == "admin":
        render_admin_dashboard()
    else:
        render_user_app()
        render_appointment_booking()
        
