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
# Page Config
# -------------------------
st.set_page_config(page_title="🧠 Stroke Detection App", layout="centered")

# -------------------------
# App Branding
# -------------------------
st.markdown(
    """
    <h1 style='text-align: center; font-family: "Times New Roman", serif; color: #FF69B4;'>
        🧠 NeuroNexusAI
    </h1>
    """,
    unsafe_allow_html=True
)

# -------------------------
# Download & Load Model
# -------------------------
@st.cache_resource(show_spinner=False)
def load_stroke_model():
    model_path = "stroke_model.h5"
    if not os.path.exists(model_path):
        file_id = "12Azoft-5R2x8uDTMr2wkTQIHT-c2274z"  # your Drive file ID
        url = f"https://drive.google.com/uc?id={file_id}"
        with st.spinner("⏳ Downloading model from Google Drive..."):
            gdown.download(url, model_path, quiet=False)
    return load_model(model_path)

model = load_stroke_model()

# -------------------------
# Preprocess image for classification
# -------------------------
def preprocess_image(image):
    image = cv2.resize(image, (224, 224))  # Resize for model
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Ensure RGB
    image = img_to_array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)
    return image

# -------------------------
# Classify image (stroke vs no stroke)
# -------------------------
def classify_image(image):
    processed = preprocess_image(image)
    prediction = model.predict(processed)[0][0]  # probability of stroke
    stroke_prob = float(prediction)
    no_stroke_prob = 1 - stroke_prob
    return stroke_prob, no_stroke_prob

# -------------------------
# Highlight suspicious stroke regions
# -------------------------
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
# Simple in-memory auth store + helpers
# -------------------------
def ensure_state():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "role" not in st.session_state:
        st.session_state.role = None
    if "username" not in st.session_state:
        st.session_state.username = None
    if "users" not in st.session_state:
        st.session_state.users = {
            "Sathish": {"password": "Praveenasathish", "role": "admin"}  # default admin
        }
    if "settings" not in st.session_state:
        st.session_state.settings = {
            "BOT_TOKEN": "YOUR_BOT_TOKEN",
            "CHAT_ID": "YOUR_CHAT_ID"
        }
    if "report_log" not in st.session_state:
        st.session_state.report_log = []

ensure_state()

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
    return True, f"User '{new_username}' created."

def delete_user(username):
    if username == "Sathish":
        return False, "Cannot delete the default admin."
    if username not in st.session_state.users:
        return False, "User not found."
    del st.session_state.users[username]
    return True, f"User '{username}' deleted."

def reset_password(username, new_password):
    if username not in st.session_state.users:
        return False, "User not found."
    st.session_state.users[username]["password"] = new_password
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

    # Tabs for user management, Telegram settings, etc. (same as your code)...
    # [Keep your full admin dashboard logic here]

# -------------------------
# Stroke App Main UI (User)
# -------------------------
def render_user_app():
    st.title("🧠 Stroke Detection from CT/MRI Scans")
    st.write("Upload a brain scan image to check stroke probability and view affected regions.")
    # [Keep your full patient details, upload, prediction, Telegram reporting logic here]

    with st.sidebar:
        st.header("👤 Account")
        st.write(f"Logged in as: {st.session_state.username} ({st.session_state.role})")
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
