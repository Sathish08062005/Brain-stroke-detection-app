import streamlit as st
import numpy as np
import cv2
import json
import requests
import os
import gdown
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd

# -------------------------
# Users & Appointments file for persistence
# -------------------------
USERS_FILE = "users.json"
APPOINTMENTS_FILE = "appointments.json"  # persistent storage for appointments


def save_users_to_file():
    try:
        with open(USERS_FILE, "w") as f:
            json.dump(st.session_state.users, f, indent=2)
    except Exception as e:
        st.error(f"Error saving users file: {e}")


# Appointment persistence helpers
def save_appointments_to_file():
    try:
        with open(APPOINTMENTS_FILE, "w") as f:
            json.dump(st.session_state.appointments, f, indent=2)
    except Exception as e:
        st.error(f"Error saving appointments file: {e}")


def load_appointments_from_file():
    if os.path.exists(APPOINTMENTS_FILE):
        try:
            with open(APPOINTMENTS_FILE, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
        except Exception:
            return []
    return []


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
    username = st.text_input("Username", key="login_username")
    password = st.text_input("Password", type="password", key="login_password")
    colA, colB = st.columns([1, 1])
    with colA:
        if st.button("Login", use_container_width=True, key="login_btn"):
            if login(username, password):
                st.success("Login successful ✅")
                st.rerun()
            else:
                st.error("❌ Invalid Username or Password")
    with colB:
        st.caption("No registration here. Users must be created by the admin.")


# -------------------------
# Admin Dashboard
# -------------------------
def render_admin_dashboard():
    st.title("🛡 Admin Dashboard")
    st.write(f"Welcome, {st.session_state.username} (admin)")

    with st.sidebar:
        st.header("⚙ Admin Actions")
        if st.button("🚪 Logout", key="admin_logout_btn"):
            logout()
            st.rerun()

    tabs = st.tabs(
        ["👤 Create User", "🧑‍🤝‍🧑 Manage Users", "📤 Export/Import", "📨 Telegram Settings", "📊 Confusion Matrix"]
    )

    # --- Create User Tab
    with tabs[0]:
        st.subheader("Create a new user")
        new_username = st.text_input("New Username", key="new_username")
        new_password = st.text_input("New Password", type="password", key="new_user_password")
        role = st.selectbox("Role", ["user", "admin"], index=0, key="new_user_role")
        if st.button("Create User", key="create_user_btn"):
            ok, msg = add_user(new_username, new_password, role)
            (st.success if ok else st.error)(msg)

    # --- Manage Users Tab
    with tabs[1]:
        st.subheader("All Users")
        users = st.session_state.users
        if users:
            for uname, meta in users.items():
                cols = st.columns([2, 1, 2, 2])
                cols[0].write(f"{uname}")
                cols[1].write(meta["role"])
                with cols[2]:
                    new_pw = st.text_input(
                        f"New Password for {uname}", key=f"pw_{uname}", type="password"
                    )
                    if st.button(f"Reset Password: {uname}", key=f"btn_reset_{uname}"):
                        ok, msg = reset_password(uname, new_pw)
                        (st.success if ok else st.error)(msg)
                with cols[3]:
                    if st.button(f"Delete {uname}", key=f"btn_del_{uname}"):
                        ok, msg = delete_user(uname)
                        (st.success if ok else st.error)(msg)
        else:
            st.info("No users yet.")

    # --- Export/Import Tab
    with tabs[2]:
        st.subheader("Export / Import Users")
        st.download_button(
            "📥 Download users.json",
            data=export_users_json(),
            file_name="users.json",
            mime="application/json",
            key="download_users_btn",
        )
        up = st.file_uploader("Import users.json", type=["json"], key="import_users_uploader")
        if up is not None:
            ok, msg = import_users_json(up.read())
            (st.success if ok else st.error)(msg)

    # --- Telegram Settings Tab
    with tabs[3]:
        st.subheader("Telegram Settings")
        bot_token = st.text_input(
            "BOT_TOKEN", value=st.session_state.settings.get("BOT_TOKEN", ""), key="bot_token"
        )
        chat_id = st.text_input(
            "CHAT_ID", value=st.session_state.settings.get("CHAT_ID", ""), key="chat_id"
        )
        if st.button("Save Telegram Settings", key="save_telegram_btn"):
            st.session_state.settings["BOT_TOKEN"] = bot_token
            st.session_state.settings["CHAT_ID"] = chat_id
            st.success("Saved Telegram settings.")

    # --- Confusion Matrix Tab (NEW)
    with tabs[4]:
        st.subheader("📊 Model Evaluation - Confusion Matrix")
        uploaded_csv = st.file_uploader(
            "Upload Test Dataset CSV (image_path,label)", type=["csv"], key="conf_matrix_csv"
        )
        if uploaded_csv is not None:
            try:
                df = pd.read_csv(uploaded_csv)
                if {"image_path","label"}.issubset(df.columns):
                    y_true = []
                    y_pred = []
                    for _, row in df.iterrows():
                        img_path = row['image_path']
                        label = row['label']  # 0 = No Stroke, 1 = Stroke
                        if os.path.exists(img_path):
                            img = cv2.imread(img_path)
                            pred_stroke_prob, _ = classify_image(img)
                            pred_label = int(pred_stroke_prob > 0.5)
                            y_true.append(int(label))
                            y_pred.append(pred_label)
                    # Compute confusion matrix
                    cm = confusion_matrix(y_true, y_pred)
                    fig, ax = plt.subplots(figsize=(5,4))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["No Stroke","Stroke"], yticklabels=["No Stroke","Stroke"], ax=ax)
                    ax.set_xlabel("Predicted")
                    ax.set_ylabel("Actual")
                    st.pyplot(fig)

                    acc = np.mean(np.array(y_true) == np.array(y_pred))
                    st.info(f"✅ Model Accuracy on Test Set: {acc*100:.2f}%")
                else:
                    st.error("CSV must contain 'image_path' and 'label' columns.")
            except Exception as e:
                st.error(f"Error reading CSV: {e}")

    # Doctor Appointment Management (admin view)
    with st.expander("🩺 View Doctor Appointments"):
        render_admin_appointments()

    st.divider()
    st.subheader("📝 Recently Sent Reports")
    if st.session_state.report_log:
        for i, r in enumerate(st.session_state.report_log[::-1][:10], 1):
            st.write(
                f"{i}. {r.get('patient_name','')} | Stroke: {r.get('stroke_percent',''):.2f}% | No Stroke: {r.get('no_stroke_percent',''):.2f}% | By: {r.get('by','')}"
            )
    else:
        st.caption("No reports yet.")


# -------------------------
# Main Routing
# -------------------------
if not st.session_state.logged_in:
    render_login()
else:
    if st.session_state.role == "admin":
        render_admin_dashboard()
    else:
        # Load your existing user app code unchanged
        from user_app import render_user_app
        render_user_app()
