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
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import seaborn as sns
import pandas as pd
import base64  # ← ADD THIS IMPORT

# SIMPLE BACKGROUND FALLBACK
try:
    with open("1234.jpg", "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("data:image/jpeg;base64,{b64_encoded}");
            background-size: cover;
        }}
        .main .block-container {{
            background: rgba(255,255,255,0.95);
            border-radius: 10px;
            padding: 2rem;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
except Exception as e:
    st.sidebar.warning(f"Background not loaded: {e}")

# -------------------------
# Users & Appointments file for persistence
# -------------------------
USERS_FILE = "users.json"
APPOINTMENTS_FILE = "appointments.json"  # persistent storage for appointments
VITAL_SIGNS_FILE = "vital_signs.json"   # persistent storage for vital signs


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


# Vital Signs persistence helpers
def save_vital_signs_to_file():
    try:
        with open(VITAL_SIGNS_FILE, "w") as f:
            json.dump(st.session_state.vital_signs, f, indent=2)
    except Exception as e:
        st.error(f"Error saving vital signs file: {e}")


def load_vital_signs_from_file():
    if os.path.exists(VITAL_SIGNS_FILE):
        try:
            with open(VITAL_SIGNS_FILE, "r") as f:
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
# Model Evaluation Metrics Functions
# -------------------------
def generate_model_metrics():
    """
    Generate sample model evaluation metrics for demonstration.
    In a real application, you would use your test dataset here.
    """
    # Sample data for demonstration - replace with actual test data
    np.random.seed(42)
    
    # Simulate predictions and true labels for 100 samples
    y_true = np.random.choice([0, 1], size=100, p=[0.3, 0.7])  # 30% negative, 70% positive
    y_pred_proba = np.random.rand(100)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Calculate metrics
    cm = confusion_matrix(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    # Calculate additional metrics
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision_score = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_score = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision_score * recall_score) / (precision_score + recall_score) if (precision_score + recall_score) > 0 else 0
    
    return {
        'confusion_matrix': cm,
        'fpr': fpr,
        'tpr': tpr,
        'roc_auc': roc_auc,
        'precision_curve': precision,
        'recall_curve': recall,
        'pr_auc': pr_auc,
        'accuracy': accuracy,
        'precision': precision_score,
        'recall': recall_score,
        'f1_score': f1_score,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
    }


def plot_confusion_matrix(cm):
    """Plot confusion matrix"""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['No Stroke', 'Stroke'],
                yticklabels=['No Stroke', 'Stroke'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    return fig


def plot_roc_curve(fpr, tpr, roc_auc):
    """Plot ROC curve"""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc="lower right")
    return fig


def plot_precision_recall_curve(precision, recall, pr_auc):
    """Plot Precision-Recall curve"""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc="upper right")
    return fig


def display_model_metrics():
    """Display all model evaluation metrics"""
    st.subheader("📊 Model Evaluation Metrics")
    
    # Generate metrics
    metrics = generate_model_metrics()
    
    # Display key metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", f"{metrics['accuracy']:.2%}")
    with col2:
        st.metric("Precision", f"{metrics['precision']:.2%}")
    with col3:
        st.metric("Recall", f"{metrics['recall']:.2%}")
    with col4:
        st.metric("F1-Score", f"{metrics['f1_score']:.2%}")
    
    # Display confusion matrix details
    st.write("Confusion Matrix Details:")
    cm_col1, cm_col2, cm_col3, cm_col4 = st.columns(4)
    with cm_col1:
        st.metric("True Positives", metrics['tp'])
    with cm_col2:
        st.metric("True Negatives", metrics['tn'])
    with cm_col3:
        st.metric("False Positives", metrics['fp'])
    with cm_col4:
        st.metric("False Negatives", metrics['fn'])
    
    # Plot metrics
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(plot_confusion_matrix(metrics['confusion_matrix']))
    with col2:
        st.pyplot(plot_roc_curve(metrics['fpr'], metrics['tpr'], metrics['roc_auc']))
    
    # Precision-Recall curve
    st.pyplot(plot_precision_recall_curve(metrics['precision_curve'], metrics['recall_curve'], metrics['pr_auc']))
    
    # Model performance summary
    st.subheader("📈 Performance Summary")
    performance_data = {
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC', 'PR AUC'],
        'Value': [
            f"{metrics['accuracy']:.2%}",
            f"{metrics['precision']:.2%}",
            f"{metrics['recall']:.2%}",
            f"{metrics['f1_score']:.2%}",
            f"{metrics['roc_auc']:.2%}",
            f"{metrics['pr_auc']:.2%}"
        ]
    }
    st.table(pd.DataFrame(performance_data))


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
    if "vital_signs" not in st.session_state:
        st.session_state.vital_signs = load_vital_signs_from_file()


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

    # Updated tabs for admin with separate appointment management tab
    tabs = st.tabs(
        ["👤 Create User", "🧑‍🤝‍🧑 Manage Users", "📤 Export/Import", "📨 Telegram Settings", "🩺 Appointment Requests"]
    )

    with tabs[0]:
        st.subheader("Create a new user")
        new_username = st.text_input("New Username", key="new_username")
        new_password = st.text_input("New Password", type="password", key="new_user_password")
        role = st.selectbox("Role", ["user", "admin"], index=0, key="new_user_role")
        if st.button("Create User", key="create_user_btn"):
            ok, msg = add_user(new_username, new_password, role)
            (st.success if ok else st.error)(msg)

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

    # Doctor Appointment Management in separate tab
    with tabs[4]:
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
# Stroke App Main UI
# -------------------------
def render_user_app():
    # Use tabs for user interface
    tabs = st.tabs(["🧠 Stroke Detection", "📊 Vital Signs", "🩺 Book Appointment", "🌿 Post-Stroke Care"])
    
    # Tab 1: Stroke Detection
    with tabs[0]:
        render_stroke_detection()
    
    # Tab 2: Vital Signs
    with tabs[1]:
        render_vital_signs()
    
    # Tab 3: Book Appointment
    with tabs[2]:
        render_appointment_portal()
    
    # Tab 4: Post-Stroke Care
    with tabs[3]:
        render_post_stroke_care()
    
    # Sidebar (common for all tabs)
    with st.sidebar:
        st.header("👤 Account")
        st.write(f"Logged in as: {st.session_state.username} ({st.session_state.role})")
        if st.button("🚪 Logout", key="user_logout_btn"):
            logout()
            st.rerun()


# -------------------------
# Stroke Detection Tab Content
# -------------------------
def render_stroke_detection():
    st.title("🧠 Stroke Detection from CT/MRI Scans")
    st.write("Upload a brain scan image to check stroke probability and view affected regions.")

    col1, col2 = st.columns(2)
    with col1:
        patient_name = st.text_input("Patient Name", value="John Doe", key="user_patient_name")
        patient_age = st.number_input("Patient Age", min_value=1, max_value=120, value=45, key="user_patient_age")
        patient_gender = st.selectbox("Gender", ["Male", "Female", "Other"], key="user_patient_gender")
    with col2:
        patient_id = st.text_input("Patient ID / Hospital No.", value="P12345", key="user_patient_id")
        patient_contact = st.text_input("Patient Contact Number", value="9876543210", key="user_patient_contact")
        patient_address = st.text_area("Patient Address", value="Chennai, India", key="user_patient_address")

    st.write("---")

    st.sidebar.header("📞 Emergency Contact Settings")
    relative_name = st.sidebar.text_input("Relative Name", value="Brother", key="user_relative_name")
    relative_number = st.sidebar.text_input("Relative Phone Number", value="9025845243", key="user_relative_number")

    uploaded_file = st.file_uploader("📤 Upload CT/MRI Image", type=["jpg", "png", "jpeg"], key="upload_scan")

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        st.image(image, caption="🖼 Uploaded Scan", use_column_width=True)

        stroke_prob, no_stroke_prob = classify_image(image)
        stroke_percent = stroke_prob * 100
        no_stroke_percent = no_stroke_prob * 100

        st.subheader("🧾 Patient Information")
        st.write(f"Name: {patient_name}")
        st.write(f"Age: {patient_age}")
        st.write(f"Gender: {patient_gender}")
        st.write(f"Patient ID: {patient_id}")
        st.write(f"Contact: {patient_contact}")
        st.write(f"Address: {patient_address}")

        st.subheader("🔍 Prediction Result:")
        st.write(f"🩸 Stroke Probability: {stroke_percent:.2f}%")
        st.write(f"✅ No Stroke Probability: {no_stroke_percent:.2f}%")

        if stroke_percent > 80:
            st.error("🔴 Immediate attention needed — very high stroke risk!")
            st.warning("⏱ Suggested Action: Seek emergency care within 1–3 hours.")
            st.markdown("📞 Emergency Call: [Call 108 (India)](tel:108)")
            st.markdown(f"📞 Call {relative_name}: [Call {relative_number}](tel:{relative_number})")
        elif 60 < stroke_percent <= 80:
            st.warning("🟠 Moderate to high stroke risk — medical consultation advised.")
            st.info("⏱ Suggested Action: Get hospital check-up within 6 hours.")
            st.markdown("📞 Emergency Call: [Call 108 (India)](tel:108)")
            st.markdown(f"📞 Call {relative_name}: [Call {relative_number}](tel:{relative_number})")
        elif 50 < stroke_percent <= 60:
            st.info("🟡 Slightly above normal stroke risk — further monitoring suggested.")
            st.info("⏱ Suggested Action: Visit a doctor within 24 hours.")
            st.markdown(f"📞 Call {relative_name}: [Call {relative_number}](tel:{relative_number})")
        elif no_stroke_percent > 90:
            st.success("🟢 Very low stroke risk — scan looks healthy.")
            st.info("⏱ Suggested Action: Routine monitoring only.")
        elif 70 < no_stroke_percent <= 90:
            st.info("🟡 Low stroke risk — but caution advised if symptoms exist.")
            st.info("⏱ Suggested Action: Consult a doctor if symptoms appear.")
            st.markdown(f"📞 Call {relative_name}: [Call {relative_number}](tel:{relative_number})")

        if stroke_prob > 0.5:
            marked_image = highlight_stroke_regions(image)
            st.image(marked_image, caption="🩸 Stroke Regions Highlighted", use_column_width=True)

        # Display Model Evaluation Metrics
        display_model_metrics()

        if st.button("💾 Save & Send to Telegram", key="send_telegram_btn"):
            BOT_TOKEN = st.session_state.settings.get("BOT_TOKEN", "")
            CHAT_ID = st.session_state.settings.get("CHAT_ID", "")

            message = (
                "🧾 Patient Stroke Report\n\n"
                f"👤 Name: {patient_name}\n"
                f"🎂 Age: {patient_age}\n"
                f"⚧ Gender: {patient_gender}\n"
                f"🆔 Patient ID: {patient_id}\n"
                f"📞 Contact: {patient_contact}\n"
                f"🏠 Address: {patient_address}\n\n"
                f"🩸 Stroke Probability: {stroke_percent:.2f}%\n"
                f"✅ No Stroke Probability: {no_stroke_percent:.2f}%"
            )
            url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
            try:
                response = requests.post(url, data={"chat_id": CHAT_ID, "text": message})
                if response.status_code == 200:
                    st.success("✅ Patient report sent to Telegram successfully!")
                    st.session_state.report_log.append(
                        {
                            "patient_name": patient_name,
                            "stroke_percent": stroke_percent,
                            "no_stroke_percent": no_stroke_percent,
                            "by": st.session_state.username or "unknown",
                        }
                    )
                else:
                    st.error("❌ Failed to send report to Telegram.")
            except Exception as e:
                st.error(f"❌ Error sending to Telegram: {e}")


# -------------------------
# Vital Signs Tab Content
# -------------------------
def render_vital_signs():
    st.title("📊 Adult Vital Signs Monitoring")
    st.write("Enter your vital signs data to monitor your health status.")
    
    # Display normal ranges reference
    st.subheader("📋 Normal Vital Signs Ranges")
    
    # Create a table for normal ranges
    normal_ranges = {
        "Vital Sign": [
            "Heart Rate (Pulse)",
            "Temperature", 
            "Respiratory Rate",
            "Blood Pressure (Systolic)",
            "Blood Pressure (Diastolic)", 
            "SpO2 (Oxygen Saturation)"
        ],
        "Normal Range": [
            "60 - 100 beats per minute",
            "97 - 99°F (36.1-37.2°C)",
            "12 - 20 breaths per minute", 
            "90 - 120 mmHg",
            "60 - 80 mmHg",
            "95 - 100%"
        ]
    }
    
    st.table(normal_ranges)
    
    st.write("---")
    
    # Vital Signs Input Form
    st.subheader("🩺 Enter Your Vital Signs")
    
    with st.form(key="vital_signs_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        
        with col1:
            patient_name = st.text_input("Patient Name", value="John Doe", key="vital_patient_name")
            heart_rate = st.number_input("Heart Rate (bpm)", min_value=0, max_value=200, value=72, key="heart_rate")
            temperature = st.number_input("Temperature (°F)", min_value=90.0, max_value=110.0, value=98.6, step=0.1, key="temperature")
            respiratory_rate = st.number_input("Respiratory Rate (breaths/min)", min_value=0, max_value=60, value=16, key="respiratory_rate")
            
        with col2:
            systolic_bp = st.number_input("Systolic BP (mmHg)", min_value=0, max_value=300, value=120, key="systolic_bp")
            diastolic_bp = st.number_input("Diastolic BP (mmHg)", min_value=0, max_value=200, value=80, key="diastolic_bp")
            oxygen_saturation = st.number_input("Oxygen Saturation (%)", min_value=0, max_value=100, value=98, key="oxygen_saturation")
            notes = st.text_area("Additional Notes", placeholder="Any symptoms or concerns...", key="vital_notes")
        
        submit_button = st.form_submit_button("💾 Save Vital Signs")
        
        if submit_button:
            if not patient_name:
                st.error("Please enter patient name.")
            else:
                # Check for abnormal values and provide warnings
                warnings = []
                
                if heart_rate < 60 or heart_rate > 100:
                    warnings.append(f"⚠ Heart rate ({heart_rate} bpm) is outside normal range (60-100 bpm)")
                
                if temperature < 97 or temperature > 99:
                    warnings.append(f"⚠ Temperature ({temperature}°F) is outside normal range (97-99°F)")
                
                if respiratory_rate < 12 or respiratory_rate > 20:
                    warnings.append(f"⚠ Respiratory rate ({respiratory_rate} breaths/min) is outside normal range (12-20 breaths/min)")
                
                if systolic_bp < 90 or systolic_bp > 120:
                    warnings.append(f"⚠ Systolic BP ({systolic_bp} mmHg) is outside normal range (90-120 mmHg)")
                
                if diastolic_bp < 60 or diastolic_bp > 80:
                    warnings.append(f"⚠ Diastolic BP ({diastolic_bp} mmHg) is outside normal range (60-80 mmHg)")
                
                if oxygen_saturation < 95:
                    warnings.append(f"🚨 Oxygen saturation ({oxygen_saturation}%) is below normal range (95-100%) - Seek medical attention!")
                
                # Save vital signs data
                vital_data = {
                    "patient_name": patient_name,
                    "heart_rate": heart_rate,
                    "temperature": temperature,
                    "respiratory_rate": respiratory_rate,
                    "systolic_bp": systolic_bp,
                    "diastolic_bp": diastolic_bp,
                    "oxygen_saturation": oxygen_saturation,
                    "notes": notes,
                    "timestamp": str(pd.Timestamp.now()),
                    "recorded_by": st.session_state.username or "unknown"
                }
                
                st.session_state.vital_signs.append(vital_data)
                save_vital_signs_to_file()
                
                st.success("✅ Vital signs saved successfully!")
                
                # Display warnings if any
                if warnings:
                    st.warning("Health Alerts:")
                    for warning in warnings:
                        st.write(warning)
                
                # Show summary
                st.subheader("📈 Current Reading Summary")
                summary_cols = st.columns(3)
                with summary_cols[0]:
                    st.metric("Heart Rate", f"{heart_rate} bpm")
                    st.metric("Temperature", f"{temperature}°F")
                with summary_cols[1]:
                    st.metric("Respiratory Rate", f"{respiratory_rate}/min")
                    st.metric("Oxygen Saturation", f"{oxygen_saturation}%")
                with summary_cols[2]:
                    st.metric("Blood Pressure", f"{systolic_bp}/{diastolic_bp}")
    
    # Display previous vital signs records
    st.write("---")
    st.subheader("📋 Previous Vital Signs Records")
    
    user_vitals = [
        v for v in st.session_state.vital_signs 
        if v.get("recorded_by") == st.session_state.username
    ]
    
    if not user_vitals:
        st.info("No vital signs records yet.")
    else:
        # Show latest 5 records
        recent_vitals = user_vitals[::-1][:5]
        
        for i, vital in enumerate(recent_vitals):
            with st.expander(f"Record {i+1} - {vital['timestamp'][:16]}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"Patient: {vital['patient_name']}")
                    st.write(f"Heart Rate: {vital['heart_rate']} bpm")
                    st.write(f"Temperature: {vital['temperature']}°F")
                    st.write(f"Respiratory Rate: {vital['respiratory_rate']}/min")
                with col2:
                    st.write(f"Blood Pressure: {vital['systolic_bp']}/{vital['diastolic_bp']} mmHg")
                    st.write(f"Oxygen Saturation: {vital['oxygen_saturation']}%")
                    if vital.get('notes'):
                        st.write(f"Notes: {vital['notes']}")


# -------------------------
# Doctor Appointment Portal (User Side)
# -------------------------
def render_appointment_portal():
    st.title("🩺 Doctor Appointment Booking")
    st.write("Book an appointment with a neurologist or radiologist for consultation.")

    # Show current appointment status for this user
    st.write("### 📅 Your Appointment Requests")
    user_appts = [
        a for a in st.session_state.appointments if a.get("requested_by") == st.session_state.username
    ]
    if not user_appts:
        st.info("No appointment requests yet.")
    else:
        for a in user_appts[::-1]:
            status = a.get("status", "Pending")
            color = "🔴 Rejected" if status == "Rejected" else (
                "🟢 Approved" if status == "Approved" else "🟡 Pending"
            )
            st.write(
                f"👤 {a['patient_name']} | 🩺 {a['doctor']} | 🗓 {a['date']} at {a['time']} → {color}"
            )
    
    st.write("---")
    st.subheader("📝 Book New Appointment")

    with st.form(key="appointment_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            appt_patient_name = st.text_input("Patient Name", value="John Doe", key="appt_patient_name")
            appt_mobile = st.text_input("Mobile Number", value="9876543210", key="appt_patient_mobile")
            appt_age = st.number_input("Age", min_value=1, max_value=120, value=45, key="appt_patient_age")
        with col2:
            appt_date = st.date_input("Appointment Date", key="appt_date")
            appt_time = st.time_input("Preferred Time", key="appt_time")
            doctor = st.selectbox(
                "Select Doctor",
                [
                    "Dr. Ramesh (Neurologist, Apollo)",
                    "Dr. Priya (Radiologist, Fortis)",
                    "Dr. Kumar (Stroke Specialist, MIOT)",
                    "Dr. Divya (CT Analysis Expert, Kauvery)",
                ],
                key="appt_doctor",
            )
        submit = st.form_submit_button("📩 Send Appointment Request")

        if submit:
            if not appt_patient_name or not appt_mobile:
                st.error("Please fill in all required fields.")
            else:
                appt = {
                    "patient_name": appt_patient_name,
                    "mobile": appt_mobile,
                    "age": appt_age,
                    "date": str(appt_date),
                    "time": str(appt_time),
                    "doctor": doctor,
                    "status": "Pending",
                    "requested_by": st.session_state.username or "unknown",
                }
                st.session_state.appointments.append(appt)
                save_appointments_to_file()
                st.success("✅ Appointment request sent to Admin for approval.")
                st.rerun()


# -------------------------
# Admin: Manage Doctor Appointments (color-coded buttons)
# -------------------------
def render_admin_appointments():
    st.subheader("🩺 Doctor Appointment Requests")
    if not st.session_state.appointments:
        st.info("No appointment requests yet.")
        return

    for idx, appt in enumerate(st.session_state.appointments):
        container = st.container()
        with container:
            st.write(f"Patient: {appt['patient_name']} ({appt.get('age', '')} yrs)")
            st.write(f"📞 {appt['mobile']} | 🩺 {appt['doctor']}")
            st.write(f"🗓 {appt['date']} at {appt['time']}")
            st.write(f"🧑‍💻 Requested by: {appt.get('requested_by', 'unknown')}")
            
            # Status with color coding
            status = appt.get('status', 'Pending')
            if status == 'Approved':
                st.success(f"📋 Status: {status}")
            elif status == 'Rejected':
                st.error(f"📋 Status: {status}")
            else:
                st.warning(f"📋 Status: {status}")
            
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                if st.button(f"✅ Approve", key=f"approve_{idx}"):
                    st.session_state.appointments[idx]["status"] = "Approved"
                    save_appointments_to_file()
                    st.success(f"Appointment approved for {appt['patient_name']}")
                    st.rerun()
            with col2:
                if st.button(f"❌ Reject", key=f"reject_{idx}"):
                    st.session_state.appointments[idx]["status"] = "Rejected"
                    save_appointments_to_file()
                    st.error(f"Appointment rejected for {appt['patient_name']}")
                    st.rerun()
            with col3:
                if st.button(f"🗑 Delete", key=f"delete_{idx}"):
                    removed = st.session_state.appointments.pop(idx)
                    save_appointments_to_file()
                    st.info(f"Deleted appointment for {removed['patient_name']}")
                    st.rerun()
            st.write("---")


# -------------------------
# Post-Stroke Care Recommendations Function
# -------------------------
def render_post_stroke_care():
    st.title("🌿 Post-Stroke Care & Lifestyle Recommendations")
    st.write(
        "After a brain stroke, recovery is not just medical treatment — lifestyle and diet play a major role. "
        "Here are your daily care recommendations:"
    )

    # Custom CSS for the box styling
    st.markdown("""
    <style>
    .recommendation-box {
        border: 2px solid #4CAF50;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        background-color: #f9f9f9;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .recommendation-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
    }
    .recommendation-title {
        font-weight: bold;
        color: #2E7D32;
        font-size: 18px;
    }
    .recommendation-time {
        color: #666;
        font-size: 14px;
        background-color: #E8F5E8;
        padding: 4px 8px;
        border-radius: 15px;
    }
    .recommendation-content {
        color: #333;
        line-height: 1.6;
    }
    </style>
    """, unsafe_allow_html=True)

    # Nutrition & Foods
    st.markdown("""
    <div class="recommendation-box">
        <div class="recommendation-header">
            <div class="recommendation-title">🥗 Fruits & Vegetables</div>
            <div class="recommendation-time">10:48am</div>
        </div>
        <div class="recommendation-content">
            • Fresh fruits (berries, oranges, apples)<br>
            • Leafy greens and colorful vegetables<br>
            • Limit salt and processed foods<br>
            • Drink plenty of water
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Physical Exercise
    st.markdown("""
    <div class="recommendation-box">
        <div class="recommendation-header">
            <div class="recommendation-title">💪 Physical Exercise</div>
            <div class="recommendation-time">02:30pm</div>
        </div>
        <div class="recommendation-content">
            • Gentle yoga and stretching<br>
            • Short walks daily<br>
            • Balance exercises<br>
            • Breathing exercises
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Mental Health
    st.markdown("""
    <div class="recommendation-box">
        <div class="recommendation-header">
            <div class="recommendation-title">🧠 Mental Wellness</div>
            <div class="recommendation-time">04:15pm</div>
        </div>
        <div class="recommendation-content">
            • Meditation and mindfulness<br>
            • Cognitive exercises<br>
            • Social interaction<br>
            • Stress management
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Medication & Checkups
    st.markdown("""
    <div class="recommendation-box">
        <div class="recommendation-header">
            <div class="recommendation-title">💊 Medication Schedule</div>
            <div class="recommendation-time">08:00am & 08:00pm</div>
        </div>
        <div class="recommendation-content">
            • Take prescribed medications on time<br>
            • Regular blood pressure monitoring<br>
            • Weekly doctor consultations<br>
            • Follow rehabilitation program
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Sleep & Rest
    st.markdown("""
    <div class="recommendation-box">
        <div class="recommendation-header">
            <div class="recommendation-title">😴 Sleep & Rest</div>
            <div class="recommendation-time">10:00pm</div>
        </div>
        <div class="recommendation-content">
            • 7-8 hours of quality sleep<br>
            • Regular sleep schedule<br>
            • Relaxation techniques<br>
            • Avoid caffeine before bed
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Additional Tips
    st.markdown("""
    <div class="recommendation-box">
        <div class="recommendation-header">
            <div class="recommendation-title">📋 Daily Checklist</div>
            <div class="recommendation-time">All Day</div>
        </div>
        <div class="recommendation-content">
            • Monitor blood pressure twice daily<br>
            • Take medications as prescribed<br>
            • 30 minutes of light exercise<br>
            • Healthy meals with fruits/vegetables<br>
            • Stay hydrated (8 glasses water)<br>
            • Practice relaxation techniques
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.write("---")
    st.info("💡 Tip: Follow these recommendations consistently for better recovery outcomes. Adjust timings based on your personal schedule and doctor's advice.")


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

# -------------------------
# Footer with "created by Sathish"
# -------------------------
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        right: 0;
        padding: 10px;
        color: #FF69B4;
        font-size: 14px;
        font-weight: bold;
        background-color: transparent;
        z-index: 999;
    }
    </style>
    <div class="footer">
        created by Sathish
    </div>
    """,
    unsafe_allow_html=True
)
