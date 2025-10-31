import streamlit as st
import numpy as np
import cv2
import json
import requests
import os
import gdown
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# New imports for evaluation & plotting
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import io
import base64

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
        ["👤 Create User", "🧑‍🤝‍🧑 Manage Users", "📤 Export/Import", "📨 Telegram Settings"]
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

    # --- NEW: Model Evaluation Panel for Admin ---
    render_model_evaluation_panel()


# -------------------------
# Stroke App Main UI
# -------------------------
def render_user_app():
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

    st.write("---")
    if st.button("🩺 Book Doctor Appointment", key="book_appointment_btn"):
        st.session_state.show_appt_form = True
        st.rerun()

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
                f"👤 {a['patient_name']} | 🩺 {a['doctor']} | 🗓 {a['date']} at {a['time']} → *{color}*"
            )

    with st.sidebar:
        st.header("👤 Account")
        st.write(f"Logged in as: {st.session_state.username} ({st.session_state.role})")
        if st.button("🚪 Logout", key="user_logout_btn"):
            logout()
            st.rerun()

    if st.session_state.get("show_appt_form", False):
        render_appointment_portal()

    # -------------------------
    # Post-Stroke Care & Lifestyle Recommendations
    # -------------------------
    render_post_stroke_care()

    # --- NEW: Model Evaluation Panel for Users (can be hidden to admin only if desired) ---
    render_model_evaluation_panel()


# -------------------------
# Doctor Appointment Portal (User Side)
# -------------------------
def render_appointment_portal():
    st.title("🩺 Doctor Appointment Booking")
    st.write("Book an appointment with a neurologist or radiologist for consultation.")

    with st.form(key="appointment_form", clear_on_submit=False):
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
        cancel = st.form_submit_button("✖ Cancel")

        if submit:
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
            st.session_state.show_appt_form = False
            st.rerun()
        if cancel:
            st.session_state.show_appt_form = False
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
            st.write(f"📋 Status: {appt.get('status', 'Pending')}")
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                if st.button(f"✅ Approve_{idx}", key=f"approve_{idx}"):
                    st.session_state.appointments[idx]["status"] = "Approved"
                    save_appointments_to_file()
                    st.success(f"Appointment approved for {appt['patient_name']}")
                    st.rerun()
            with col2:
                if st.button(f"❌ Reject_{idx}", key=f"reject_{idx}"):
                    st.session_state.appointments[idx]["status"] = "Rejected"
                    save_appointments_to_file()
                    st.error(f"Appointment rejected for {appt['patient_name']}")
                    st.rerun()
            with col3:
                if st.button(f"🗑 Delete_{idx}", key=f"delete_{idx}"):
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
        "Here are some important suggestions:"
    )

    with st.expander("🥗 Nutrition & Foods"):
        st.markdown("""
        - *Fruits & Vegetables:* Fresh fruits (berries, oranges, apples) and leafy greens.  
        - *Whole Grains:* Brown rice, oats, whole wheat bread.  
        - *Proteins:* Fish rich in Omega-3 (salmon, sardines), eggs, legumes.  
        - *Nuts & Seeds:* Almonds, walnuts, flaxseeds — in moderation.  
        - *Limit:* Salt, fried foods, processed foods, and sugary snacks.  
        - *Hydration:* Drink plenty of water and natural juices (avoid added sugar).
        """)

    with st.expander("🧘 Physical & Mental Exercises"):
        st.markdown("""
        - *Yoga & Stretching:* Gentle yoga and flexibility exercises to improve mobility.  
        - *Walking & Aerobics:* Short walks, light aerobic exercises as tolerated.  
        - *Balance & Coordination Exercises:* Helps prevent falls.  
        - *Breathing Exercises / Pranayama:* Enhances oxygenation and reduces stress.  
        - *Meditation & Mindfulness:* Supports mental health, reduces anxiety and depression.
        """)

    with st.expander("💊 Lifestyle & Habits"):
        st.markdown("""
        - *Sleep:* Maintain regular sleep cycles (7–8 hours).  
        - *Stress Management:* Meditation, counseling, music therapy.  
        - *Regular Check-ups:* Monitor blood pressure, cholesterol, and blood sugar.  
        - *Avoid Smoking & Alcohol:* Critical for stroke prevention and recovery.  
        - *Follow Doctor’s Advice:* Stick to prescribed medications and rehabilitation programs.
        """)

    with st.expander("📚 Additional Tips"):
        st.markdown("""
        - Keep a *recovery journal* for diet, exercise, and mood tracking.  
        - Engage in *social support*: family, stroke support groups.  
        - Stay *mentally active*: puzzles, reading, cognitive exercises.  
        - *Small, consistent steps*: recovery is gradual; consistency matters.
        """)


# -------------------------
# NEW: Model Evaluation Panel (Confusion Matrix + ROC + Metrics)
# -------------------------
def _plot_confusion_matrix(cm, labels):
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title("Confusion Matrix")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(labels))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(labels, rotation=45)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(labels)
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    fig.tight_layout()
    return fig


def _plot_roc(fpr, tpr, roc_auc):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle='--', lw=1)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def _fig_to_download_link(fig, filename="plot.png"):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    href = f'<a href="data:file/png;base64,{b64}" download="{filename}">📥 Download {filename}</a>'
    return href


def render_model_evaluation_panel():
    st.subheader("📊 Model Evaluation (Confusion Matrix & ROC)")

    st.markdown(
        "Upload a CSV with columns **`true_label`** (0 or 1) and **`pred_prob`** (probability of stroke between 0 and 1). "
        "Then choose a threshold to compute the confusion matrix and metrics. "
        "If you don't have a CSV, you can create one from your test set predictions."
    )

    uploaded = st.file_uploader("Upload evaluation CSV", type=["csv"], key="eval_csv_uploader")
    sample_col1, sample_col2 = st.columns([1, 3])
    with sample_col1:
        if st.button("Show sample CSV"):
            st.code("true_label,pred_prob\n0,0.12\n1,0.87\n0,0.33\n1,0.92\n...", language="text")
    if uploaded is None:
        st.info("No evaluation CSV uploaded. Upload to compute metrics.")
        return

    try:
        import pandas as pd

        df = pd.read_csv(uploaded)
        if "true_label" not in df.columns or "pred_prob" not in df.columns:
            st.error("CSV must contain 'true_label' and 'pred_prob' columns.")
            return

        y_true = df["true_label"].astype(int).values
        y_prob = df["pred_prob"].astype(float).values

        # threshold slider
        thresh = st.slider("Probability threshold for classification", min_value=0.0, max_value=1.0, value=0.5, step=0.01, key="eval_thresh")

        y_pred = (y_prob >= thresh).astype(int)

        # metrics
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        cm = confusion_matrix(y_true, y_pred)
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)

        col1, col2 = st.columns([1, 1])
        with col1:
            st.metric("Accuracy", f"{acc:.3f}")
            st.metric("Precision", f"{prec:.3f}")
            st.metric("Recall", f"{rec:.3f}")
            st.metric("F1 Score", f"{f1:.3f}")

        with col2:
            st.write(f"AUC: **{roc_auc:.3f}**")
            st.write(f"Threshold: **{thresh:.2f}**")
            st.write("---")

        # Plots
        fig_cm = _plot_confusion_matrix(cm, labels=["No Stroke (0)", "Stroke (1)"])
        st.pyplot(fig_cm)
        st.markdown(_fig_to_download_link(fig_cm, filename="confusion_matrix.png"), unsafe_allow_html=True)

        fig_roc = _plot_roc(fpr, tpr, roc_auc)
        st.pyplot(fig_roc)
        st.markdown(_fig_to_download_link(fig_roc, filename="roc_curve.png"), unsafe_allow_html=True)

        # Optional: show raw table
        if st.expander("Show uploaded data"):
            st.dataframe(df.head(200))

    except Exception as e:
        st.error(f"Error processing evaluation file: {e}")


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
