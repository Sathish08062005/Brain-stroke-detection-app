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
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, roc_curve, auc, classification_report

# Optional DICOM support
try:
    import pydicom
    HAS_PYDICOM = True
except Exception:
    HAS_PYDICOM = False


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


    # --- Confusion Matrix Tab (NEW) - admin-side CSV-based option retained for admins
    with tabs[4]:
        st.subheader("📊 Model Evaluation - Confusion Matrix (Admin)")
        uploaded_csv = st.file_uploader(
            "Upload Test Dataset CSV (image_path,label)", type=["csv"], key="conf_matrix_csv_admin"
        )
        if uploaded_csv is not None:
            try:
                import pandas as pd
                df = pd.read_csv(uploaded_csv)
                if {"image_path", "label"}.issubset(df.columns):
                    y_true = []
                    y_pred = []
                    y_prob = []
                    for _, row in df.iterrows():
                        img_path = row["image_path"]
                        label = row["label"]
                        if os.path.exists(img_path):
                            img = cv2.imread(img_path)
                            pred_stroke_prob, _ = classify_image(img)
                            pred_label = int(pred_stroke_prob > 0.5)
                            y_true.append(int(label))
                            y_pred.append(pred_label)
                            y_prob.append(pred_stroke_prob)
                    if len(y_true) > 0:
                        cm = confusion_matrix(y_true, y_pred)
                        fig, ax = plt.subplots(figsize=(5, 4))
                        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                                    xticklabels=["No Stroke", "Stroke"], yticklabels=["No Stroke", "Stroke"], ax=ax)
                        ax.set_xlabel("Predicted")
                        ax.set_ylabel("Actual")
                        st.pyplot(fig)

                        acc = accuracy_score(y_true, y_pred)
                        prec = precision_score(y_true, y_pred, zero_division=0)
                        rec = recall_score(y_true, y_pred, zero_division=0)
                        st.write(f"**✅ Accuracy:** {acc * 100:.2f}%")
                        st.write(f"**🎯 Precision:** {prec * 100:.2f}%")
                        st.write(f"**🔁 Recall:** {rec * 100:.2f}%")

                        # ROC
                        if len(y_prob) > 1 and len(set(y_true)) > 1:
                            fpr, tpr, _ = roc_curve(y_true, y_prob)
                            roc_auc = auc(fpr, tpr)
                            st.write("### 📈 ROC Curve")
                            fig2, ax2 = plt.subplots()
                            ax2.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
                            ax2.plot([0, 1], [0, 1], linestyle="--", color="gray")
                            ax2.set_xlabel("False Positive Rate")
                            ax2.set_ylabel("True Positive Rate")
                            ax2.legend()
                            st.pyplot(fig2)
                    else:
                        st.error("No valid images found from CSV paths.")
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

       # containers to collect evaluation information
    eval_image_names = []
    eval_y_true = []
    eval_y_pred = []
    eval_y_prob = []

    if uploaded_files:
        st.subheader("🩻 Uploaded Scans and Predictions")
        st.write("For each uploaded image: confirm the true label (if known) or leave the suggested label (from filename).")
        for uploaded_file in uploaded_files:
            filename = uploaded_file.name
            # load image bytes into cv2
            file_bytes = uploaded_file.read()
            img = None
            # DICOM handling
            if filename.lower().endswith(".dcm"):
                if HAS_PYDICOM:
                    try:
                        ds = pydicom.dcmread(st.binary_buffer(file_bytes))
                    except Exception:
                        # pydicom expects path-like or file-like; create from bytes
                        import io
                        ds = pydicom.dcmread(io.BytesIO(file_bytes))
                    try:
                        arr = ds.pixel_array
                        # normalize arr to 0-255
                        arr = arr.astype(np.float32)
                        arr -= arr.min()
                        if arr.max() != 0:
                            arr = arr / arr.max()
                        arr = (arr * 255).astype(np.uint8)
                        if arr.ndim == 2:
                            img = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
                        elif arr.ndim == 3:
                            # already multi-channel
                            img = arr
                        else:
                            img = cv2.cvtColor(arr[..., 0], cv2.COLOR_GRAY2BGR)
                    except Exception as e:
                        st.warning(f"Could not parse DICOM pixels for {filename}: {e}")
                        img = None
                else:
                    st.warning(f"DICOM file detected ({filename}) but pydicom is not installed. Skipping DICOM.")
                    img = None
            else:
                # regular image formats
                file_bytes_np = np.frombuffer(file_bytes, np.uint8)
                img = cv2.imdecode(file_bytes_np, cv2.IMREAD_COLOR)

            if img is None:
                st.error(f"Unable to read image: {filename}")
                continue

            # Show original
            cols = st.columns([1, 1])
            with cols[0]:
                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=f"Original: {filename}", use_column_width=True)
            # Predict
            stroke_prob, no_stroke_prob = classify_image(img)
            pred_label = 1 if stroke_prob > 0.5 else 0

            # Suggest a ground truth if filename contains keywords
            suggested_true = None
            lname = filename.lower()
            if "stroke" in lname or "yes" in lname or "1" in lname:
                suggested_true = 1
            elif "no" in lname or "normal" in lname or "healthy" in lname or "0" in lname:
                suggested_true = 0

            # let user confirm/set true label
            with cols[1]:
                st.write(f"**Prediction:** {'Stroke' if pred_label == 1 else 'No Stroke'}")
                st.write(f"**Stroke Probability:** {stroke_prob*100:.2f}%")
                if stroke_prob > 0.7:
                    st.error("⚠ High Stroke Probability Detected! Immediate medical attention recommended.")
                elif stroke_prob > 0.4:
                    st.warning("⚠ Moderate Stroke Risk — further diagnosis advised.")
                else:
                    st.success("✅ No Stroke Detected.")

                true_label = None
                if suggested_true is not None:
                    true_label = st.radio(
                        f"True label for {filename} (suggested)",
                        ("No Stroke", "Stroke"),
                        index=suggested_true,
                        key=f"true_{filename}"
                    )
                else:
                    true_label = st.radio(
                        f"True label for {filename}",
                        ("No Stroke", "Stroke"),
                        index=0,
                        key=f"true_{filename}"
                    )
                # convert to 0/1
                true_val = 1 if (true_label == "Stroke") else 0

                # show highlighted regions if predicted stroke
                if pred_label == 1:
                    marked = highlight_stroke_regions(img)
                    st.image(cv2.cvtColor(marked, cv2.COLOR_BGR2RGB), caption=f"Highlighted: {filename}", use_column_width=True)

                # Store eval info
                eval_image_names.append(filename)
                eval_y_true.append(true_val)
                eval_y_pred.append(pred_label)
                eval_y_prob.append(stroke_prob)

                st.markdown("---")

        # After processing all images: show evaluation metrics if more than one item
        if len(eval_y_true) >= 1:
            st.write("## 🧮 Evaluation (from uploaded batch)")
            # compute metrics
            y_true_arr = np.array(eval_y_true)
            y_pred_arr = np.array(eval_y_pred)
            y_prob_arr = np.array(eval_y_prob)

            # Basic metrics
            acc = accuracy_score(y_true_arr, y_pred_arr)
            prec = precision_score(y_true_arr, y_pred_arr, zero_division=0)
            rec = recall_score(y_true_arr, y_pred_arr, zero_division=0)

            cols_m = st.columns(3)
            cols_m[0].metric("Accuracy", f"{acc*100:.2f}%")
            cols_m[1].metric("Precision", f"{prec*100:.2f}%")
            cols_m[2].metric("Recall", f"{rec*100:.2f}%")

            # Confusion matrix
            cm = confusion_matrix(y_true_arr, y_pred_arr)
            fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Stroke", "Stroke"], yticklabels=["No Stroke", "Stroke"], ax=ax_cm)
            ax_cm.set_xlabel("Predicted")
            ax_cm.set_ylabel("Actual")
            st.pyplot(fig_cm)

            # ROC (only if there is both classes present)
            if len(set(y_true_arr)) > 1:
                fpr, tpr, _ = roc_curve(y_true_arr, y_prob_arr)
                roc_auc = auc(fpr, tpr)
                st.write("### ROC Curve")
                fig_roc, ax_roc = plt.subplots()
                ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
                ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray")
                ax_roc.set_xlabel("False Positive Rate")
                ax_roc.set_ylabel("True Positive Rate")
                ax_roc.legend()
                st.pyplot(fig_roc)
            else:
                st.info("ROC curve requires at least one example from each class (Stroke and No Stroke).")



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
# Main Routing
# -------------------------
if not st.session_state.logged_in:
    render_login()
else:
    if st.session_state.role == "admin":
        render_admin_dashboard()
    else:
        render_user_app()
