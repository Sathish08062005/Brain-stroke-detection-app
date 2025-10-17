import streamlit as st

# -------------------------
# Initialize session state
# -------------------------
if "username" not in st.session_state:
    st.session_state.username = "JohnDoe"  # default user for testing

if "appointments" not in st.session_state:
    st.session_state.appointments = []

if "appt_temp" not in st.session_state:
    st.session_state.appt_temp = {
        "name": "John Doe",
        "mobile": "9876543210",
        "age": 45,
        "date": None,
        "time": None,
        "doctor": None
    }

# -------------------------
# Doctor Appointment Portal
# -------------------------
def render_appointment_portal():
    st.title("🩺 Doctor Appointment Booking")
    st.write("Book an appointment with a neurologist or radiologist for consultation.")

    col1, col2 = st.columns(2)
    with col1:
        st.session_state.appt_temp["name"] = st.text_input(
            "Patient Name", value=st.session_state.appt_temp["name"], key="appt_name"
        )
        st.session_state.appt_temp["mobile"] = st.text_input(
            "Mobile Number", value=st.session_state.appt_temp["mobile"], key="appt_mobile"
        )
        st.session_state.appt_temp["age"] = st.number_input(
            "Age", min_value=1, max_value=120, value=st.session_state.appt_temp["age"], key="appt_age"
        )
    with col2:
        st.session_state.appt_temp["date"] = st.date_input(
            "Appointment Date", value=st.session_state.appt_temp["date"] or st.session_state.appt_temp["date"], key="appt_date"
        )
        st.session_state.appt_temp["time"] = st.time_input(
            "Preferred Time", value=st.session_state.appt_temp["time"] or st.session_state.appt_temp["time"], key="appt_time"
        )

    st.session_state.appt_temp["doctor"] = st.selectbox(
        "Select Doctor",
        [
            "Dr. Ramesh (Neurologist, Apollo)",
            "Dr. Priya (Radiologist, Fortis)",
            "Dr. Kumar (Stroke Specialist, MIOT)",
            "Dr. Divya (CT Analysis Expert, Kauvery)",
        ],
        index=0,
        key="appt_doctor_select"
    )

    # Submit button triggers only when clicked
    if st.button("📩 Send Appointment Request", key="send_appt_btn2"):
        appointment = {
            "patient_name": st.session_state.appt_temp["name"],
            "mobile": st.session_state.appt_temp["mobile"],
            "age": st.session_state.appt_temp["age"],
            "date": str(st.session_state.appt_temp["date"]),
            "time": str(st.session_state.appt_temp["time"]),
            "doctor": st.session_state.appt_temp["doctor"],
            "status": "Pending",
            "requested_by": st.session_state.username,
        }
        st.session_state.appointments.append(appointment)
        st.success("✅ Appointment request sent to Admin for approval.")
        # Clear form values
        st.session_state.appt_temp = {
            "name": "John Doe",
            "mobile": "9876543210",
            "age": 45,
            "date": None,
            "time": None,
            "doctor": None
        }

    st.write("---")
    st.subheader("📋 All Appointment Requests")
    if st.session_state.appointments:
        for idx, appt in enumerate(st.session_state.appointments):
            st.write(f"**Patient:** {appt['patient_name']} ({appt['age']} yrs)")
            st.write(f"📞 {appt['mobile']} | 🩺 {appt['doctor']}")
            st.write(f"🗓 {appt['date']} at {appt['time']}")
            st.write(f"🧑‍💻 Requested by: {appt['requested_by']}")
            st.write(f"📋 Status: {appt['status']}")
            st.write("---")
    else:
        st.info("No appointment requests yet.")

# -------------------------
# Run the portal
# -------------------------
if __name__ == "__main__":
    render_appointment_portal()
