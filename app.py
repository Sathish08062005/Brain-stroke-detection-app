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
# Footer with "Created By Sathish"
# -------------------------
st.markdown(
    """
    <style>
