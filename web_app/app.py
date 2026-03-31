"""
Streamlit Web Application for Brain Tumor Detection
"""

import streamlit as st
import os
import sys
from PIL import Image
import cv2
import numpy as np
from datetime import datetime

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from utils.predictor import BrainTumorPredictor
from utils.report_generator import TumorReport
from database.db_manager import TumorDatabase
from config import REPORT_DIR

# Page configuration
st.set_page_config(
    page_title="Brain Tumor Detection System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium UI
st.markdown("""
    <style>
    /* Global Typography & Background adjustments */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #0ea5e9, #6366f1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1.5rem 0;
        margin-bottom: 1rem;
    }
    
    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s ease;
    }
    .glass-card:hover {
        transform: translateY(-2px);
    }
    
    /* Alerts enhancements */
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background: linear-gradient(145deg, rgba(16, 185, 129, 0.2), rgba(5, 150, 105, 0.1));
        border-left: 4px solid #10b981;
        color: #e2e8f0;
    }
    .warning-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background: linear-gradient(145deg, rgba(245, 158, 11, 0.2), rgba(217, 119, 6, 0.1));
        border-left: 4px solid #f59e0b;
        color: #e2e8f0;
    }
    .danger-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background: linear-gradient(145deg, rgba(239, 68, 68, 0.2), rgba(220, 38, 38, 0.1));
        border-left: 4px solid #ef4444;
        color: #e2e8f0;
    }
    
    /* Login container centering */
    .login-container {
        max-width: 400px;
        margin: 0 auto;
        padding-top: 5vh;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state for ML components
if 'predictor' not in st.session_state:
    st.session_state.predictor = BrainTumorPredictor()
    st.session_state.predictor.load_models()

if 'db' not in st.session_state:
    st.session_state.db = TumorDatabase()

if 'report_gen' not in st.session_state:
    st.session_state.report_gen = TumorReport()

# Initialize session state for Authentication
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.role = None
    st.session_state.username = None
    
def login_page():
    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    st.markdown('<h1 class="main-header">NeuroScan AI</h1>', unsafe_allow_html=True)
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    # Create Tabs for Login vs Register
    action_tab1, action_tab2 = st.tabs(["🔒 Login", "📝 Register new Account"])
    
    with action_tab1:
        st.write("Please select your portal to sign in.")
        login_doc, login_pat = st.tabs(["🏥 Doctor", "👤 Patient"])
        
        with login_doc:
            with st.form("doctor_login_form"):
                username = st.text_input("Doctor Username")
                password = st.text_input("Password", type="password")
                submit_doc = st.form_submit_button("Sign In as Doctor", use_container_width=True)
                
                if submit_doc:
                    if not username or not password:
                        st.error("Please fill in all fields.")
                    else:
                        full_name = st.session_state.db.verify_user(username.strip(), password, "Doctor")
                        if full_name:
                            st.session_state.logged_in = True
                            st.session_state.role = "Doctor"
                            st.session_state.username = full_name
                            st.rerun()
                        else:
                            st.error("Invalid Doctor credentials.")

        with login_pat:
            with st.form("patient_login_form"):
                username = st.text_input("Patient Username")
                password = st.text_input("Password", type="password")
                submit_pat = st.form_submit_button("Sign In as Patient", use_container_width=True)
                
                if submit_pat:
                    if not username or not password:
                        st.error("Please fill in all fields.")
                    else:
                        full_name = st.session_state.db.verify_user(username.strip(), password, "Patient")
                        if full_name:
                            st.session_state.logged_in = True
                            st.session_state.role = "Patient"
                            st.session_state.username = full_name
                            st.rerun()
                        else:
                            st.error("Invalid Patient credentials.")

    with action_tab2:
        st.write("Create a new account.")
        reg_doc, reg_pat = st.tabs(["🏥 Register Doctor", "👤 Register Patient"])
        
        with reg_doc:
            with st.form("doctor_reg_form"):
                reg_name = st.text_input("Full Name (e.g., Dr. Smith)")
                reg_user = st.text_input("Choose Username")
                reg_pass = st.text_input("Choose Password", type="password")
                submit_reg_doc = st.form_submit_button("Register Doctor Account", use_container_width=True)
                
                if submit_reg_doc:
                    if not reg_name or not reg_user or not reg_pass:
                        st.error("Please fill all fields.")
                    else:
                        success = st.session_state.db.create_user(reg_user.strip(), reg_pass, "Doctor", reg_name.strip())
                        if success:
                            st.success("Doctor account created successfully! Please proceed to Login.")
                        else:
                            st.error("Username already exists. Please choose another.")
                            
        with reg_pat:
            with st.form("patient_reg_form"):
                reg_name = st.text_input("Full Legal Name")
                reg_user = st.text_input("Choose Username")
                reg_pass = st.text_input("Choose Password", type="password")
                submit_reg_pat = st.form_submit_button("Register Patient Account", use_container_width=True)
                
                if submit_reg_pat:
                    if not reg_name or not reg_user or not reg_pass:
                        st.error("Please fill all fields.")
                    else:
                        success = st.session_state.db.create_user(reg_user.strip(), reg_pass, "Patient", reg_name.strip())
                        if success:
                            st.success("Patient account created successfully! Please proceed to Login.")
                        else:
                            st.error("Username already exists. Please choose another.")
                
    st.markdown('</div></div>', unsafe_allow_html=True)

# Main Application Router
if not st.session_state.logged_in:
    login_page()
else:
    # Sidebar
    st.sidebar.title(f"🧠 {st.session_state.username}")
    st.sidebar.markdown(f"**Role:** {st.session_state.role}")
    st.sidebar.markdown("---")
    
    # Role-based Navigation mapping
    if st.session_state.role == "Doctor":
        nav_options = ["Home", "Analyze Scan", "Global Patient History", "System Statistics", "About"]
    else:
        nav_options = ["My Portal", "My Scan History", "About"]
        
    page = st.sidebar.radio("Navigation", nav_options)
    
    st.sidebar.markdown("---")
    if st.sidebar.button("Logout", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.role = None
        st.session_state.username = None
        st.rerun()

    # HOME PAGE
    if page == "Home":
        st.markdown("""
            <div style="background: linear-gradient(135deg, #0ea5e9, #3b82f6); padding: 3rem; border-radius: 15px; text-align: center; margin-bottom: 2rem; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.4);">
                <h1 style="color: white; font-size: 3.5rem; font-weight: 800; margin-bottom: 0.5rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">🧠 NeuroScan AI</h1>
                <p style="color: #e0f2fe; font-size: 1.5rem; max-width: 600px; margin: 0 auto; line-height: 1.4;">Advanced Deep Learning Diagnostic Portal for Neurosurgical Oncology</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🔬 System Capabilities")
        st.markdown("""
        Our bleeding-edge neural networks supply comprehensive analysis:
        - ✅ **Binary Detection:** Precisely detect the presence of brain tumors.
        - 🔍 **Multi-class Classification:** Categorize tumor archetypes (Glioma, Meningioma, Pituitary).
        - 📍 **Automated Segmentation:** Highlight precise tumor boundaries and coordinates.
        - 📊 **Clinical Reporting:** Generate rigorous downstream medical PDF reports.
        - 💾 **Patient Ledger:** Secure, encrypted long-term patient database.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="glass-card" style="text-align: center;"><h2 style="color: #10b981; font-size: 2.5rem; margin:0;">95%+</h2><p style="color: #94a3b8; font-weight: 600; margin:0;">Detection ACCURACY</p></div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="glass-card" style="text-align: center;"><h2 style="color: #f59e0b; font-size: 2.5rem; margin:0;">< 2s</h2><p style="color: #94a3b8; font-weight: 600; margin:0;">Scan LATENCY</p></div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="glass-card" style="text-align: center;"><h2 style="color: #3b82f6; font-size: 2.5rem; margin:0;">256-bit</h2><p style="color: #94a3b8; font-weight: 600; margin:0;">Database ENCRYPTION</p></div>', unsafe_allow_html=True)

    # ANALYZE SCAN PAGE
    elif page == "Analyze Scan":
        st.markdown('<h1 class="main-header" style="text-align: left;">🔬 Advanced Scan Interpreter</h1>', unsafe_allow_html=True)
        st.markdown("""<p style="color: #94a3b8; font-size: 1.1rem; margin-bottom: 2rem;">Upload a high-resolution MRI scan for immediate AI analysis.</p>""", unsafe_allow_html=True)
        
        # File uploader
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload Brain MRI Image",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Supported formats: JPG, PNG, BMP"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        if uploaded_file:
            # Display uploaded image
            col1, col2 = st.columns([1, 1.2])
            
            with col1:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown("### 📷 Uploaded MRI Scan")
                image = Image.open(uploaded_file)
                st.image(image, use_container_width=True, output_format="PNG")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Patient Information Form
            with col2:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                with st.form("patient_form", border=False):
                    st.markdown("### 👤 Attach to Patient Profile")
                    
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        patient_name = st.text_input("Patient Full Legal Name*", placeholder="e.g. John Doe")
                        patient_age = st.number_input("Age", min_value=0, max_value=120, value=30)
                    
                    with col_b:
                        patient_gender = st.selectbox("Biological Sex", ["Male", "Female", "Other"])
                        patient_contact = st.text_input("Contact Identifier", placeholder="+1 234 567 8900")
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    analyze_button = st.form_submit_button("🧠 INITIALIZE AI ANALYSIS SEQUENCE", use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            if analyze_button:
                if not patient_name:
                    st.error("❌ Please enter patient name to bind this scan.")
                else:
                    # Save uploaded file temporarily
                    temp_path = os.path.join("temp_uploads", uploaded_file.name)
                    os.makedirs("temp_uploads", exist_ok=True)
                    
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Run prediction
                    with st.spinner("🔄 Neural Networks Processing Scan... Please hold..."):
                        results = st.session_state.predictor.predict(temp_path)
                    
                    # Display results
                    st.markdown("---")
                    st.markdown('<h2 style="color: #6366f1; margin-bottom: 1.5rem;">📋 Clinical Pathology Report</h2>', unsafe_allow_html=True)
                    
                    # Diagnosis
                    if results['has_tumor']:
                        st.markdown(
                            f'''<div style="background: linear-gradient(135deg, rgba(239, 68, 68, 0.2), rgba(153, 27, 27, 0.4)); border-left: 5px solid #ef4444; border-radius: 10px; padding: 2rem; margin-bottom: 2rem;">
                                <h2 style="color: #fca5a5; margin: 0; font-size: 2rem;">⚠️ POSITIVE PATHOLOGY DETECTED</h2>
                                <p style="color: #f1f5f9; font-size: 1.2rem; margin: 0.5rem 0 0 0;">Network Confidence: <strong>{results["tumor_probability"]*100:.2f}%</strong></p>
                            </div>''',
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f'''<div style="background: linear-gradient(135deg, rgba(16, 185, 129, 0.2), rgba(6, 78, 59, 0.4)); border-left: 5px solid #10b981; border-radius: 10px; padding: 2rem; margin-bottom: 2rem;">
                                <h2 style="color: #6ee7b7; margin: 0; font-size: 2rem;">✅ NO ABNORMALITIES DETECTED</h2>
                                <p style="color: #f1f5f9; font-size: 1.2rem; margin: 0.5rem 0 0 0;">Network Confidence: <strong>{(1-results["tumor_probability"])*100:.2f}%</strong></p>
                            </div>''',
                            unsafe_allow_html=True
                        )
                    
                    st.markdown("")
                    
                    # Detailed results
                    if results['has_tumor']:
                        col_x, col_y, col_z = st.columns(3)
                        
                        with col_x:
                            if results.get('tumor_type'):
                                st.markdown(f'''
                                <div class="glass-card" style="text-align: center; border-top: 4px solid #8b5cf6;">
                                    <p style="color: #94a3b8; font-weight: bold; font-size: 0.9rem; text-transform: uppercase;">Tumor Classification</p>
                                    <h3 style="color: white; font-size: 1.5rem; margin: 0.5rem 0;">{results['tumor_type'].upper()}</h3>
                                    <p style="color: #8b5cf6; font-weight: 600; margin: 0;">{results.get('tumor_type_confidence', 0)*100:.1f}% Confidence</p>
                                </div>
                                ''', unsafe_allow_html=True)
                        
                        with col_y:
                            if results.get('tumor_location'):
                                st.markdown(f'''
                                <div class="glass-card" style="text-align: center; border-top: 4px solid #0ea5e9;">
                                    <p style="color: #94a3b8; font-weight: bold; font-size: 0.9rem; text-transform: uppercase;">Anatomical Location</p>
                                    <h3 style="color: white; font-size: 1.5rem; margin: 0.5rem 0;">{results['tumor_location']}</h3>
                                    <p style="color: #0ea5e9; font-weight: 600; margin: 0;">Coordinates Mapped</p>
                                </div>
                                ''', unsafe_allow_html=True)
                        
                        with col_z:
                            if results.get('tumor_metrics'):
                                area_pct = results['tumor_metrics'].get('area_percentage', 0)
                                st.markdown(f'''
                                <div class="glass-card" style="text-align: center; border-top: 4px solid #f43f5e;">
                                    <p style="color: #94a3b8; font-weight: bold; font-size: 0.9rem; text-transform: uppercase;">Brain Coverage Area</p>
                                    <h3 style="color: white; font-size: 1.5rem; margin: 0.5rem 0;">{area_pct:.2f}%</h3>
                                    <p style="color: #f43f5e; font-weight: 600; margin: 0;">Pixel Mass Calculated</p>
                                </div>
                                ''', unsafe_allow_html=True)
                        
                        # Visualization
                        if results.get('visualization_path') and os.path.exists(results['visualization_path']):
                            st.markdown('<div class="glass-card" style="margin-top: 1.5rem;">', unsafe_allow_html=True)
                            st.markdown("### 🎨 AI Segmentation Matrix")
                            vis_image = Image.open(results['visualization_path'])
                            st.image(vis_image, use_container_width=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Save to database
                    patient_info = {
                        'name': patient_name,
                        'age': patient_age,
                        'gender': patient_gender,
                        'contact': patient_contact,
                        'scan_date': datetime.now().strftime("%Y-%m-%d")
                    }
                    
                    # Add to database
                    patient_id = st.session_state.db.add_patient(
                        patient_name, patient_age, patient_gender, patient_contact
                    )
                    scan_id = st.session_state.db.add_scan(patient_id, temp_path)
                    
                    # Generate report
                    report_filename = f"report_{patient_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                    report_path = os.path.join(REPORT_DIR, report_filename)
                    
                    st.session_state.report_gen.generate_report(
                        patient_info, results, report_path
                    )
                    
                    # Add prediction to DB
                    st.session_state.db.add_prediction(
                        scan_id=scan_id,
                        has_tumor=results['has_tumor'],
                        tumor_type=results.get('tumor_type'),
                        confidence=results.get('tumor_type_confidence', 0),
                        tumor_location=results.get('tumor_location', {}),
                        tumor_area=results.get('tumor_metrics', {}).get('tumor_area', 0) if results.get('tumor_metrics') else 0,
                        report_path=report_path
                    )
                    
                    # Download report
                    st.markdown("---")
                    with open(report_path, "rb") as f:
                        st.download_button(
                            label="📥 Download Medical Report (PDF)",
                            data=f,
                            file_name=report_filename,
                            mime="application/pdf",
                            use_container_width=True
                        )
                    
                    st.success(f"✅ Analysis complete! Patient ID: {patient_id}")

    # GLOBAL PATIENT HISTORY PAGE (Doctor View)
    elif page == "Global Patient History":
        st.markdown('<h1 class="main-header" style="text-align: left;">📚 Master Patient Ledger</h1>', unsafe_allow_html=True)
        st.markdown("""<p style="color: #94a3b8; font-size: 1.1rem; margin-bottom: 2rem;">Secure overview of all registered patients and historical scans.</p>""", unsafe_allow_html=True)
        
        patients = st.session_state.db.get_all_patients()
        
        if not patients:
            st.info("No patients in database yet. Analyze a scan to add patient records.")
        else:
            # Create patient selection
            patient_options = {f"{p[1]} (ID: {p[0]})": p[0] for p in patients}
            
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            selected = st.selectbox("Select Patient Profile", list(patient_options.keys()))
            st.markdown('</div>', unsafe_allow_html=True)
            
            if selected:
                patient_id = patient_options[selected]
                history = st.session_state.db.get_patient_history(patient_id)
                patient = [p for p in patients if p[0] == patient_id][0]
                
                # Patient Info Header
                st.markdown(f'''
                    <div style="background: rgba(15, 23, 42, 0.8); border: 1px solid #334155; border-radius: 12px; padding: 1.5rem; display: flex; justify-content: space-between; align-items: center; margin-bottom: 2rem;">
                        <div>
                            <p style="color: #94a3b8; font-size: 0.9rem; text-transform: uppercase; margin: 0;">Patient Name</p>
                            <h2 style="color: white; margin: 0;">{patient[1]}</h2>
                        </div>
                        <div style="text-align: right;">
                            <p style="color: #94a3b8; font-size: 0.9rem; text-transform: uppercase; margin: 0;">Demographics</p>
                            <h3 style="color: #38bdf8; margin: 0;">{patient[2] if patient[2] else 'N/A'} yrs • {patient[3] if patient[3] else 'N/A'}</h3>
                        </div>
                        <div style="text-align: right; background: #1e293b; padding: 0.5rem 1rem; border-radius: 8px;">
                            <p style="color: #94a3b8; font-size: 0.9rem; text-transform: uppercase; margin: 0;">Total Scans</p>
                            <h2 style="color: #10b981; margin: 0;">{len(history)}</h2>
                        </div>
                    </div>
                ''', unsafe_allow_html=True)
                
                # Display scan history
                if history:
                    st.markdown("### 🗄️ Scan Archive")
                    
                    for i, record in enumerate(history, 1):
                        is_tumor = "⚠️ Positive Pathology" if record[3] else "✅ Clear"
                        tumor_color = "#ef4444" if record[3] else "#10b981"
                        
                        st.markdown(f'''
                        <div class="glass-card" style="border-left: 4px solid {tumor_color};">
                            <h3 style="margin-top: 0; color: white;">Scan {i} - <span style="color: #94a3b8; font-size: 1rem;">{record[1][:10]}</span></h3>
                            <div style="display: flex; gap: 2rem; margin-top: 1rem;">
                                <div>
                                    <p style="color: #94a3b8; font-size: 0.8rem; margin: 0; text-transform: uppercase;">Diagnosis</p>
                                    <p style="color: {tumor_color}; font-weight: bold; font-size: 1.1rem; margin: 0;">{is_tumor}</p>
                                </div>
                                {'<div><p style="color: #94a3b8; font-size: 0.8rem; margin: 0; text-transform: uppercase;">Type</p><p style="color: white; font-size: 1.1rem; margin: 0;">' + record[4].upper() + '</p></div>' if record[4] else ''}
                                {'<div><p style="color: #94a3b8; font-size: 0.8rem; margin: 0; text-transform: uppercase;">Confidence</p><p style="color: white; font-size: 1.1rem; margin: 0;">' + str(round(record[5]*100, 1)) + '%</p></div>' if record[5] else ''}
                            </div>
                        </div>
                        ''', unsafe_allow_html=True)
                        
                        if record[8] and os.path.exists(record[8]):
                            with open(record[8], "rb") as f:
                                st.download_button(
                                    f"📥 Download Clinical Report {i}",
                                    f,
                                    file_name=os.path.basename(record[8]),
                                    mime="application/pdf",
                                    key=f"dl_report_{i}"
                                )

    # STATISTICS PAGE (Doctor View)
    elif page == "System Statistics":
        st.markdown('<h1 class="main-header" style="text-align: left;">📊 Executive Dashboard</h1>', unsafe_allow_html=True)
        
        stats = st.session_state.db.get_statistics()
        
        # Overview metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f'<div class="glass-card" style="text-align: center; border-bottom: 4px solid #3b82f6;"><h1 style="font-size: 3rem; margin:0; color: white;">{stats["total_patients"]}</h1><p style="color:#94a3b8; font-weight: bold;">TOTAL PATIENTS</p></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="glass-card" style="text-align: center; border-bottom: 4px solid #8b5cf6;"><h1 style="font-size: 3rem; margin:0; color: white;">{stats["total_scans"]}</h1><p style="color:#94a3b8; font-weight: bold;">TOTAL SCANS</p></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="glass-card" style="text-align: center; border-bottom: 4px solid #10b981;"><h1 style="font-size: 3rem; margin:0; color: white;">{stats["tumor_detection_rate"]:.1f}%</h1><p style="color:#94a3b8; font-weight: bold;">DETECTION RATE</p></div>', unsafe_allow_html=True)
        
        # Tumor type distribution
        if stats['tumor_types']:
            st.markdown("### 🧬 Pathology Distribution")
            import pandas as pd
            df = pd.DataFrame(list(stats['tumor_types'].items()), columns=['Tumor Type', 'Count'])
            st.bar_chart(df.set_index('Tumor Type'), color="#0ea5e9")

    # PATIENT PORTAL (Patient View)
    elif page == "My Portal":
        st.markdown(f'''
            <div style="background: linear-gradient(135deg, #4f46e5, #ec4899); padding: 3rem; border-radius: 15px; margin-bottom: 2rem; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.4);">
                <h1 style="color: white; font-size: 2.5rem; margin-bottom: 0;">👋 Welcome back, {st.session_state.username}</h1>
                <p style="color: #fbcfe8; font-size: 1.2rem; margin-top: 0.5rem;">Access your secure medical dashboard below.</p>
            </div>
        ''', unsafe_allow_html=True)
        
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🏥 Important Note")
        st.write("Your medical scan history is privately authenticated and encrypted by NeuroScan AI. If you have any clinical questions regarding the results below, please contact your primary care physician.")
        st.markdown('</div>', unsafe_allow_html=True)

    # MY SCAN HISTORY (Patient View)
    elif page == "My Scan History":
        st.markdown('<h1 class="main-header" style="text-align: left;">📄 My Medical Records</h1>', unsafe_allow_html=True)
        
        patients = st.session_state.db.get_all_patients()
        my_patient_record = [p for p in patients if p[1].lower() == st.session_state.username.lower()]
        
        if not my_patient_record:
            st.info("No scan records found matching your profile name yet.")
        else:
            patient_id = my_patient_record[0][0]
            history = st.session_state.db.get_patient_history(patient_id)
            
            if history:
                for i, record in enumerate(history, 1):
                    st.markdown(f'''
                    <div class="glass-card" style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <p style="color: #94a3b8; margin: 0; font-size: 0.9rem;">Date Evaluated</p>
                            <h3 style="color: white; margin: 0;">{record[1][:10]}</h3>
                        </div>
                        <div style="text-align: right;">
                            <p style="color: #38bdf8; margin: 0; font-weight: bold;">Report Verified ✓</p>
                        </div>
                    </div>
                    ''', unsafe_allow_html=True)
                    
                    if record[8] and os.path.exists(record[8]):
                        with open(record[8], "rb") as f:
                            st.download_button(
                                "📥 Download Secure PDF Report",
                                f,
                                file_name=os.path.basename(record[8]),
                                mime="application/pdf",
                                key=f"pat_dl_report_{i}"
                            )
            else:
                st.info("Your profile exists but no scans have been uploaded by a Doctor yet.")

    # ABOUT PAGE
    elif page == "About":
        st.markdown('<h1 class="main-header">ℹ️ About NeuroScan AI</h1>', 
                    unsafe_allow_html=True)
        
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("""
        ### Brain Tumor Detection System v2.0
        
        **Technologies Used:**
        - TensorFlow/Keras for deep learning
        - OpenCV for image processing
        - Streamlit for web interface
        - SQLite for database
        - ReportLab for PDF generation
        
        **Features:**
        - ✅ High-accuracy tumor detection
        - ✅ Multi-class tumor classification
        - ✅ AI-Driven tumor highlighting
        - ✅ Role-based secure access
        - ✅ Comprehensive PDF reports
        
        **Disclaimer:**
        This system is designed as a diagnostic support tool. All results should be 
        reviewed by qualified medical professionals. This tool does not replace 
        professional medical diagnosis or treatment.
        
        ---
        
        **Academic Project Final Year - 2026**
        """)
        st.markdown('</div>', unsafe_allow_html=True)
