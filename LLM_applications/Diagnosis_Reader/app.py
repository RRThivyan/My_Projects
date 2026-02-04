import streamlit as st
from pathlib import Path
from datetime import datetime
import os
from dotenv import load_dotenv
import fitz
import re
import requests
from typing import Tuple
import time

# ----------------- Load Config -----------------
load_dotenv()

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME")
API_VERSION = os.getenv("API_VERSION")
API_KEY = os.getenv("API_KEY")

TEMPERATURE = float(os.getenv("TEMPERATURE", 0.1))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 3000))  # Changed default to 3000

MAX_EXTRACTED_CHARS = 200_000
MAX_LINE_LEN = 600

# ----------------- Custom CSS -----------------
def inject_custom_css():
    st.markdown("""
    <style>
    /* Main container styling */
    .main {
        padding: 2rem;
    }
    
    /* Header styling with gradient */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem 1.5rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .header-title {
        font-size: 3rem;
        font-weight: 800;
        color: white;
        text-align: center;
        margin-bottom: 0.5rem;
        background: linear-gradient(45deg, #FFD700, #FFECB3);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .header-subtitle {
        font-size: 1.2rem;
        color: rgba(255,255,255,0.9);
        text-align: center;
        max-width: 800px;
        margin: 0 auto;
        line-height: 1.6;
    }
    
    /* Card styling */
    .card {
        background: white;
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 5px 20px rgba(0,0,0,0.08);
        margin-bottom: 1.5rem;
        border: 1px solid rgba(0,0,0,0.05);
        transition: transform 0.3s ease;
    }
    
    .card:hover {
        transform: translateY(-5px);
    }
    
    .card-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #2d3748;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    /* Upload area styling */
    .upload-area {
        border: 3px dashed #667eea;
        border-radius: 20px;
        padding: 3rem 2rem;
        text-align: center;
        background: rgba(102, 126, 234, 0.05);
        transition: all 0.3s ease;
        margin: 2rem 0;
    }
    
    .upload-area:hover {
        background: rgba(102, 126, 234, 0.1);
        border-color: #764ba2;
    }
    
    /* Status indicators */
    .status-box {
        padding: 1rem;
        border-radius: 15px;
        margin: 1rem 0;
        display: flex;
        align-items: center;
        gap: 10px;
        font-weight: 600;
    }
    
    .status-success {
        background: rgba(72, 187, 120, 0.1);
        color: #48bb78;
        border: 2px solid #48bb78;
    }
    
    .status-info {
        background: rgba(66, 153, 225, 0.1);
        color: #4299e1;
        border: 2px solid #4299e1;
    }
    
    .status-processing {
        background: rgba(237, 137, 54, 0.1);
        color: #ed8936;
        border: 2px solid #ed8936;
    }
    
    /* Summary box */
    .summary-container {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 20px;
        padding: 2rem;
        border: 2px solid #dee2e6;
        margin: 2rem 0;
    }
    
    .summary-title {
        font-size: 2rem;
        font-weight: 800;
        color: #2d3748;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    
    /* Download button */
    .stDownloadButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 1rem 2rem;
        border-radius: 15px;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        width: 100%;
        margin-top: 1rem;
    }
    
    .stDownloadButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Icon colors */
    .icon-blue {
        color: #667eea;
    }
    
    .icon-purple {
        color: #764ba2;
    }
    
    /* Feature grid */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .feature-item {
        text-align: center;
        padding: 1.5rem;
        background: white;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.05);
    }
    
    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    
    /* Security badge */
    .security-badge {
        background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 5px;
        margin-top: 1rem;
    }
    
    /* Upload info box */
    .upload-info {
        background: rgba(102, 126, 234, 0.05);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 2px solid rgba(102, 126, 234, 0.2);
    }
    
    /* Summary text styling */
    .summary-text {
        font-family: 'Courier New', monospace;
        line-height: 1.6;
        color: #2d3748;
        white-space: pre-wrap;
        font-size: 0.95rem;
    }
    </style>
    """, unsafe_allow_html=True)

# ----------------- PDF Functions -----------------
def analyze_pdf_text(pdf_path: Path) -> Tuple[str, bool]:
    doc = fitz.open(pdf_path)
    text_pages = 0
    text_chunks = []
    for page in doc:
        txt = page.get_text("text").strip()
        if len(txt) > 50:
            text_pages += 1
            text_chunks.append(txt)
    return "\n\n".join(text_chunks), text_pages >= 1

def extract_with_docling(pdf_path: Path) -> str:
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.datamodel.base_models import InputFormat
    from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend

    pdf_opts = PdfPipelineOptions(do_ocr=True, do_table_structure=False)
    pdf_opts.ocr_options.force_full_page_ocr = True

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(
            pipeline_options=pdf_opts,
            backend=PyPdfiumDocumentBackend,
        )}
    )
    result = converter.convert(pdf_path)
    return result.document.export_to_markdown() or ""

def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\bpage\s*\d+(\s*of\s*\d+)?\b", " ", text, flags=re.I)
    text = re.sub(r"[\-_]{8,}", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [line.strip()[:MAX_LINE_LEN] for line in text.splitlines() if line.strip()]
    return "\n".join(lines)[:MAX_EXTRACTED_CHARS]

def build_doctor_prompt(extracted_text: str) -> str:
    return f"""
You are extracting information from a medical laboratory report to create a doctor-facing summary.

CRITICAL RULES:
1. Extract ONLY factual information that is EXPLICITLY written in the report
2. Do NOT add any interpretation, diagnosis, or medical advice
3. Do NOT infer anything not explicitly stated
4. If a test result value is NOT provided in the text, do NOT create it
5. Preserve exact values, units, dates, and technical terms as written
6. If you see repetitive numbers or patterns (like "1 1 1 1 1"), ignore them as they are likely OCR artifacts
7. Focus only on the meaningful medical information

SECTIONS TO CREATE (in this exact order):

1. PATIENT & REPORT DETAILS
   - Extract: Patient name, age, gender as written
   - Extract: Referring doctor if mentioned
   - Extract: Address as written
   - Extract: Laboratory name
   - Extract: Collection date and time (SCT)
   - Extract: Report date and time (RRT)
   - Extract: Sample type
   - Extract: Lab codes and barcodes
   - Extract: Tests performed from "TEST ASKED" or similar sections

2. EXTRACTED TEST RESULTS
   - For EACH test that has actual values provided:
     * Test name exactly as written
     * Value (ONLY if provided in text)
     * Units (ONLY if provided in text)
     * Reference range (ONLY if provided in text)
     * Method/technology (ONLY if provided in text)
     * Any notes or disclaimers that are explicitly written
   - If a test is listed but NO values are provided, only mention it was performed

3. REPORT NOTES & DISCLAIMERS
   - Only include notes that are explicitly written in the report
   - Include any critical value indicators if mentioned
   - Include test availability status if mentioned

4. LABORATORY INFORMATION
   - Only include if explicitly mentioned in the report

IMPORTANT: If the text contains patterns like "1 1 1 1 1" or repeated numbers, these are OCR artifacts. IGNORE THEM COMPLETELY.

REPORT TEXT TO ANALYZE:
{extracted_text}
""".strip()

def call_azure_openai(prompt: str, temperature: float = 0.1, max_tokens: int = 3000) -> str:  # Default changed to 3000
    url = f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{DEPLOYMENT_NAME}/chat/completions?api-version={API_VERSION}"
    headers = {"Content-Type": "application/json", "api-key": API_KEY}
    payload = {
        "messages": [
            {"role": "system", "content": "You are a factual medical report summarizer that extracts information exactly as presented. You do not interpret, infer, or add any information not explicitly written. You ignore OCR artifacts like repeated numbers."},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    response = requests.post(url, headers=headers, json=payload, timeout=120)
    if response.status_code != 200:
        raise RuntimeError(f"Azure OpenAI error {response.status_code}: {response.text}")
    return response.json()["choices"][0]["message"]["content"].strip()

# ----------------- Streamlit App -----------------
def main():
    # Page config
    st.set_page_config(
        page_title="SmartReport AI | Medical PDF Summarizer",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Inject custom CSS
    inject_custom_css()
    
    # Header with gradient
    st.markdown("""
    <div class="header-container">
        <h1 class="header-title">🩺 SmartReport AI</h1>
        <p class="header-subtitle">
        Transform complex medical reports into clear, doctor-friendly summaries in seconds.<br>
        Powered by advanced AI for accurate and reliable results.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Features grid
    st.markdown("""
    <div class="feature-grid">
        <div class="feature-item">
            <div class="feature-icon">⚡</div>
            <h3>Instant Processing</h3>
            <p>Get summaries in seconds, not hours</p>
        </div>
        <div class="feature-item">
            <div class="feature-icon">🛡️</div>
            <h3>Secure & Private</h3>
            <p>Your documents are processed securely</p>
        </div>
        <div class="feature-item">
            <div class="feature-icon">🤖</div>
            <h3>AI-Powered</h3>
            <p>Advanced medical AI for accurate extraction</p>
        </div>
        <div class="feature-item">
            <div class="feature-icon">📊</div>
            <h3>Doctor-Friendly</h3>
            <p>Structured summaries for easy review</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Main content in two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📄 Upload Medical Report</div>', unsafe_allow_html=True)
        
        # Upload area
        st.markdown('<div class="upload-area">', unsafe_allow_html=True)
        
        # Fixed file uploader with proper label
        uploaded_file = st.file_uploader(
            "Upload PDF Report",
            type=["pdf"],
            help="Upload your medical report in PDF format",
            label_visibility="collapsed"
        )
        
        if uploaded_file:
            st.markdown(f"""
            <div class="upload-info">
                <div style="font-size: 1.2rem; color: #667eea; font-weight: 600; margin-bottom: 0.5rem;">
                    ✅ File Ready for Processing
                </div>
                <div style="color: #4a5568;">
                    <strong>Filename:</strong> {uploaded_file.name}<br>
                    <strong>Size:</strong> {uploaded_file.size / 1024:.1f} KB
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="color: #667eea; font-size: 1.5rem; margin-bottom: 1rem;">
                📤 Drag & Drop or Click to Upload
            </div>
            <div style="color: #718096; margin-bottom: 1rem;">
                Supports PDF files up to 50MB
            </div>
            <div style="color: #a0aec0; font-size: 0.9rem;">
                🔒 All processing is done securely and files are deleted immediately
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)  # Close upload area
        
        # Advanced options - Only show summary length, not temperature
        with st.expander("⚙️ Summary Settings", expanded=False):
            st.markdown("""
            <div style="background: #f7fafc; padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
                <p style="color: #4a5568; font-size: 0.9rem; margin: 0;">
                    <strong>Note:</strong> AI creativity is fixed at low (0.1) for medical accuracy.
                    Adjust summary length only for detail level.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            tokens_input = st.slider(
                "Summary Length", 
                min_value=100, 
                max_value=3000, 
                value=MAX_TOKENS,  # Now defaults to 3000
                step=50,
                help="Shorter = more concise, Longer = more detailed"
            )
        
        st.markdown('</div>', unsafe_allow_html=True)  # Close card
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">✨ Generated Summary</div>', unsafe_allow_html=True)
        
        summary_area = st.empty()
        summary_area.markdown("""
        <div style="text-align: center; padding: 3rem 1rem; color: #a0aec0;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">📋</div>
            <h3 style="color: #718096;">Your summary will appear here</h3>
            <p>Upload a PDF to see the magic happen!</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Security badge at the bottom
    st.markdown("""
    <div style="text-align: center; margin-top: 2rem;">
        <div class="security-badge">
            🔒 HIPAA Compliant Processing • Files Deleted After Processing
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Process PDF if uploaded
    if uploaded_file:
        # Create temp file
        temp_pdf_path = Path("temp_uploaded.pdf")
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_file.read())
        
        # Status updates
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Extraction
        status_text.markdown('<div class="status-processing">🔄 Extracting text from PDF...</div>', unsafe_allow_html=True)
        progress_bar.progress(25)
        
        text, is_digital = analyze_pdf_text(temp_pdf_path)
        if not is_digital:
            status_text.markdown('<div class="status-info">🔍 Digital text not found, running OCR...</div>', unsafe_allow_html=True)
            text = extract_with_docling(temp_pdf_path)
        
        progress_bar.progress(50)
        
        # Step 2: Processing
        status_text.markdown('<div class="status-processing">🧹 Cleaning and preparing text...</div>', unsafe_allow_html=True)
        extracted = clean_text(text)
        prompt = build_doctor_prompt(extracted)
        
        progress_bar.progress(75)
        
        # Step 3: AI Generation (using fixed temperature for accuracy)
        status_text.markdown('<div class="status-processing">🤖 Generating doctor summary with AI...</div>', unsafe_allow_html=True)
        
        # Use fixed temperature (0.1) for medical accuracy, only user-adjusted tokens
        summary = call_azure_openai(
            prompt=prompt,
            temperature=TEMPERATURE,
            max_tokens=tokens_input
        )
        
        # Step 4: Complete
        progress_bar.progress(100)
        status_text.markdown('<div class="status-success">✅ Summary generated successfully!</div>', unsafe_allow_html=True)
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()
        
        # Display summary in a beautiful container with monospace font for better readability
        summary_area.markdown(f"""
        <div class="summary-container">
            <div class="summary-title">📋 Medical Summary</div>
            <div style="background: white; padding: 2rem; border-radius: 15px; border: 1px solid #e2e8f0;">
                <div class="summary-text">
                    {summary.replace(chr(10), '<br>')}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Download button
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        download_filename = f"medical_summary_{timestamp}.txt"
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.download_button(
                label="⬇️ Download Summary",
                data=summary,
                file_name=download_filename,
                mime="text/plain",
                help="Save this summary to your device"
            )
        
        # Clean up
        temp_pdf_path.unlink()
        
        # Success message with option to upload another
        st.balloons()
        
        # Add a nice success message with option to process another file
        st.markdown("""
        <div style="background: linear-gradient(135deg, #48bb78 0%, #38a169 100%); 
                    color: white; padding: 1.5rem; border-radius: 15px; margin-top: 2rem;">
            <h3 style="margin: 0 0 1rem 0;">✨ Processing Complete!</h3>
            <p style="margin: 0;">Your summary is ready. You can download it above or upload another report.</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()