"""HR Policy Assistant — Beautiful, Modern & Highly Interactive Streamlit Web Application."""

import sys
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components

# Ensure project root is in Python path
sys.path.append(str(Path(__file__).resolve().parent))

from src.config import (
    PDF_PATH,
    CHROMA_COLLECTION_NAME,
    ASSIGNMENT_QUESTIONS,
    OPENAI_API_KEY
)
from src.rag_pipeline import HRPolicyRAGPipeline
from src.document_loader import load_pdf_pages
from src.chunker import chunk_documents
from src.vectorstore import index_documents

# Page configuration
st.set_page_config(
    page_title="HR Policy Assistant — Enso Group",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom Styling (Lavender-to-Light-Blue Theme, Glassmorphism, Micro-Interactions)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    *, *::before, *::after {
        box-sizing: border-box !important;
    }

    html, body {
        overflow-x: hidden !important;
        width: 100% !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    /* Universal Full-Page Lavender to Light Blue Gradient Background */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stApp"], section.main, .main, [data-testid="stMain"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background: linear-gradient(135deg, #EEF2FF 0%, #F5F3FF 40%, #E0F2FE 80%, #EDE9FE 100%) !important;
        background-attachment: fixed !important;
        color: #0F172A !important;
        min-height: 100vh !important;
        overflow-x: hidden !important;
    }

    [data-testid="stHeader"] {
        background: transparent !important;
    }

    /* Main Container with responsive fluid width */
    .block-container {
        width: 100% !important;
        max-width: 960px !important;
        padding-top: 1.2rem !important;
        padding-bottom: 3.5rem !important;
        padding-left: clamp(1rem, 3.5vw, 2rem) !important;
        padding-right: clamp(1rem, 3.5vw, 2rem) !important;
        margin: 0 auto !important;
        position: relative;
        z-index: 2;
    }

    /* Premium Navigation Header */
    .nav-header {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        justify-content: space-between;
        gap: 0.85rem;
        background: rgba(255, 255, 255, 0.92);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        padding: 0.9rem 1.4rem;
        border-radius: 18px;
        border: 1px solid rgba(255, 255, 255, 0.8);
        box-shadow: 0 10px 30px -5px rgba(79, 70, 229, 0.08), 0 4px 10px -2px rgba(0, 0, 0, 0.03);
        margin-bottom: 1.8rem;
        width: 100%;
        transition: transform 0.2s ease;
    }

    .brand-logo-area {
        display: flex;
        align-items: center;
        gap: 0.8rem;
    }

    .brand-icon {
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
        color: #FFFFFF;
        width: 44px;
        height: 44px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 12px;
        font-size: 1.4rem;
        box-shadow: 0 4px 14px rgba(79, 70, 229, 0.35);
        flex-shrink: 0;
    }

    .brand-title {
        font-size: clamp(1.2rem, 3vw, 1.4rem);
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.02em;
    }

    .ai-badge {
        background: linear-gradient(135deg, #F0FDFA 0%, #E6FFFA 100%);
        color: #0D9488;
        font-size: 0.75rem;
        font-weight: 700;
        padding: 0.25rem 0.7rem;
        border-radius: 9999px;
        border: 1px solid #99F6E4;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-left: 0.4rem;
        box-shadow: 0 2px 6px rgba(13, 148, 136, 0.12);
    }

    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.85rem;
        font-weight: 600;
        padding: 0.4rem 0.95rem;
        border-radius: 9999px;
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        color: #334155;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    }

    .status-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background-color: #10B981;
        box-shadow: 0 0 8px #10B981;
        animation: pulseDot 2s infinite;
        flex-shrink: 0;
    }

    @keyframes pulseDot {
        0% { transform: scale(0.95); opacity: 0.8; }
        50% { transform: scale(1.15); opacity: 1; }
        100% { transform: scale(0.95); opacity: 0.8; }
    }

    /* Hero Section */
    .hero-container {
        text-align: center;
        padding: 1.2rem 0.5rem 1.8rem 0.5rem;
        width: 100%;
    }

    .hero-title {
        font-size: clamp(1.9rem, 5.5vw, 2.85rem);
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.03em;
        line-height: 1.2;
        margin-bottom: 0.6rem;
        word-break: break-word;
    }

    .hero-highlight {
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 60%, #9333EA 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-subtitle {
        font-size: clamp(1rem, 2.5vw, 1.18rem);
        color: #475569;
        font-weight: 500;
        margin-bottom: 0.45rem;
        line-height: 1.5;
    }

    .hero-tagline {
        font-size: clamp(0.88rem, 2vw, 0.98rem);
        color: #0D9488;
        font-weight: 700;
        letter-spacing: 0.02em;
    }

    /* Chat Assistant Header */
    .chat-header {
        display: flex;
        align-items: center;
        gap: 0.9rem;
        padding-bottom: 1.2rem;
        border-bottom: 1px solid #F1F5F9;
        margin-bottom: 1.5rem;
        width: 100%;
    }

    .assistant-avatar {
        width: 46px;
        height: 46px;
        border-radius: 14px;
        background: linear-gradient(135deg, #EEF2FF 0%, #E0E7FF 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.45rem;
        border: 1.5px solid #C7D2FE;
        box-shadow: 0 4px 10px rgba(79, 70, 229, 0.15);
        flex-shrink: 0;
    }

    .chat-welcome-title {
        font-size: clamp(1.05rem, 2.5vw, 1.18rem);
        font-weight: 800;
        color: #1E1B4B;
        margin: 0;
    }

    .chat-welcome-sub {
        font-size: clamp(0.84rem, 2vw, 0.92rem);
        color: #64748B;
        margin: 0;
    }

    /* Message Bubbles */
    .msg-user-container {
        display: flex;
        justify-content: flex-end;
        width: 100%;
        margin: 0.9rem 0;
    }

    .msg-user-bubble {
        background: linear-gradient(135deg, #4F46E5 0%, #6366F1 100%);
        color: #FFFFFF !important;
        padding: 0.9rem 1.35rem;
        border-radius: 18px 18px 4px 18px;
        max-width: min(85%, 680px);
        font-size: clamp(0.94rem, 2vw, 1.02rem);
        font-weight: 500;
        line-height: 1.55;
        box-shadow: 0 6px 18px rgba(79, 70, 229, 0.25);
        word-break: break-word;
        overflow-wrap: anywhere;
    }

    .msg-bot-container {
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        width: 100%;
        margin: 0.9rem 0;
    }

    .msg-bot-bubble {
        background: #FFFFFF;
        color: #0F172A !important;
        border-left: 5px solid #4F46E5;
        border-top: 1px solid #E2E8F0;
        border-right: 1px solid #E2E8F0;
        border-bottom: 1px solid #E2E8F0;
        padding: 1.25rem 1.6rem;
        border-radius: 4px 18px 18px 18px;
        width: 100%;
        max-width: 100%;
        font-size: clamp(0.96rem, 2vw, 1.06rem);
        line-height: 1.7;
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.04);
        word-break: break-word;
        overflow-wrap: anywhere;
    }

    .msg-not-found-bubble {
        background: #FEF2F2;
        color: #991B1B !important;
        border-left: 5px solid #EF4444;
        border-top: 1px solid #FEE2E2;
        border-right: 1px solid #FEE2E2;
        border-bottom: 1px solid #FEE2E2;
        padding: 1.15rem 1.5rem;
        border-radius: 4px 18px 18px 18px;
        width: 100%;
        font-size: clamp(0.96rem, 2vw, 1.04rem);
        line-height: 1.6;
        word-break: break-word;
        overflow-wrap: anywhere;
    }

    .citation-wrapper {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 0.5rem;
        margin-top: 0.9rem;
        padding-top: 0.85rem;
        border-top: 1px solid #F1F5F9;
        width: 100%;
    }

    .citation-label {
        font-size: 0.82rem;
        font-weight: 700;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .citation-badge {
        background: #EEF2FF;
        color: #4338CA !important;
        border: 1px solid #C7D2FE;
        font-size: 0.84rem;
        font-weight: 700;
        padding: 0.35rem 0.8rem;
        border-radius: 8px;
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        box-shadow: 0 2px 4px rgba(79, 70, 229, 0.1);
        transition: transform 0.2s ease;
    }
    .citation-badge:hover {
        transform: translateY(-1px);
        background: #E0E7FF;
    }

    /* Section Headings */
    .section-title {
        font-size: clamp(1.25rem, 3vw, 1.45rem);
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.02em;
        margin-bottom: 0.25rem;
    }

    .section-subtitle {
        font-size: clamp(0.86rem, 2vw, 0.96rem);
        color: #64748B;
        margin-bottom: 1.1rem;
    }

    /* Input Box */
    .stTextInput > div > div > input {
        border-radius: 14px !important;
        border: 2px solid #CBD5E1 !important;
        padding: 0.9rem 1.25rem !important;
        font-size: clamp(0.96rem, 2vw, 1.05rem) !important;
        background-color: #FFFFFF !important;
        color: #0F172A !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03) !important;
        transition: all 0.25s ease !important;
        width: 100% !important;
    }

    .stTextInput > div > div > input:focus {
        border-color: #4F46E5 !important;
        box-shadow: 0 0 0 4px rgba(79, 70, 229, 0.2) !important;
        transform: translateY(-1px);
    }

    /* Buttons Styling */
    .stButton > button {
        width: 100% !important;
        min-height: 52px !important;
        border-radius: 14px !important;
        font-weight: 600 !important;
        font-size: clamp(0.88rem, 2vw, 0.96rem) !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        white-space: normal !important;
        word-break: break-word !important;
        padding: 0.8rem 1.15rem !important;
        line-height: 1.4 !important;
        cursor: pointer !important;
    }

    /* Primary Ask Button - Vibrant Gradient */
    button[data-testid="baseButton-primary"], button[kind="primary"] {
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        text-align: center !important;
        justify-content: center !important;
        font-size: 1.05rem !important;
        box-shadow: 0 6px 18px rgba(79, 70, 229, 0.38) !important;
    }

    button[data-testid="baseButton-primary"]:hover, button[kind="primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 25px rgba(79, 70, 229, 0.5) !important;
        color: #FFFFFF !important;
    }

    button[data-testid="baseButton-primary"]:active, button[kind="primary"]:active {
        transform: scale(0.97) !important;
    }

    /* Secondary Clear Button */
    button[data-testid="baseButton-secondary"], button[kind="secondary"] {
        background: #FFFFFF !important;
        color: #475569 !important;
        border: 1.5px solid #CBD5E1 !important;
        border-radius: 14px !important;
        text-align: center !important;
        justify-content: center !important;
        font-size: 0.95rem !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04) !important;
    }

    button[data-testid="baseButton-secondary"]:hover, button[kind="secondary"]:hover {
        background: #F8FAFC !important;
        border-color: #94A3B8 !important;
        color: #1E293B !important;
        transform: translateY(-1px) !important;
    }

    /* Suggested Questions as Interactive Cards with Hover Glow */
    div[data-testid="column"] button {
        background: rgba(255, 255, 255, 0.95) !important;
        border: 1.5px solid #E2E8F0 !important;
        color: #1E1B4B !important;
        border-radius: 16px !important;
        padding: 0.95rem 1.25rem !important;
        text-align: left !important;
        justify-content: flex-start !important;
        box-shadow: 0 4px 14px -2px rgba(79, 70, 229, 0.06) !important;
        margin-bottom: 0.65rem !important;
        width: 100% !important;
        position: relative;
        overflow: hidden;
    }

    div[data-testid="column"] button:hover {
        background: #FFFFFF !important;
        border-color: #6366F1 !important;
        transform: translateY(-3px) !important;
        box-shadow: 0 12px 28px -3px rgba(79, 70, 229, 0.2) !important;
        color: #4F46E5 !important;
    }

    div[data-testid="column"] button:active {
        transform: scale(0.98) !important;
    }

    /* Responsive Breakpoints */
    @media screen and (max-width: 768px) {
        div[data-testid="column"] {
            flex: 1 1 100% !important;
            width: 100% !important;
            min-width: 100% !important;
            margin-bottom: 0.5rem !important;
        }
        .block-container {
            padding-left: 0.85rem !important;
            padding-right: 0.85rem !important;
        }
        .nav-header {
            flex-direction: column !important;
            align-items: flex-start !important;
            padding: 0.85rem 1rem !important;
        }
        .msg-user-bubble {
            max-width: 90%;
        }
        .msg-bot-bubble, .msg-not-found-bubble {
            max-width: 100%;
        }
    }
</style>
""", unsafe_allow_html=True)

# Interactive Background Click & Touch Ripple Effect Component
components.html("""
<script>
(function() {
    try {
        const doc = window.parent.document;
        if (doc.getElementById('interactive-bg-canvas')) return;

        // Create full-screen canvas behind content
        const canvas = doc.createElement('canvas');
        canvas.id = 'interactive-bg-canvas';
        canvas.style.position = 'fixed';
        canvas.style.top = '0';
        canvas.style.left = '0';
        canvas.style.width = '100vw';
        canvas.style.height = '100vh';
        canvas.style.pointerEvents = 'none';
        canvas.style.zIndex = '1';
        canvas.style.opacity = '0.85';
        doc.body.appendChild(canvas);

        const ctx = canvas.getContext('2d');
        let width = canvas.width = window.parent.innerWidth;
        let height = canvas.height = window.parent.innerHeight;

        window.parent.addEventListener('resize', () => {
            width = canvas.width = window.parent.innerWidth;
            height = canvas.height = window.parent.innerHeight;
        });

        // Interactive Ripples and Particles
        const ripples = [];
        const particles = [];

        // Ambient floating background particles
        const particleCount = 28;
        for (let i = 0; i < particleCount; i++) {
            particles.push({
                x: Math.random() * width,
                y: Math.random() * height,
                vx: (Math.random() - 0.5) * 0.6,
                vy: (Math.random() - 0.5) * 0.6,
                radius: Math.random() * 2.5 + 2,
                color: ['rgba(99, 102, 241, 0.45)', 'rgba(139, 92, 246, 0.45)', 'rgba(20, 184, 166, 0.4)'][Math.floor(Math.random() * 3)]
            });
        }

        // On Click / Touch anywhere on parent document
        doc.addEventListener('pointerdown', (e) => {
            // Ripple Wave 1 (Inner Indigo Glow)
            ripples.push({
                x: e.clientX,
                y: e.clientY,
                radius: 4,
                maxRadius: 130,
                alpha: 0.65,
                color: 'rgba(99, 102, 241, '
            });
            // Ripple Wave 2 (Outer Teal Glow)
            ripples.push({
                x: e.clientX,
                y: e.clientY,
                radius: 2,
                maxRadius: 170,
                alpha: 0.45,
                color: 'rgba(20, 184, 166, '
            });

            // Burst sparkling particles on touch/click
            for (let i = 0; i < 9; i++) {
                const angle = Math.random() * Math.PI * 2;
                const speed = Math.random() * 3.2 + 1.2;
                particles.push({
                    x: e.clientX,
                    y: e.clientY,
                    vx: Math.cos(angle) * speed,
                    vy: Math.sin(angle) * speed,
                    radius: Math.random() * 3 + 1.5,
                    alpha: 1.0,
                    life: 40,
                    color: 'rgba(124, 58, 237, '
                });
            }
        });

        // Animation Loop
        function animate() {
            ctx.clearRect(0, 0, width, height);

            // Update & render floating particles
            for (let i = particles.length - 1; i >= 0; i--) {
                const p = particles[i];
                p.x += p.vx;
                p.y += p.vy;

                if (p.life !== undefined) {
                    p.life--;
                    p.alpha = p.life / 40;
                    if (p.life <= 0) {
                        particles.splice(i, 1);
                        continue;
                    }
                } else {
                    if (p.x < 0) p.x = width;
                    if (p.x > width) p.x = 0;
                    if (p.y < 0) p.y = height;
                    if (p.y > height) p.y = 0;
                }

                ctx.beginPath();
                ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
                ctx.fillStyle = p.alpha !== undefined ? p.color + (p.alpha * 0.8) + ')' : p.color;
                ctx.fill();
            }

            // Draw connective delicate constellation lines between nearby particles
            for (let i = 0; i < particles.length; i++) {
                for (let j = i + 1; j < particles.length; j++) {
                    const dx = particles[i].x - particles[j].x;
                    const dy = particles[i].y - particles[j].y;
                    const dist = Math.sqrt(dx * dx + dy * dy);
                    if (dist < 110) {
                        ctx.beginPath();
                        ctx.strokeStyle = `rgba(165, 180, 252, ${0.28 * (1 - dist / 110)})`;
                        ctx.lineWidth = 1;
                        ctx.moveTo(particles[i].x, particles[i].y);
                        ctx.lineTo(particles[j].x, particles[j].y);
                        ctx.stroke();
                    }
                }
            }

            // Update & render click ripple rings
            for (let i = ripples.length - 1; i >= 0; i--) {
                const r = ripples[i];
                r.radius += 4;
                r.alpha -= 0.018;

                if (r.alpha <= 0 || r.radius >= r.maxRadius) {
                    ripples.splice(i, 1);
                    continue;
                }

                ctx.beginPath();
                ctx.arc(r.x, r.y, r.radius, 0, Math.PI * 2);
                ctx.strokeStyle = r.color + r.alpha + ')';
                ctx.lineWidth = 2.5;
                ctx.stroke();
            }

            requestAnimationFrame(animate);
        }
        animate();
    } catch(err) {
        console.warn('Interactive background canvas:', err);
    }
})();
</script>
""", height=0, width=0)


@st.cache_resource
def get_pipeline(api_key: str = "") -> HRPolicyRAGPipeline:
    """Initialize and cache the RAG pipeline."""
    return HRPolicyRAGPipeline(openai_api_key=api_key or OPENAI_API_KEY)


def main():
    pipeline = get_pipeline()
    chunk_count = pipeline.get_collection_count()
    pdf_exists = PDF_PATH.exists()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "input_query" not in st.session_state:
        st.session_state.input_query = ""

    # ==========================================
    # 1. Responsive Website Header
    # ==========================================
    status_text = f"Manual Ready ({chunk_count} Chunks)" if chunk_count > 0 else "Indexing Required"
    status_color = "#10B981" if chunk_count > 0 else "#F59E0B"

    st.markdown(f"""
    <header class="nav-header">
        <div class="brand-logo-area">
            <div class="brand-icon">📋</div>
            <div>
                <span class="brand-title">HR Policy Assistant</span>
                <span class="ai-badge">AI-Powered</span>
            </div>
        </div>
        <div class="status-pill">
            <span class="status-dot" style="background-color: {status_color};"></span>
            <span>{status_text}</span>
        </div>
    </header>
    """, unsafe_allow_html=True)

    # ==========================================
    # 2. Hero Section
    # ==========================================
    st.markdown("""
    <section class="hero-container">
        <h1 class="hero-title">Your HR Policies, <span class="hero-highlight">Made Simple.</span></h1>
        <p class="hero-subtitle">Get clear answers to your HR questions, backed by the official HR policy manual.</p>
        <p class="hero-tagline">Ask questions. Find policies. Understand your benefits.</p>
    </section>
    """, unsafe_allow_html=True)

    # Check if DB needs indexing
    if chunk_count == 0 and pdf_exists:
        st.warning("Knowledge base is not indexed yet. Click below to prepare ChromaDB vector store.")
        if st.button("Index HR Manual (79 Pages)", type="primary"):
            with st.spinner("Extracting and indexing all 79 pages..."):
                docs, _ = load_pdf_pages(PDF_PATH)
                chunks, _ = chunk_documents(docs)
                index_documents(chunks, force_reindex=True)
                st.cache_resource.clear()
                st.success("Indexing completed successfully!")
                st.rerun()

    # ==========================================
    # 3. Main Chatbot Card
    # ==========================================
    st.markdown("""
    <div class="chat-header">
        <div class="assistant-avatar">🤖</div>
        <div>
            <p class="chat-welcome-title">Enso HR Assistant</p>
            <p class="chat-welcome-sub">I only answer from verified policy text and cite exact pages.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Display Conversation History
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="msg-user-container">
                <div class="msg-user-bubble">{msg['content']}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            ans = msg["content"]
            pages = msg.get("pages", [])
            if ans == "Not found":
                st.markdown('<div class="msg-bot-container"><div class="msg-not-found-bubble"><strong>Status:</strong> Not found<br><span style="font-size: 0.92rem; color: #7F1D1D;">The Enso Group HR Policy Manual does not contain sufficient evidence to answer this question.</span></div></div>', unsafe_allow_html=True)
            else:
                badges_html = "".join([f'<span class="citation-badge">📄 Page {p}</span>' for p in pages])
                citations_section = f'<div class="citation-wrapper"><span class="citation-label">Verified Sources:</span>{badges_html}</div>' if pages else ""
                bot_html = f'<div class="msg-bot-container"><div class="msg-bot-bubble">{ans}{citations_section}</div></div>'
                st.markdown(bot_html, unsafe_allow_html=True)

    # Question Input Area
    query_val = st.text_input(
        "Your Question:",
        value=st.session_state.input_query,
        placeholder="Type your question here (e.g. How many days of Privilege Leave does an employee get?)...",
        label_visibility="collapsed"
    )

    col_ask, col_clear = st.columns([1.5, 1])
    with col_ask:
        ask_pressed = st.button("Ask Assistant", type="primary", use_container_width=True)
    with col_clear:
        clear_pressed = st.button("Clear Chat", type="secondary", use_container_width=True)

    if clear_pressed:
        st.session_state.messages = []
        st.session_state.input_query = ""
        st.rerun()

    if ask_pressed and query_val.strip():
        question_text = query_val.strip()
        st.session_state.messages.append({"role": "user", "content": question_text})
        st.session_state.input_query = ""

        with st.spinner("Searching HR Manual and generating policy answer..."):
            response = pipeline.answer_question(question_text)
            st.session_state.messages.append({
                "role": "assistant",
                "content": response.get("answer", "Not found"),
                "pages": response.get("source_pages", []),
                "evidence": response.get("evidence", [])
            })
        st.rerun()

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    # ==========================================
    # 4. Suggested Question Cards ("Try asking")
    # ==========================================
    st.markdown("""
    <div>
        <h2 class="section-title">Try asking</h2>
        <p class="section-subtitle">Click any assignment question below to test the retrieval and reasoning engine:</p>
    </div>
    """, unsafe_allow_html=True)

    icons = ["🏖️", "⏱️", "🏥", "📊", "👶"]
    col_a, col_b = st.columns(2)

    for idx, sample_q in enumerate(ASSIGNMENT_QUESTIONS):
        target_col = col_a if idx % 2 == 0 else col_b
        icon = icons[idx % len(icons)]
        with target_col:
            button_label = f"{icon}  {sample_q}"
            if st.button(button_label, key=f"suggested_q_{idx}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": sample_q})
                with st.spinner("Searching HR Manual and generating policy answer..."):
                    res = pipeline.answer_question(sample_q)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": res.get("answer", "Not found"),
                        "pages": res.get("source_pages", []),
                        "evidence": res.get("evidence", [])
                    })
                st.rerun()

    # ==========================================
    # 5. Document Knowledge Base Footer
    # ==========================================
    st.markdown("<div style='height: 2.5rem;'></div>", unsafe_allow_html=True)
    with st.expander("ℹ️ Knowledge Base & Architecture Details"):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Source Manual", "hr_manual.pdf", "79 Pages")
        c2.metric("ChromaDB Chunks", f"{chunk_count}", "Persistent")
        c3.metric("Local Embeddings", "MiniLM-L6-v2", "ONNX")
        c4.metric("Engine Mode", "Strict RAG", "Zero Hallucination")
        st.caption("Answers are strictly derived from the official Enso Group HR Policy Manual. Questions without document proof return exactly `Not found`.")


if __name__ == "__main__":
    main()
