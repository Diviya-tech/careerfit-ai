"""
CareerFit AI — Streamlit Application
AI-powered resume analyzer built with RAG (Retrieval Augmented Generation).

Architecture:
    User Input → Pass 1: Extraction → RAG Pipeline → Pass 2: LLM Analysis → Pass 3: Results
    
Modules:
    src/resume_parser.py    — PDF/DOCX text extraction
    src/job_parser.py       — Job description parsing + live job fetching
    src/skill_extractor.py  — LLM skill extraction + similarity detection
    src/rag_pipeline.py     — Chunking, embedding, vector store, retrieval
    src/match_score.py      — Scoring logic + learning roadmap generation
"""

import streamlit as st
from src.resume_parser import extract_text
from src.job_parser import fetch_live_jobs
from src.skill_extractor import extract_skills_from_text, find_skill_matches
from src.rag_pipeline import build_vectorstore, retrieve_relevant_context
from src.match_score import get_llm, analyze_fit, get_learning_roadmap

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(page_title="CareerFit AI", page_icon="🎯", layout="wide")

st.title("🎯 CareerFit AI")
st.markdown("*Upload your resume + paste a job description → get your match score, skill gaps, and rewritten bullets*")
st.divider()

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.header("🔑 API Keys")
    groq_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
    rapidapi_key = st.text_input("RapidAPI Key", type="password", placeholder="your rapidapi key")
    st.divider()
    st.header("🔍 Job Search Settings")
    job_title = st.text_input("Target Job Title", placeholder="e.g. Data Scientist")
    job_location = st.text_input("Location", placeholder="e.g. New York or Remote")
    fetch_jobs = st.button("🌐 Fetch Live Job Postings", use_container_width=True)

# ── Main UI ───────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("📄 Your Resume")
    resume_file = st.file_uploader("Upload Resume", type=["pdf", "docx"])

with col2:
    st.subheader("📋 Job Description")
    jd_input_method = st.radio("Input method", ["Paste text", "Upload PDF"], horizontal=True)
    if jd_input_method == "Paste text":
        jd_text_input = st.text_area("Paste job description here", height=200)
    else:
        jd_file = st.file_uploader("Upload JD", type=["pdf", "docx"])

# ── Fetch Live Jobs ───────────────────────────────────────────
if fetch_jobs:
    if not rapidapi_key or not job_title:
        st.sidebar.error("Enter your RapidAPI key and job title first!")
    else:
        with st.spinner("🔍 Fetching live job postings from Dice & LinkedIn..."):
            jobs = fetch_live_jobs(job_title, job_location or "Remote", rapidapi_key)
            if jobs:
                st.subheader(f"🌐 Live Job Postings for '{job_title}'")
                for i, job in enumerate(jobs):
                    with st.expander(f"#{i+1} {job.get('job_title', 'N/A')} @ {job.get('employer_name', 'N/A')}"):
                        st.write(f"📍 **Location:** {job.get('job_city', 'Remote')}")
                        st.write(f"🔗 **Apply:** {job.get('job_apply_link', 'N/A')}")
                        desc = job.get('job_description', '')[:500]
                        st.write(f"📝 **Description:** {desc}...")
            else:
                st.sidebar.error("No jobs found. Check your API key or try different search terms.")

st.divider()

# ── Analyze Button ────────────────────────────────────────────
analyze_btn = st.button("🚀 Analyze My Fit", use_container_width=True, type="primary")

if analyze_btn:
    if not groq_key:
        st.error("Please enter your Groq API key in the sidebar!")
    elif not resume_file:
        st.error("Please upload your resume!")
    else:
        # Extract texts
        resume_text = extract_text(resume_file)

        if jd_input_method == "Paste text":
            jd_text = jd_text_input
        else:
            if 'jd_file' in locals() and jd_file:
                jd_text = extract_text(jd_file)
            else:
                st.error("Please upload or paste a job description!")
                st.stop()

        if not jd_text.strip():
            st.error("Job description is empty!")
            st.stop()

        llm = get_llm(groq_key)

        # ── PASS 1: Extract Skills ────────────────────────────
        with st.spinner("🔍 Pass 1/3: Extracting skills from your resume..."):
            try:
                resume_skills = extract_skills_from_text(resume_text, "resume", llm)
            except Exception as e:
                st.error(f"Error parsing resume: {e}")
                st.stop()

        with st.spinner("🔍 Pass 1/3: Extracting requirements from job description..."):
            try:
                jd_skills = extract_skills_from_text(jd_text, "job description", llm)
            except Exception as e:
                st.error(f"Error parsing JD: {e}")
                st.stop()

        # ── Skill Similarity Detection ────────────────────────
        with st.spinner("🧬 Computing skill similarity scores..."):
            try:
                resume_skill_list = resume_skills.get("technical_skills", [])
                jd_skill_list = jd_skills.get("technical_skills", [])
                skill_matches = find_skill_matches(resume_skill_list, jd_skill_list)
            except Exception as e:
                st.warning(f"Skill similarity detection failed: {e}")
                skill_matches = {'exact_matches': [], 'partial_matches': [], 'missing_skills': jd_skill_list}

        # ── RAG: Build Vector Store ───────────────────────────
        with st.spinner("🧠 Building RAG index from your resume..."):
            try:
                vectorstore = build_vectorstore(resume_text)
                jd_requirements = " ".join(jd_skills.get("technical_skills", []) +
                                           jd_skills.get("key_responsibilities", []))
                rag_context = retrieve_relevant_context(vectorstore, jd_requirements, k=6)
            except Exception as e:
                st.warning(f"RAG indexing failed, proceeding without it: {e}")
                rag_context = resume_text[:2000]

        # ── PASS 2: Deep Analysis ─────────────────────────────
        with st.spinner("🤖 Pass 2/3: Deep analysis — comparing your fit..."):
            try:
                result = analyze_fit(resume_skills, jd_skills, rag_context, jd_text, skill_matches, llm)
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                st.stop()

        # ════════════════════════════════════════════════════════
        # DISPLAY RESULTS
        # ════════════════════════════════════════════════════════

        # Scores
        st.subheader("📊 Match Analysis")
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("🎯 Overall Fit", f"{result['overall_score']}%")
        col_b.metric("💻 Technical Skills", f"{result['technical_skills_score']}%")
        col_c.metric("💼 Experience", f"{result['experience_score']}%")
        col_d.metric("🎓 Education", f"{result['education_score']}%")

        score = result['overall_score']
        if score >= 75:
            st.success(f"✅ Strong Match! {result['verdict']}")
        elif score >= 50:
            st.warning(f"⚠️ Moderate Match. {result['verdict']}")
        else:
            st.error(f"❌ Weak Match. {result['verdict']}")
        st.progress(score / 100)
        st.divider()

        # Skill Similarity Results
        col_e, col_f, col_g = st.columns(3)
        with col_e:
            st.subheader("✅ Matched Skills")
            for skill in result.get('matched_skills', []):
                st.success(f"✓ {skill}")

        with col_f:
            st.subheader("🔶 Partial Matches")
            for skill in result.get('partial_matches', []):
                st.warning(f"~ {skill}")
            # Show similarity scores for partial matches
            if skill_matches.get('partial_matches'):
                st.caption("Detected via embedding similarity:")
                for r_skill, j_skill, sim_score in skill_matches['partial_matches']:
                    st.caption(f"  {r_skill} ≈ {j_skill} ({sim_score:.0%})")

        with col_g:
            st.subheader("❌ Missing Skills")
            for skill in result.get('missing_skills', []):
                st.error(f"✗ {skill}")

        st.divider()

        # Strengths & Weaknesses
        col_h, col_i = st.columns(2)
        with col_h:
            st.subheader("💪 Strengths")
            for s in result.get('strengths', []):
                st.write(f"• {s}")
        with col_i:
            st.subheader("🔧 Areas to Improve")
            for w in result.get('weaknesses', []):
                st.write(f"• {w}")

        st.divider()

        # Rewritten Bullets
        st.subheader("✏️ AI-Rewritten Resume Bullets (Tailored to This JD)")
        st.caption("Copy these into your resume to improve keyword match")
        for i, bullet in enumerate(result.get('rewritten_bullets', []), 1):
            st.info(f"{i}. {bullet}")

        st.divider()

        # Interview Risk Questions
        st.subheader("🎤 Likely Interview Questions (Based on Your Gaps)")
        st.caption("Prepare answers for these — recruiters WILL ask about your gaps")
        for q in result.get('interview_risk_questions', []):
            st.warning(f"❓ {q}")

        st.divider()

        # ── PASS 3: Learning Roadmap ──────────────────────────
        missing = result.get('missing_skills', [])
        partial = [p for p in result.get('partial_matches', [])]
        if missing or partial:
            st.subheader("🗺️ Your Personalized Learning Roadmap")
            st.caption("How to close your skill gaps — prioritized by impact")
            with st.spinner("🤖 Pass 3/3: Building your personalized roadmap..."):
                try:
                    roadmap_data = get_learning_roadmap(missing, partial, llm)

                    if roadmap_data.get('priority_order'):
                        st.info(f"📌 **Priority:** {roadmap_data['priority_order']}")
                    if roadmap_data.get('total_estimated_time'):
                        st.info(f"⏱️ **Total Time:** {roadmap_data['total_estimated_time']}")

                    for item in roadmap_data.get('roadmap', []):
                        with st.expander(f"📚 {item['skill']} — {item['time_to_learn']} ({item.get('difficulty', 'N/A')})"):
                            st.write(f"**Gap:** {item.get('current_gap', 'N/A')}")
                            st.write("**Free Resources:**")
                            for res in item.get('free_resources', []):
                                st.write(f"  🔗 {res}")
                            st.write(f"**Practice Project:** {item.get('practice_project', 'N/A')}")
                            st.write(f"**Add to Resume as:** _{item.get('resume_line', 'N/A')}_")

                except Exception as e:
                    st.error(f"Roadmap generation failed: {e}")

        # ── Save to History ───────────────────────────────────
        if 'history' not in st.session_state:
            st.session_state.history = []

        st.session_state.history.append({
            "score": result['overall_score'],
            "tech_score": result['technical_skills_score'],
            "exp_score": result['experience_score'],
            "edu_score": result['education_score'],
            "verdict": result['verdict'],
            "matched": len(result.get('matched_skills', [])),
            "missing": len(result.get('missing_skills', []))
        })

# ── Application History ───────────────────────────────────────
if 'history' in st.session_state and st.session_state.history:
    st.divider()
    st.subheader("📁 Application History (This Session)")
    for i, h in enumerate(st.session_state.history):
        col_x, col_y, col_z = st.columns([2, 1, 1])
        with col_x:
            st.write(f"**Application #{i+1}** — {h['verdict']}")
        with col_y:
            st.write(f"🎯 {h['score']}% | 💻 {h['tech_score']}% | 💼 {h['exp_score']}%")
        with col_z:
            st.write(f"✅ {h['matched']} matched | ❌ {h['missing']} missing")
