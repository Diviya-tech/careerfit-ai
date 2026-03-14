"""
Skill Extractor Module
Uses LLM to extract structured skills from resume and job descriptions.
Includes skill similarity detection using cosine similarity.
"""

import json
import numpy as np
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings


# ── Skill Similarity Detection ─────────────────────────────────
# This is what makes CareerFit AI smarter than a keyword matcher.
# Instead of treating "PyTorch" and "TensorFlow" as completely different,
# we use embedding similarity to detect related skills.

_embeddings_model = None

def _get_embeddings_model():
    """Lazy load the embeddings model (singleton)."""
    global _embeddings_model
    if _embeddings_model is None:
        _embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return _embeddings_model


def compute_skill_similarity(skill_a, skill_b):
    """
    Compute cosine similarity between two skill strings using embeddings.
    
    Args:
        skill_a: First skill string
        skill_b: Second skill string
        
    Returns:
        float: Cosine similarity score (0.0 to 1.0)
    """
    model = _get_embeddings_model()
    vec_a = model.embed_query(skill_a)
    vec_b = model.embed_query(skill_b)
    
    # Cosine similarity
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return float(dot_product / (norm_a * norm_b))


def find_skill_matches(resume_skills, jd_skills, exact_threshold=0.82, partial_threshold=0.55):   
    """
    Compare resume skills against JD skills using embedding similarity.
    
    Instead of exact string matching, this uses semantic similarity:
    - "PyTorch" vs "TensorFlow" → partial match (both are DL frameworks)
    - "Python" vs "Python" → exact match
    - "Python" vs "Kubernetes" → no match
    
    Args:
        resume_skills: List of skills from resume
        jd_skills: List of skills from job description
        exact_threshold: Similarity score for exact match (default 0.92)
        partial_threshold: Similarity score for partial match (default 0.75)
        
    Returns:
        dict: {
            'exact_matches': [(resume_skill, jd_skill, score)],
            'partial_matches': [(resume_skill, jd_skill, score)],
            'missing_skills': [jd_skill]
        }
    """
    exact_matches = []
    partial_matches = []
    matched_jd_skills = set()

    for jd_skill in jd_skills:
        best_score = 0
        best_resume_skill = None

        for resume_skill in resume_skills:
            score = compute_skill_similarity(resume_skill.lower(), jd_skill.lower())
            if score > best_score:
                best_score = score
                best_resume_skill = resume_skill

        if best_score >= exact_threshold:
            exact_matches.append((best_resume_skill, jd_skill, round(best_score, 3)))
            matched_jd_skills.add(jd_skill)
        elif best_score >= partial_threshold:
            partial_matches.append((best_resume_skill, jd_skill, round(best_score, 3)))
            matched_jd_skills.add(jd_skill)

    missing_skills = [s for s in jd_skills if s not in matched_jd_skills]

    return {
        'exact_matches': exact_matches,
        'partial_matches': partial_matches,
        'missing_skills': missing_skills
    }


# ── LLM-Powered Skill Extraction ──────────────────────────────

def extract_skills_from_text(text, doc_type, llm):
    """
    Extract structured skills from a resume or job description using LLM.
    
    Args:
        text: Document text content
        doc_type: Either 'resume' or 'job description'
        llm: Initialized ChatGroq LLM instance
        
    Returns:
        dict: Structured skills data including technical_skills, soft_skills, etc.
    """
    prompt = f"""You are an expert resume parser. Extract ALL skills from this {doc_type}.

IMPORTANT RULES:
- Include BOTH specific tools (e.g. "Scikit-learn") AND broad categories (e.g. "Machine Learning")
- If coursework mentions a skill, include it
- If a project demonstrates a skill, include it (e.g. neural network project = "Deep Learning")
- If they use visualization tools, include "Data Visualization" as a skill
- If they use statistical libraries, include "Statistical Analysis" as a skill
- Extract EVERY skill you can find — more is better than less

TEXT:
{text[:6000]}

Return ONLY valid JSON, no extra text:
{{
    "technical_skills": [<list ALL technical skills — both specific tools AND broad categories>],
    "soft_skills": [<list of soft skills mentioned or implied>],
    "experience_years": "<estimated total years of experience or 'Not specified'>",
    "education": [<list of degrees, certifications>],
    "key_responsibilities": [<list of top 5 responsibilities or requirements>],
    "industry_keywords": [<list of domain-specific keywords>]
}}
"""
    response = llm.invoke(prompt)
    raw = response.content.strip()
    
    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0].strip()
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0].strip()
    
    return json.loads(raw)
