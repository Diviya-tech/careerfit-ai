"""
Match Score Module
Handles the core analysis: comparing resume vs JD using RAG context,
generating scores, and creating personalized learning roadmaps.
"""

import json
from langchain_groq import ChatGroq


def get_llm(groq_key):
    """
    Initialize the Groq LLM instance.
    
    Args:
        groq_key: Groq API key
        
    Returns:
        ChatGroq: Initialized LLM instance
    """
    return ChatGroq(
        groq_api_key=groq_key,
        model_name="openai/gpt-oss-120b",
        temperature=0.1
    )


def analyze_fit(resume_skills, jd_skills, rag_context, jd_text, skill_matches, llm):
    """
    Deep comparison of resume vs JD using extracted skills, 
    RAG-retrieved context, and skill similarity results.
    
    This is Pass 2 of the 3-pass pipeline. It combines:
    - Structured skill data from Pass 1
    - RAG-retrieved relevant resume sections
    - Skill similarity scores for partial matches
    
    Args:
        resume_skills: Dict of extracted resume skills
        jd_skills: Dict of extracted JD requirements
        rag_context: RAG-retrieved relevant resume sections
        jd_text: Full job description text
        skill_matches: Results from skill similarity detection
        llm: Initialized ChatGroq instance
        
    Returns:
        dict: Complete analysis results with scores and recommendations
    """
    # Format skill similarity results for the prompt
    exact = [f"{r} ↔ {j} (score: {s})" for r, j, s in skill_matches.get('exact_matches', [])]
    partial = [f"{r} ≈ {j} (score: {s})" for r, j, s in skill_matches.get('partial_matches', [])]
    missing = skill_matches.get('missing_skills', [])

    prompt = f"""You are a senior technical recruiter with 15 years of experience. 
You must perform a STRICT and HONEST evaluation. Do NOT inflate scores.

SCORING RULES:
- A skill only counts as "matched" if confirmed by the similarity analysis below
- Partial matches (score 0.75-0.92) should count as half credit
- If the JD requires 5+ years and resume shows 2 years, experience_score should be LOW
- education_score: 100 if degree matches exactly, 70 if related field, 40 if unrelated
- overall_score = 40% technical + 30% experience + 20% education + 10% soft skills
- Be harsh but fair — 90+ means near-perfect match

SKILL SIMILARITY ANALYSIS (computed via embedding cosine similarity):
Exact Matches: {json.dumps(exact)}
Partial Matches: {json.dumps(partial)}
Missing Skills: {json.dumps(missing)}

RESUME SKILLS EXTRACTED:
{json.dumps(resume_skills, indent=2)}

JOB DESCRIPTION REQUIREMENTS:
{json.dumps(jd_skills, indent=2)}

MOST RELEVANT RESUME SECTIONS (retrieved via RAG):
{rag_context}

FULL JOB DESCRIPTION:
{jd_text[:2000]}

Provide analysis in this EXACT JSON format ONLY:
{{
    "overall_score": <integer 0-100>,
    "technical_skills_score": <integer 0-100>,
    "experience_score": <integer 0-100>,
    "education_score": <integer 0-100>,
    "matched_skills": [<ONLY skills confirmed as exact matches>],
    "partial_matches": [<skills detected as similar but not exact>],
    "missing_skills": [<skills NOT found in resume at all>],
    "strengths": [<3 specific strengths with evidence>],
    "weaknesses": [<3 specific gaps with explanation>],
    "rewritten_bullets": [<5 resume bullet points rewritten to match JD keywords>],
    "interview_risk_questions": [<3 questions recruiters will ask based on gaps>],
    "verdict": "<2 sentence honest assessment>"
}}
"""
    response = llm.invoke(prompt)
    raw = response.content.strip()

    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0].strip()
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0].strip()

    return json.loads(raw)


def get_learning_roadmap(missing_skills, partial_matches, llm):
    """
    Generate a personalized learning roadmap for missing and partial skills.
    
    This is Pass 3 of the pipeline. It creates actionable next steps
    with specific resources, time estimates, and practice projects.
    
    Args:
        missing_skills: List of skills not found in resume
        partial_matches: List of skills with partial match
        llm: Initialized ChatGroq instance
        
    Returns:
        dict: Structured learning roadmap
    """
    all_gaps = missing_skills[:5] + (partial_matches[:2] if partial_matches else [])
    skills_str = ", ".join(all_gaps)

    prompt = f"""You are a career development advisor. Create a realistic learning roadmap 
for someone who needs to learn these skills: {skills_str}

Be SPECIFIC with resources — give actual course names and platforms.
Do NOT make up URLs.

Return ONLY valid JSON:
{{
    "roadmap": [
        {{
            "skill": "<skill name>",
            "current_gap": "<what they're missing specifically>",
            "time_to_learn": "<realistic time estimate>",
            "difficulty": "<beginner/intermediate/advanced>",
            "free_resources": [
                "<specific resource 1 — name + platform>",
                "<specific resource 2 — name + platform>"
            ],
            "practice_project": "<specific project idea to demonstrate this skill>",
            "resume_line": "<exact bullet point to add to resume once learned>"
        }}
    ],
    "priority_order": "<which skill to learn first and why>",
    "total_estimated_time": "<total time to close all gaps>"
}}
"""
    response = llm.invoke(prompt)
    raw = response.content.strip()

    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0].strip()
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0].strip()

    return json.loads(raw)
