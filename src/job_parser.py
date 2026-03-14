"""
Job Parser Module
Handles job description text input and extraction from PDF/DOCX.
"""

import requests
from .resume_parser import extract_text


def extract_jd_text(jd_file=None, jd_text=None):
    """
    Extract job description text from either pasted text or uploaded file.
    
    Args:
        jd_file: Uploaded file object (optional)
        jd_text: Pasted text string (optional)
        
    Returns:
        str: Job description text
    """
    if jd_text and jd_text.strip():
        return jd_text.strip()
    elif jd_file:
        return extract_text(jd_file)
    return ""


def fetch_live_jobs(job_title, location, rapidapi_key, num_results=10):
    """
    Fetch live job postings from JSearch API (LinkedIn & Dice).
    
    Args:
        job_title: Target job title to search
        location: Job location or 'Remote'
        rapidapi_key: RapidAPI key for JSearch
        num_results: Number of results to return
        
    Returns:
        list: List of job posting dictionaries
    """
    url = "https://jsearch.p.rapidapi.com/search"
    headers = {
        "X-RapidAPI-Key": rapidapi_key,
        "X-RapidAPI-Host": "jsearch.p.rapidapi.com"
    }
    params = {
        "query": f"{job_title} in {location}",
        "num_pages": "1",
        "page": "1"
    }

    try:
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            data = response.json()
            return data.get("data", [])[:num_results]
    except Exception:
        pass

    return []
