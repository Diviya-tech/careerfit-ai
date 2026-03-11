# 🎯 CareerFit AI — AI-Powered Resume Analyzer

**Stop guessing if you fit the role. Know it.**

CareerFit AI is an AI-powered resume analysis tool built with RAG (Retrieval Augmented Generation) that gives you an honest, structured match score against any job description — not just a vague "you're a good fit."

🔗 **Live App:** [Try it on Streamlit](https://careerfit-ai.streamlit.app)

---

## 📌 The Problem

Every job seeker does this:
- Copy resume → paste into ChatGPT → "Am I a good fit?" → get a vague answer → repeat

The problem? You're blindly relying on a generic AI tool that:
- Gives inflated scores to be "helpful"
- Doesn't break down WHERE you fall short
- Forgets everything when you start a new chat
- Provides no structured, repeatable analysis

**CareerFit AI solves this** by giving you a brutally honest, structured analysis every single time — broken down by technical skills, experience, and education — with actionable steps to close your gaps.

---

## 🧠 How It Works — The RAG Pipeline

This isn't just a ChatGPT wrapper. Here's what happens under the hood:

### What is RAG?

RAG (Retrieval Augmented Generation) bridges the gap between what an LLM knows and what it *needs* to know. Instead of sending your entire resume to the AI and hoping it catches everything, RAG:

1. **Chunks** your resume into small, meaningful pieces (300 words each)
2. **Embeds** each chunk into a vector — a list of numbers that captures the *meaning* of your words
3. **Stores** these vectors in ChromaDB (a vector database)
4. **Retrieves** only the most relevant resume sections when a job description comes in
5. **Generates** a focused, accurate analysis using only what matters

**Think of it like this:**
- Without RAG = closed-book exam (the AI tries to remember everything at once)
- With RAG = open-book exam (the AI knows exactly which page to flip to)

### The 3-Pass Analysis Pipeline

| Pass | What It Does | Why It Matters |
|------|-------------|----------------|
| **Pass 1: Extract** | Separately parses skills from resume AND job description | Focused extraction > one giant prompt |
| **Pass 2: Compare** | Uses RAG-retrieved context + extracted skills for deep comparison | Only relevant resume sections are analyzed against JD requirements |
| **Pass 3: Roadmap** | Generates personalized learning plan for missing skills | Actionable next steps, not just a score |

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **LLM** | OpenAI GPT-OSS 120B (via Groq) | Powers all analysis and generation |
| **RAG Framework** | LangChain | Orchestrates the retrieval-augmented pipeline |
| **Vector Database** | ChromaDB | Stores and retrieves resume embeddings |
| **Embeddings** | HuggingFace `all-MiniLM-L6-v2` | Converts text chunks into semantic vectors |
| **PDF Processing** | PyPDF | Extracts text from uploaded resume/JD PDFs |
| **Job Data** | JSearch API (RapidAPI) | Fetches live job postings from LinkedIn & Dice |
| **Frontend** | Streamlit | Interactive web application |
| **Inference API** | Groq Cloud | Ultra-fast, free LLM inference |

---

## 📊 What You Get

Upload your resume + paste any job description and CareerFit AI gives you:

| Feature | Description |
|---------|-------------|
| 🎯 **Match Score** | Overall fit percentage broken down by Technical Skills, Experience, and Education |
| ✅ **Matched Skills** | Skills that genuinely appear in both your resume and the JD |
| ❌ **Missing Skills** | Skills required by the JD but not found in your resume |
| 💪 **Strengths** | What makes you a strong candidate, with evidence |
| 🔧 **Weaknesses** | Specific gaps with honest explanations |
| ✏️ **Rewritten Bullets** | Your resume bullets rewritten to match the JD's keywords |
| 🗺️ **Learning Roadmap** | Personalized plan to close skill gaps with free resources and time estimates |
| 🌐 **Live Job Postings** | Real-time job listings from LinkedIn and Dice via JSearch API |
| 📁 **Application History** | Track scores across multiple job applications in one session |

### Demo Screenshots

**Match Analysis:**

![Match Analysis](screenshots/match_analysis.png)

**Skills Breakdown:**

![Skills](screenshots/skills_breakdown.png)

**Learning Roadmap:**

![Roadmap](screenshots/learning_roadmap.png)

---

## 🚀 Getting Started

### Prerequisites

You'll need two free API keys:

| Key | Where to Get It | Cost |
|-----|----------------|------|
| **Groq API Key** | [console.groq.com](https://console.groq.com) | Free |
| **RapidAPI Key** | [rapidapi.com/letscrape-6bRBa3QguO5/api/jsearch](https://rapidapi.com/letscrape-6bRBa3QguO5/api/jsearch) | Free (200 requests/month) |

### Installation

```bash
# Clone the repo
git clone https://github.com/Diviya-tech/careerfit-ai.git
cd careerfit-ai

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

### Usage

1. Open the app in your browser (localhost:8501)
2. Enter your Groq and RapidAPI keys in the sidebar
3. Upload your resume PDF
4. Paste a job description (or upload as PDF)
5. Click **Analyze My Fit**
6. Review your scores, gaps, and recommendations

---

## 📁 Project Structure

```
careerfit-ai/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── .gitignore            # Excludes .env and other sensitive files
└── screenshots/          # Demo screenshots
    ├── match_analysis.png
    ├── skills_breakdown.png
    └── learning_roadmap.png
```

---

## 🔮 What's Next

- [ ] **CareerPath AI** — Scrapes live job postings and builds a personalized learning roadmap based on market demand
- [ ] **Interview Prep Mode** — Generates role-specific interview questions based on your gaps
- [ ] **Resume Version Tracking** — Compare how your fit score improves as you update your resume
- [ ] **Multi-format Support** — Accept DOCX and TXT resumes in addition to PDF

---

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.

---

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

---

## 👤 Author

**Your Name**
- LinkedIn: [your-linkedin](https://linkedin.com/in/sridivyadasari)
- GitHub: [your-github](https://github.com/Diviya-tech)

---

*Built with ❤️ as part of my GenAI portfolio — proving that the best way to understand AI is to build with it.*
