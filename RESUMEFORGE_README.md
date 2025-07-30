# ResumeForge - AI-Powered Resume Optimization

A production-ready application that optimizes resumes for ATS systems while preserving the original layout and visual design.

## Project Structure

```
startup-nextjs/
├── frontend/                 # Next.js frontend (from starter template)
│   ├── src/
│   │   ├── app/             # Next.js 13+ app directory
│   │   ├── components/      # React components
│   │   └── styles/          # CSS/styling
│   └── package.json
│
├── backend/                 # Clean FastAPI backend
│   ├── app/
│   │   ├── core/           # Core configuration
│   │   │   └── config.py   # Environment & settings management
│   │   ├── services/       # Business logic services
│   │   │   ├── ai_processor.py      # AI prompts & processing
│   │   │   ├── pdf_reconstructor.py # Layout-aware PDF generation
│   │   │   ├── blob_storage.py      # File storage (Vercel Blob)
│   │   │   └── package_creator.py   # ZIP package creation
│   │   ├── models/         # Database models (to be created)
│   │   └── api/            # API endpoints (to be created)
│   │       └── v1/
│   ├── alembic/            # Database migrations
│   │   └── versions/
│   │       └── 001_initial_migration.py
│   └── requirements.txt
│
├── shared/                 # Shared types/schemas (to be created)
├── reference.md           # The proven algorithm documentation
└── README.md
```

## Key Components Imported

### 🧠 AI Processing (`ai_processor.py`)
- **Resume optimization prompts** - Carefully crafted for ATS optimization
- **Cover letter generation** - Professional, targeted prompts  
- **GPT-4o Vision layout extraction** - Core innovation for preserving layout
- **Token usage tracking** and error handling

### 📄 PDF Reconstruction (`pdf_reconstructor.py`)
- **Layout-aware PDF generation** using ReportLab + GPT-4o coordinates
- **Markdown parsing** for structured content
- **Font matching** and styling preservation
- **Text wrapping** within bounding boxes

### 💾 File Management (`blob_storage.py`)
- **Vercel Blob Storage** integration
- **Local storage fallback** for development
- **File type validation** and security
- **Upload/download** with error handling

### 📦 Package Creation (`package_creator.py`)
- **Professional ZIP packages** with resume + cover letter
- **Intelligent file naming** based on job details
- **README generation** and instructions
- **Metadata tracking**

### ⚙️ Configuration (`config.py`)
- **Pydantic-based settings** with validation
- **Environment-specific configs** (dev/prod/test)
- **OpenAI, database, and storage** configuration
- **Security and rate limiting** settings

### 🗄️ Database Schema (`001_initial_migration.py`)
- **User management** with subscription tiers
- **ProcessingJob tracking** with comprehensive metadata
- **AI analysis results** storage (JSON fields)
- **Usage tracking** and job history
- **Proper indexes** and relationships

## The Proven Algorithm

Based on `reference.md`, the core workflow is:

1. **Extract text** from resume PDF (PyMuPDF)
2. **AI optimization** with GPT-4o-mini using targeted prompts
3. **Layout extraction** with GPT-4o Vision for bounding boxes
4. **PDF reconstruction** using ReportLab with preserved layout
5. **Cover letter generation** matching the resume style
6. **Package creation** with professional organization

## Next Steps

1. **Set up the backend FastAPI application**
2. **Create database models** based on the migration
3. **Implement API endpoints** for the processing workflow
4. **Build the frontend interface** for file upload and status tracking
5. **Set up authentication** (Firebase or similar)
6. **Configure deployment** (Vercel for frontend, backend hosting)

## Environment Variables Needed

```bash
# Database
DATABASE_URL=postgresql://...

# OpenAI
OPENAI_API_KEY=sk-...

# Blob Storage
BLOB_READ_WRITE_TOKEN=...

# Security
SECRET_KEY=...

# Optional
SENTRY_DSN=...
```

## Development Setup

```bash
# Backend
cd backend
pip install -r requirements.txt
alembic upgrade head

# Frontend  
cd ../frontend
npm install
npm run dev
```

This structure gives you a clean, production-ready foundation with all the valuable AI processing logic and proven algorithms preserved.
