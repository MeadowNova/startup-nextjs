# ResumeForge Development Context - Handoff Document

## 🎯 **Project Overview**
Building **ResumeForge** - An AI-powered resume optimization platform that preserves original layout while optimizing content for ATS systems. We've successfully extracted valuable business logic from a mixed codebase and set up a clean foundation.

## 📍 **Current Status**
- ✅ **Completed**: Exported all valuable business logic to clean project structure
- 🔄 **In Progress**: Phase 1 - Backend Foundation Setup
- 🎯 **Current Task**: Create FastAPI Application Structure (UUID: ugu6hM2LzRPh7RdXyJNS85)

## 🏗️ **Project Structure Created**
```
startup-nextjs/
├── frontend/                    # Next.js app (from starter template)
├── backend/                     # Clean FastAPI backend
│   ├── app/
│   │   ├── core/
│   │   │   └── config.py       # ✅ Configuration management
│   │   ├── services/           # ✅ All exported business logic
│   │   │   ├── ai_processor.py      # AI prompts & processing
│   │   │   ├── pdf_reconstructor.py # Layout preservation magic
│   │   │   ├── blob_storage.py      # Vercel Blob Storage integration
│   │   │   └── package_creator.py   # ZIP package creation
│   │   ├── models/             # 🔄 Need to create database models
│   │   └── api/v1/             # 🔄 Need to create API endpoints
│   ├── alembic/
│   │   └── versions/
│   │       └── 001_initial_migration.py # ✅ Database schema
│   ├── alembic.ini             # ✅ Migration config
│   └── requirements.txt        # ✅ Dependencies
├── shared/                     # Ready for shared types
├── reference.md               # ✅ The proven algorithm
└── RESUMEFORGE_README.md      # ✅ Project documentation
```

## 🧠 **Key Business Logic Preserved**

### **AI Processing (`ai_processor.py`)**
- **Resume optimization prompts** - Carefully crafted for ATS optimization
- **Cover letter generation** - Professional, targeted prompts
- **GPT-4o Vision layout extraction** - Core innovation for preserving layout
- **Token usage tracking** and comprehensive error handling

### **PDF Reconstruction (`pdf_reconstructor.py`)**
- **Layout-aware PDF generation** using ReportLab + GPT-4o coordinates
- **Markdown parsing** for structured content sections
- **Font matching** and styling preservation
- **Text wrapping** algorithms within bounding boxes

### **File Management (`blob_storage.py`)**
- **Vercel Blob Storage** integration with local fallback
- **File type validation** and security measures
- **Upload/download** with comprehensive error handling

### **Package Creation (`package_creator.py`)**
- **Professional ZIP packages** with resume + cover letter
- **Intelligent file naming** based on job details
- **README generation** and application instructions

## 🗄️ **Database Schema (Ready to Implement)**
Based on `001_initial_migration.py`:
- **User** - Authentication, subscription tiers, profile data
- **ProcessingJob** - Complete job tracking with AI metadata
- **JobHistory** - User activity tracking
- **UserSubscriptions** - Subscription management
- **UsageTracking** - API usage monitoring

## 📋 **Task Management Status**

### **Current Phase: Backend Foundation Setup**
- [🔄] **Create FastAPI Application Structure** - IN PROGRESS
- [ ] Create Database Models
- [ ] Set Up Database Connection
- [ ] Implement Authentication System
- [ ] Configure Environment & Settings

### **Upcoming Phases**
1. **Phase 2**: Core Services Integration (fix imports, test services)
2. **Phase 3**: API Development (endpoints, background tasks)
3. **Phase 4**: Frontend Development (React/Next.js UI)
4. **Phase 5**: Integration & Testing (E2E, deployment)

## 🔧 **Immediate Next Steps**

### **Current Task: Create FastAPI Application Structure**
You need to create:

1. **`backend/main.py`** - FastAPI app factory with:
   - CORS configuration for frontend
   - Middleware setup (logging, error handling)
   - Basic routing structure
   - Health check endpoints

2. **`backend/app/database.py`** - Database connection:
   - SQLAlchemy engine setup
   - Session management
   - Dependency injection for database sessions

3. **`backend/app/models/database.py`** - SQLAlchemy models:
   - User model with authentication fields
   - ProcessingJob model with AI metadata
   - Relationships and indexes

## 🔑 **Environment Variables Needed**
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

## 🚀 **The Proven Algorithm (from reference.md)**
1. **Extract text** from resume PDF (PyMuPDF)
2. **AI optimization** with GPT-4o-mini using targeted prompts
3. **Layout extraction** with GPT-4o Vision for bounding boxes
4. **PDF reconstruction** using ReportLab with preserved layout
5. **Cover letter generation** matching the resume style
6. **Package creation** with professional organization

## 🎯 **Success Criteria**
When Phase 1 is complete, you should have:
- ✅ Working FastAPI application that starts without errors
- ✅ Database models that match the migration schema
- ✅ Authentication system with JWT tokens
- ✅ All services properly imported and dependency-injected
- ✅ Environment configuration working

## 📝 **Important Notes**

### **Service Import Fixes Needed**
The exported services have import statements that need updating:
- Change `from app.config import settings` to `from app.core.config import settings`
- Update any database imports to use the new structure
- Fix any circular import issues

### **Key Files to Reference**
- `reference.md` - The proven algorithm and workflow
- `backend/app/core/config.py` - Configuration management
- `backend/alembic/versions/001_initial_migration.py` - Database schema
- `RESUMEFORGE_README.md` - Project documentation

### **Development Workflow**
1. Start with backend foundation (current phase)
2. Test each service individually before integration
3. Build API endpoints incrementally
4. Frontend development after backend is stable
5. End-to-end testing before deployment

## 🔄 **How to Continue**
1. **Load this context** in your next chat
2. **Reference the task list** using `view_tasklist` tool
3. **Continue with current task**: Create FastAPI Application Structure
4. **Update task status** as you complete each item
5. **Use the exported services** as your business logic foundation

## 💡 **Key Advantages Achieved**
- ✅ **Clean architecture** without platform dependencies
- ✅ **Proven AI algorithms** preserved and organized
- ✅ **Production-ready structure** from day one
- ✅ **Comprehensive task management** for tracking progress
- ✅ **All valuable IP** extracted and properly structured

**Ready to build a production-ready ResumeForge application!** 🚀
