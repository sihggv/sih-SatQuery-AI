<div align="center">

![SatQuery AI Header](https://i.ibb.co/Nn7PDCC8/Screenshot-2026-09-27-091747.png)

# 🛰️ SatQuery AI
### An Interactive Vision-Language Assistant for Multimodal Remote Sensing Image Analysis through Text Queries

[![Smart India Hackathon](https://img.shields.io/badge/Smart%20India%20Hackathon-2026-orange?style=for-the-badge)](https://sih.gov.in/)
[![ISRO](https://img.shields.io/badge/ISRO-SAC-blue?style=for-the-badge)](https://www.isro.gov.in/)
[![Made in India](https://img.shields.io/badge/Made%20in-India%20🇮🇳-green?style=for-the-badge)](https://www.india.gov.in/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![React](https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react)](https://reactjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

**🛰️ Making Satellite Imagery Understandable Through Natural Language**

[Features](#-key-features) • [Demo](#-demo-screenshots) • [Installation](#-installation-guide) • [Tech Stack](#-tech-stack) • [Team](#-team)

</div>

---

## 📖 Table of Contents

1. [What is SatQuery AI?](#-what-is-satquery-ai)
2. [The Problem We Solve](#-the-problem-we-solve)
3. [Our Solution](#-our-solution)
4. [Key Features](#-key-features)
5. [Demo Screenshots](#-demo-screenshots)
6. [How It Works](#-how-it-works)
7. [Supported Inputs & Tasks](#-supported-inputs--tasks)
8. [Tech Stack](#-tech-stack)
9. [Installation Guide](#-installation-guide)
10. [Running the Project](#-running-the-project)
11. [Project Structure](#-project-structure)
12. [API Documentation](#-api-documentation)
13. [Demo Credentials](#-demo-credentials)
14. [Datasets & Benchmarks](#-datasets--benchmarks)
15. [Troubleshooting](#-troubleshooting)
16. [Team](#-team)
17. [License](#-license)

---

## 🛰️ What is SatQuery AI?

**SatQuery AI** is an **interactive vision-language assistant** for analysing **single and paired remote-sensing images** through **natural-language queries**. Built for the **Indian Space Research Organisation (ISRO) — Space Applications Centre (SAC)** as part of **Smart India Hackathon 2026**, it makes satellite imagery analysis accessible to **non-expert users** without requiring GIS workflows, model selection, or specialised remote-sensing knowledge.

Instead of building yet another isolated application for a single task (land-cover classification, object detection, VQA, or change detection), SatQuery AI uses an **agentic, query-driven framework** that automatically:

- **Interprets** the user's natural-language query
- **Validates** input imagery (modality, format, metadata, compatibility)
- **Selects** the right remote-sensing specialist models from a tool registry
- **Executes** the appropriate workflow and combines outputs
- **Returns** evidence-grounded textual and visual results with confidence scores
- **Provides** a complete auditable execution trace

> 💡 **"SatQuery" = Satellite + Query**  
> Ask a question in plain English → Get satellite-grounded answers with visual evidence.

**Organisation:** Indian Space Research Organisation (ISRO) — Space Applications Centre (SAC)  
**Problem Statement:** SatQuery AI — Interactive Vision-Language Assistant for Multimodal Remote Sensing Image Analysis through Text Queries

---

## 🚨 The Problem We Solve

### Existing Remote-Sensing AI Limitations:

```
┌──────────────────────────────────────────────────────────────┐
│  EXISTING SOLUTIONS = ISOLATED, SINGLE-TASK APPLICATIONS     │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ❌ Land-cover classification only                           │
│  ❌ Object detection only                                    │
│  ❌ Visual Question Answering only                           │
│  ❌ Change detection only                                    │
│                                                              │
│  Each requires:                                              │
│  • Understanding satellite-data characteristics              │
│  • Knowledge of GIS workflows                                │
│  • Manual model selection                                    │
│  • Task-specific parameter tuning                            │
│                                                              │
│  ➡️  Non-expert users CANNOT get answers from imagery        │
│      through simple natural-language queries                 │
└──────────────────────────────────────────────────────────────┘
```

### **Real-World Challenges:**

| Challenge | Impact |
|-----------|--------|
| **No unified interface** | Users must learn multiple tools for different tasks |
| **Single-modal limitation** | Optical images fail at night/cloudy conditions; SAR lacks spectral info |
| **No multitemporal reasoning** | Cannot interpret changes over time automatically |
| **Generic VLMs fail** | General-purpose LLMs/VLMs aren't adapted to remote-sensing domain, sensor characteristics, and terminology |
| **Manual GIS workflows** | Non-experts cannot extract actionable insights |
| **No cross-modal fusion** | Optical + SAR complementary info remains unused |
| **No evidence grounding** | Answers lack visual proof and execution trace |

### **Why Generic LLMs/VLMs Are Not Enough:**

> A general-purpose LLM or VLM cannot perform remote-sensing tasks reliably without **adaptation to remote-sensing imagery, sensor characteristics, and domain-specific terminology**. The solution must include **remote-sensing fine-tuning or domain adaptation** and may employ **multiple specialised models** for different tasks.

---

## 💡 Our Solution

**SatQuery AI** is an **agentic vision-language assistant** that transforms satellite imagery analysis into a **conversational, evidence-grounded experience**:

```
┌──────────────────────────────────────────────────────────────┐
│                  SATQUERY AI WORKFLOW                        │
└──────────────────────────────────────────────────────────────┘

        👤 USER: "Has the built-up area increased between 
                  these two dates, and where?"
                          │
                          ▼
        ┌──────────────────────────────────────┐
        │   🧠 AGENTIC CONTROLLER              │
        │   ────────────────────                │
        │   1. Parse natural-language query    │
        │   2. Validate inputs (modality,      │
        │      format, metadata, pairs)        │
        │   3. Classify requested task         │
        │   4. Select specialist models        │
        │      from tool registry              │
        └──────────────────┬───────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
    ┌────────┐        ┌─────────┐        ┌──────────┐
    │  VQA   │        │ Change  │        │ Optical– │
    │  Model │        │  Detector│       │ SAR Fusion│
    └────┬───┘        └────┬────┘        └─────┬────┘
         │                 │                    │
         └─────────────────┼────────────────────┘
                           ▼
        ┌──────────────────────────────────────┐
        │   OUTPUT INTEGRATION LAYER           │
        │   ────────────────────────            │
        │   • Text answer                      │
        │   • Visual evidence (masks, bboxes) │
        │   • Confidence scores                │
        │   • Execution trace                  │
        │   • Downloadable report              │
        └──────────────────────────────────────┘
                          │
                          ▼
        ✅ Evidence-grounded response with visual proof
```

### **What Makes SatQuery AI Different?**

| Feature | Generic VLM | Single-Task RS Apps | **SatQuery AI** |
|---------|-------------|---------------------|-----------------|
| Remote-sensing adaptation | ❌ | ⚠️ Partial | ✅ **Fine-tuned on BigEarthNet.txt** |
| Single-image VQA | ⚠️ Poor | ⚠️ Task-specific | ✅ **Yes** |
| Captioning / Grounding | ❌ | ⚠️ One or other | ✅ **Both** |
| Bi-temporal change analysis | ❌ | ⚠️ Separate tool | ✅ **Integrated** |
| Cross-modal Optical+SAR | ❌ | ❌ | ✅ **Yes** |
| Agentic orchestration | ❌ | ❌ | ✅ **Full trace** |
| Natural-language interface | ✅ Generic | ❌ | ✅ **Domain-aware** |
| Evidence & confidence | ❌ | ⚠️ Partial | ✅ **Full** |
| Auditable execution | ❌ | ❌ | ✅ **Yes** |

---

## ✨ Key Features

### 🎯 Single-Image Understanding (Mandatory Baseline):
- ❓ **Visual Question Answering (VQA)** — Ask any question about a single optical/multispectral or SAR image
- 📝 **Scene Description / Captioning** — Auto-generate descriptive captions
- 🎯 **Text-Guided Region Grounding** — "Highlight the water body referred to in the query"

### 🎯 Multi-Image Change Analysis (Mandatory):
- 🔄 **Change Description** — Natural-language description of what changed
- ❓ **Change-based VQA** — "Has the built-up area increased, decreased, or remained unchanged?"
- 🗺️ **Spatial Change Maps** — Where reference masks are available
- 📅 **Bi-temporal Reasoning** — Two images of the same area at different times

### 🎯 Cross-Modal Optical–SAR Analysis (Mandatory):
- 🛰️ **Joint Information Extraction** — Combine optical + SAR for richer analysis
- 🌊 **Built-up + Water Detection** — Using complementary modalities
- 🌙 **Day/Night & Cloud-Penetrating Analysis** — SAR advantage over optical
- 🔗 **Co-registered Pair Processing** — Same geographic area, different sensors

### 🎯 Agentic Orchestration (Core Novelty):
- 🧠 **Query Interpretation** — Classify the task from natural language
- ✅ **Input Validation** — Modality, format, metadata, compatibility checks
- 🧰 **Tool Registry** — Predefined registry of specialist models
- ⚙️ **Parameter Configuration** — Only permitted task parameters
- 🔀 **Automatic Workflow Selection** — Sequence models based on query
- 📊 **Output Integration** — Combine textual + spatial outputs
- 📋 **Auditable Execution Trace** — Selected task, models, parameters, outputs

### 🎯 User Interface:
- 📤 **Drag-and-drop Upload** — GeoTIFF, TIFF, PNG, JPEG (benchmark)
- 💬 **Natural-Language Query Box** — Ask anything
- 🖼️ **Visual Evidence Display** — Masks, bounding boxes, change maps
- 📈 **Confidence Scores** — Reliability of each output
- 📄 **Downloadable Reports** — PDF/JSON execution summaries
- 🔍 **Execution Trace Viewer** — Full audit trail

---

## 📸 Demo Screenshots

### 🏠 1. Home / Landing Page
![Home Page](docs/images/home.png)
*Clean interface with upload area, query box, and quick-start examples*

---

### 🔐 2. Login Page
![Login Page](docs/images/login.png)
*Secure authentication for analysts and administrators*

---

### 📤 3. Image Upload & Validation
![Upload Validation](docs/images/upload.png)
*Automatic validation of modality, format, metadata, and pair compatibility*

---

### ❓ 4. Single-Image VQA
![Single Image VQA](docs/images/vqa.png)
*Ask: "Describe the land-cover and major objects visible in this image."*

---

### 📝 5. Image Captioning
![Captioning](docs/images/captioning.png)
*Auto-generated scene description with confidence score*

---

### 🎯 6. Text-Guided Region Grounding
![Grounding](docs/images/grounding.png)
*Query: "Highlight the water body referred to in the query."*

---

### 🔄 7. Bi-temporal Change Analysis
![Change Detection](docs/images/change.png)
*Query: "What changed between these two dates, and where did the change occur?"*

---

### 🛰️ 8. Optical–SAR Cross-Modal Analysis
![Cross Modal](docs/images/cross-modal.png)
*Query: "Use the optical and SAR images together to identify built-up and water-covered regions."*

---

### 🧠 9. Agentic Execution Trace
![Execution Trace](docs/images/trace.png)
*Full audit: selected task, models, parameters, timing, and outputs*

---

### 📊 10. Confidence & Evidence Panel
![Confidence Panel](docs/images/confidence.png)
*Per-component confidence scores with visual evidence overlay*

---

### 📄 11. Downloadable Report
![Report](docs/images/report.png)
*PDF/JSON report with query, models used, evidence, and results*

---

## 🔄 How It Works

### **Agentic Pipeline:**

```
┌─────────────────────────────────────────────────────────────────┐
│              SATQUERY AI — AGENTIC PIPELINE                      │
└─────────────────────────────────────────────────────────────────┘

  📥 INPUT                          💬 QUERY
  ────────                          ────────
  • Single image                    • Natural language
  • Optical–SAR pair                • Free-form
  • Bi-temporal pair                • Domain-aware
         │                               │
         └───────────────┬───────────────┘
                         ▼
        ┌────────────────────────────────────┐
        │  🔍 INPUT VALIDATOR                │
        │  • Modality check                  │
        │  • Format (GeoTIFF/TIFF/PNG/JPEG)  │
        │  • Metadata (CRS, bands, dates)    │
        │  • Pair compatibility (co-register)│
        └────────────────┬───────────────────┘
                         ▼
        ┌────────────────────────────────────┐
        │  🧠 AGENTIC CONTROLLER             │
        │  • Query intent classification     │
        │  • Task routing                    │
        │  • Model selection from registry   │
        │  • Parameter configuration         │
        └────────────────┬───────────────────┘
                         ▼
        ┌────────────────────────────────────┐
        │  🧰 SPECIALIST MODEL REGISTRY      │
        │  ┌──────────┬──────────┬─────────┐ │
        │  │  VQA     │Captioning│Grounding│ │
        │  ├──────────┼──────────┼─────────┤ │
        │  │ Change   │Optical-  │  Other  │ │
        │  │ Detector │SAR Fusion│  Tools  │ │
        │  └──────────┴──────────┴─────────┘ │
        └────────────────┬───────────────────┘
                         ▼
        ┌────────────────────────────────────┐
        │  🔗 OUTPUT INTEGRATION             │
        │  • Text answer                     │
        │  • Visual evidence (masks/boxes)   │
        │  • Confidence scores               │
        │  • Execution trace                 │
        │  • Downloadable report             │
        └────────────────┬───────────────────┘
                         ▼
        ✅ EVIDENCE-GROUNDED RESPONSE
```

### **Task Routing Logic (Example):**

| Query | Input | Detected Task | Models Used |
|-------|-------|---------------|-------------|
| "Describe the land-cover..." | Single image | Captioning | Captioning Model |
| "Highlight the water body..." | Single image | Grounding | Grounding Model |
| "What changed between these dates?" | Bi-temporal pair | Change VQA | Change Detector + VQA |
| "Use optical and SAR together..." | Optical–SAR pair | Cross-modal | Fusion Model |
| "Has built-up area increased?" | Bi-temporal pair | Change VQA | Change Detector + Classifier |

---

## 📋 Supported Inputs & Tasks

### **Input Scope:**

| Type | Description | Use Case |
|------|-------------|----------|
| **Single Image** | One optical/multispectral or SAR image | VQA, captioning, grounding |
| **Cross-Modal Pair** | Co-registered optical + SAR of same area | Joint information extraction |
| **Bi-temporal Pair** | Two images of same area at different times | Change detection, change VQA |
| **Formats** | GeoTIFF, TIFF (primary); PNG, JPEG (benchmarks only) | Standard geospatial formats |

### **Mandatory Functional Scope:**

| # | Task | Status | Description |
|---|------|--------|-------------|
| 1 | **Remote-Sensing Adaptation** | ✅ | Fine-tuned on BigEarthNet.txt |
| 2 | **Single-Image VQA** | ✅ | Mandatory baseline |
| 3 | **Captioning OR Grounding** | ✅ | Both implemented |
| 4 | **Bi-temporal Change Analysis** | ✅ | Change description + change VQA |
| 5 | **Cross-Modal Optical–SAR** | ✅ | Complementary information extraction |
| 6 | **Agentic Orchestration** | ✅ | Auto model selection + execution |
| 7 | **Spatial Change Maps** | ✅ | Where reference masks available |

### **Representative Queries Supported:**

```
✅ "Describe the land-cover and major objects visible in this image."
✅ "Highlight the water body referred to in the query."
✅ "What changed between these two dates, and where did the change occur?"
✅ "Use the optical and SAR images together to identify built-up and water-covered regions."
✅ "Has the built-up area increased, decreased, or remained unchanged?"
```

---

## 🛠️ Tech Stack

### **Backend:**
| Component | Technology |
|-----------|------------|
| Framework | **FastAPI** (Python 3.10+) |
| Agentic Controller | Custom orchestration engine |
| Model Registry | Plugin-based specialist models |
| Database | **PostgreSQL** / SQLite |
| ORM | SQLAlchemy 2.0 |
| Auth | JWT (PyJWT + bcrypt) |
| Task Queue | Celery / BackgroundTasks |
| API Docs | Swagger UI + ReDoc |

### **AI / ML Stack:**
| Component | Technology |
|-----------|------------|
| **VLM Backbone** | Remote-sensing adapted VLM (fine-tuned) |
| **Fine-tuning Dataset** | **BigEarthNet.txt** |
| **VQA Evaluation** | VRSBench, RSVQA |
| **Change-VQA Evaluation** | CDVQA |
| **Image Processing** | Rasterio, GDAL, OpenCV, Pillow |
| **Deep Learning** | PyTorch, Transformers, timm |
| **Grounding** | Text-guided region grounding model |
| **Fusion** | Optical–SAR fusion network |
| **Change Detection** | Bi-temporal change encoder |
| **Embeddings** | CLIP / Remote-sensing CLIP variant |

### **Frontend:**
| Component | Technology |
|-----------|------------|
| Framework | **React 18** |
| Build Tool | **Vite** |
| Styling | **Tailwind CSS** |
| Routing | React Router v6 |
| HTTP Client | Axios |
| Image Viewer | OpenLayers / Leaflet / custom canvas |
| Charts | Chart.js |
| Icons | React Icons |
| Notifications | React Toastify |
| File Upload | react-dropzone |

### **DevOps:**
| Component | Tool |
|-----------|------|
| Version Control | Git + GitHub |
| Code Editor | VS Code |
| Testing | Pytest (backend), Vitest (frontend) |
| Linting | ESLint + Prettier + Black |
| Container | Docker (optional) |

---

## 📥 Installation Guide

### **⚠️ Prerequisites Checklist**

- ✅ **OS:** Windows 10/11, macOS, or Linux
- ✅ **RAM:** Minimum 8 GB (Recommended: **16 GB+** for VLM inference)
- ✅ **Storage:** Minimum 15 GB free (models + datasets)
- ✅ **GPU:** Optional but **recommended** (NVIDIA CUDA for VLM)
- ✅ **Internet:** Stable connection

---

### **Step 1: Install Core Tools**

| Tool | Version | Link |
|------|---------|------|
| **Node.js** | v18.0+ | [nodejs.org](https://nodejs.org/en/download) |
| **Python** | 3.10+ | [python.org](https://www.python.org/downloads/) |
| **Git** | Latest | [git-scm.com](https://git-scm.com/downloads) |
| **GDAL** | 3.6+ | [gdal.org](https://gdal.org/download.html) |
| **CUDA (optional)** | 11.8+ | [developer.nvidia.com](https://developer.nvidia.com/cuda-downloads) |
| **VS Code** | Latest | [code.visualstudio.com](https://code.visualstudio.com/) |

**Verify:**
```bash
node --version      # v18.x.x+
python --version    # 3.10+
git --version       # 2.x.x
gdalinfo --version  # GDAL 3.6+
```

---

### **Step 2: Clone the Repository**

```bash
git clone https://github.com/sihggv/sih-bid-compliance.git
cd sih-bid-compliance
```

---

### **Step 3: Backend Setup**

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate
venv\Scripts\activate           # Windows
# source venv/bin/activate      # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# If manual install:
pip install fastapi uvicorn python-multipart python-jose[cryptography] passlib[bcrypt] python-dotenv pydantic[email] pydantic-settings sqlalchemy alembic PyJWT bcrypt Pillow numpy pandas aiofiles httpx rasterio gdal torch torchvision transformers timm opencv-python

# Create required folders
mkdir uploads logs models cache

# Create .env file
copy .env.example .env    # Windows
cp .env.example .env      # Mac/Linux
```

---

### **Step 4: Frontend Setup**

```bash
cd ../frontend
npm install
copy .env.example .env
```

---

### **Step 5: Download Models & Datasets**

```bash
cd ../backend

# Download fine-tuned VLM (via your HuggingFace repo or provided link)
python scripts/download_models.py

# Optional: Download BigEarthNet.txt sample for testing
python scripts/download_bigearthnet_sample.py

# Optional: Download VRSBench / RSVQA / CDVQA test splits
python scripts/download_benchmarks.py
```

---

### **Step 6: Environment Files**

**`backend/.env`:**
```env
APP_NAME=SatQuery AI
APP_VERSION=1.0.0
DEBUG=True
ENVIRONMENT=development
SECRET_KEY=your-super-secret-key-min-32-characters
HOST=0.0.0.0
PORT=8000
RELOAD=True
DATABASE_URL=sqlite:///./app.db
JWT_SECRET_KEY=your-jwt-secret-min-32-characters
JWT_ALGORITHM=HS256
JWT_EXPIRY_MINUTES=60
UPLOAD_DIR=./uploads
MODEL_DIR=./models
CACHE_DIR=./cache
MAX_FILE_SIZE=524288000
ALLOWED_EXTENSIONS=.tif,.tiff,.png,.jpg,.jpeg
CORS_ORIGINS=http://localhost:5173,http://localhost:3000
DEVICE=cuda
VLM_MODEL=your-finetuned-vlm-name
CHANGE_MODEL=your-change-model
FUSION_MODEL=your-fusion-model
GROUNDING_MODEL=your-grounding-model
LOG_LEVEL=INFO
```

**`frontend/.env`:**
```env
VITE_API_BASE_URL=http://localhost:8000/api
VITE_API_PROXY_URL=http://localhost:8000
VITE_BASE_URL=/
VITE_ENABLE_SOURCEMAP=false
```

---

## 🚀 Running the Project

### **Terminal 1 — Backend:**
```bash
cd backend
venv\Scripts\activate
python -m app.main
```
**Expected:**
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Models loaded: VLM, Change, Fusion, Grounding
INFO:     Application startup complete.
```

### **Terminal 2 — Frontend:**
```bash
cd frontend
npm run dev
```
**Expected:**
```
VITE v4.5.14  ready in 460 ms
➜  Local:   http://localhost:5173/
```

### **Open in Browser:**

| Service | URL |
|---------|-----|
| 🌐 **Frontend** | http://localhost:5173 |
| 🔌 **Backend API** | http://localhost:8000 |
| 📚 **API Docs** | http://localhost:8000/api/docs |
| 📖 **ReDoc** | http://localhost:8000/api/redoc |
| ❤️ **Health** | http://localhost:8000/api/health |

---

## 📁 Project Structure

```
sih-bid-compliance/
│
├── 📂 backend/
│   ├── 📂 app/
│   │   ├── 📄 main.py                     # FastAPI entry point
│   │   │
│   │   ├── 📂 api/                        # API Endpoints
│   │   │   ├── 📄 routes_auth.py
│   │   │   ├── 📄 routes_upload.py
│   │   │   ├── 📄 routes_query.py         # Natural-language queries
│   │   │   ├── 📄 routes_agent.py         # Agentic orchestration
│   │   │   ├── 📄 routes_vqa.py
│   │   │   ├── 📄 routes_captioning.py
│   │   │   ├── 📄 routes_grounding.py
│   │   │   ├── 📄 routes_change.py
│   │   │   ├── 📄 routes_fusion.py
│   │   │   └── 📄 routes_reports.py
│   │   │
│   │   ├── 📂 agent/                      # Agentic Controller
│   │   │   ├── 📄 controller.py           # Main orchestrator
│   │   │   ├── 📄 intent_parser.py        # Query understanding
│   │   │   ├── 📄 input_validator.py      # Image/metadata checks
│   │   │   ├── 📄 task_router.py          # Task classification
│   │   │   ├── 📄 model_registry.py       # Tool registry
│   │   │   ├── 📄 workflow_executor.py    # Execution engine
│   │   │   └── 📄 output_integrator.py    # Evidence fusion
│   │   │
│   │   ├── 📂 models_ai/                  # Specialist Models
│   │   │   ├── 📄 vlm_backbone.py         # Remote-sensing VLM
│   │   │   ├── 📄 vqa_model.py            # Single-image VQA
│   │   │   ├── 📄 captioning_model.py     # Scene description
│   │   │   ├── 📄 grounding_model.py      # Region grounding
│   │   │   ├── 📄 change_detector.py      # Bi-temporal change
│   │   │   ├── 📄 change_vqa.py           # Change-based VQA
│   │   │   ├── 📄 fusion_model.py         # Optical–SAR fusion
│   │   │   └── 📄 classifier.py           # Land-cover classifier
│   │   │
│   │   ├── 📂 preprocessing/              # Image Processing
│   │   │   ├── 📄 geo_reader.py           # GeoTIFF/TIFF reader
│   │   │   ├── 📄 co_registration.py      # Pair alignment
│   │   │   ├── 📄 metadata_checker.py     # CRS/bands validation
│   │   │   ├── 📄 normalization.py
│   │   │   └── 📄 augmentation.py
│   │   │
│   │   ├── 📂 core/                       # Config & Utilities
│   │   │   ├── 📄 config.py
│   │   │   ├── 📄 auth.py
│   │   │   ├── 📄 logging.py
│   │   │   └── 📄 errors.py
│   │   │
│   │   ├── 📂 models/                     # Pydantic Schemas
│   │   │   └── 📄 schemas.py
│   │   │
│   │   └── 📂 database/
│   │       ├── 📄 db.py
│   │       └── 📄 seed.json
│   │
│   ├── 📂 scripts/                        # Setup Scripts
│   │   ├── 📄 download_models.py
│   │   ├── 📄 download_benchmarks.py
│   │   └── 📄 finetune_bigearthnet.py
│   │
│   ├── 📂 models/                         # Model Weights (git-ignored)
│   ├── 📂 uploads/                        # User uploads
│   ├── 📄 requirements.txt
│   └── 📄 .env
│
├── 📂 frontend/
│   ├── 📂 src/
│   │   ├── 📂 assets/
│   │   ├── 📂 components/
│   │   │   ├── 📄 Sidebar.jsx
│   │   │   ├── 📄 ImageUploader.jsx
│   │   │   ├── 📄 QueryBox.jsx
│   │   │   ├── 📄 ExecutionTrace.jsx
│   │   │   ├── 📄 ConfidencePanel.jsx
│   │   │   ├── 📄 EvidenceViewer.jsx
│   │   │   ├── 📄 ChangeMapViewer.jsx
│   │   │   ├── 📄 CrossModalPanel.jsx
│   │   │   └── 📄 ReportDownload.jsx
│   │   │
│   │   ├── 📂 pages/
│   │   │   ├── 📄 Home.jsx
│   │   │   ├── 📄 Login.jsx
│   │   │   ├── 📄 QueryWorkspace.jsx
│   │   │   ├── 📄 History.jsx
│   │   │   ├── 📄 Reports.jsx
│   │   │   └── 📄 About.jsx
│   │   │
│   │   ├── 📂 services/
│   │   │   └── 📄 api.js
│   │   │
│   │   ├── 📄 App.jsx
│   │   └── 📄 main.jsx
│   │
│   ├── 📄 package.json
│   ├── 📄 vite.config.js
│   └── 📄 .env
│
├── 📂 data/                               # Datasets (git-ignored)
│   ├── bigearthnet_sample/
│   ├── vrsbench/
│   ├── rsvqa/
│   └── cdvqa/
│
├── 📂 docs/
│   └── 📂 images/
│
├── 📄 .gitignore
├── 📄 README.md
└── 📄 LICENSE
```

---

## 📚 API Documentation

Once backend is running: **http://localhost:8000/api/docs**

### **Key Endpoints:**

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/auth/login` | User login |
| POST | `/api/upload/image` | Upload single / pair image |
| GET | `/api/upload/validate/{id}` | Validate image metadata |
| POST | `/api/query` | **Unified query endpoint** (agentic) |
| POST | `/api/vqa` | Single-image VQA |
| POST | `/api/caption` | Image captioning |
| POST | `/api/ground` | Text-guided grounding |
| POST | `/api/change` | Bi-temporal change analysis |
| POST | `/api/change-vqa` | Change-based VQA |
| POST | `/api/fusion` | Optical–SAR fusion analysis |
| GET | `/api/trace/{query_id}` | Execution trace |
| GET | `/api/report/{query_id}` | Downloadable report |
| GET | `/api/models` | List available models in registry |
| GET | `/api/health` | Health check |

---

## 🔑 Demo Credentials

| Role | Email | Password |
|------|-------|----------|
| 👤 **Analyst** | `analyst@example.com` | `Analyst@123` |
| 🔧 **Admin** | `admin@example.com` | `Admin@123` |

---

## 📊 Datasets & Benchmarks

| Dataset | Purpose |
|---------|---------|
| **BigEarthNet.txt** | Primary dataset for **fine-tuning** image–text representations |
| **VRSBench** | Evaluation: captioning, grounding |
| **RSVQA** | Evaluation: single-image VQA |
| **CDVQA** | Evaluation: multitemporal change-based VQA |
| **ISRO/SAC Evaluation Set** | Pre-georeferenced, co-registered **Cartosat-2S optical + RISAT SAR** pairs (annotations withheld) |

---

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| `GDAL not found` | Install GDAL matching your Python version |
| `CUDA out of memory` | Reduce batch size / use CPU fallback |
| `Model weights not found` | Run `python scripts/download_models.py` |
| `Unsupported image format` | Use GeoTIFF/TIFF; PNG/JPEG only for benchmarks |
| `Pair misaligned` | Ensure co-registered inputs; use the co-registration utility |
| `Query not understood` | Rephrase; agentic controller logs the reason in trace |
| `Slow inference` | Enable GPU; use half-precision; cache embeddings |
| `ModuleNotFoundError: rasterio` | `pip install rasterio` |
| `Port 8000 in use` | `python -m app.main --port 8001` |
| `Port 5173 in use` | `npm run dev -- --port 5174` |

### **Quick Fixes:**
```bash
# Clear caches
rmdir /s cache
rmdir /s .vite

# Re-download models
python scripts/download_models.py --force

# Reset database
del app.db
python -c "from app.database.db import init_db_sync; init_db_sync()"
```

---

## 👥 Team

**Team Name:** SatQuery AI

| Role | Name | GitHub |
|------|------|--------|
| Team Lead & ML Engineer | [Your Name] | [@yourhandle](https://github.com/yourhandle) |
| Backend / Agentic Controller | [Friend's Name] | [@friendhandle](https://github.com/friendhandle) |
| Frontend Developer | [Member Name] | [@memberhandle](https://github.com/memberhandle) |
| Research & Data | [Member Name] | [@memberhandle](https://github.com/memberhandle) |

**Organisation:** Indian Space Research Organisation (ISRO) — Space Applications Centre (SAC)  
**Event:** Smart India Hackathon 2026  
**Problem Statement:** SatQuery AI — An Interactive Vision-Language Assistant for Multimodal Remote Sensing Image Analysis through Text Queries

**Mentors:**
- Aminur Hossain — aminur@sac.isro.gov.in
- Sanjay K Singh — sks@sac.isro.gov.in
- S Devakanth Naidu — devakanth@sac.isro.gov.in

---

## 🙏 Acknowledgments

- **ISRO — Space Applications Centre (SAC)** — For the problem statement and guidance
- **Ministry of Electronics & IT (MeitY)** — For Smart India Hackathon
- **BigEarthNet, VRSBench, RSVQA, CDVQA** teams — For public datasets
- **Open Source Community** — FastAPI, PyTorch, Hugging Face, Rasterio, GDAL, React
- **Our Mentors** — For constant support

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2026 SatQuery AI Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 📞 Contact & Support

- 🐛 **Issues:** [GitHub Issues](https://github.com/sihggv/sih-bid-compliance/issues)
- 💬 **Discussions:** [GitHub Discussions](https://github.com/sihggv/sih-bid-compliance/discussions)
- 📧 **Email:** team@satquery.ai

---

<div align="center">

## 🌟 Star Us!

If you find this project helpful, please give it a ⭐ on GitHub!

[![GitHub stars](https://img.shields.io/github/stars/sihggv/sih-bid-compliance?style=social)](https://github.com/sihggv/sih-bid-compliance)

---

**Made with ❤️ in India 🇮🇳 for Smart India Hackathon 2026**

**Problem by ISRO — Space Applications Centre (SAC)**

**[⬆ Back to Top](#-satquery-ai)**

</div>
