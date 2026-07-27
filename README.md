# MyGPT: Custom GPT Multimodal Platform Built From Scratch

A production-grade, modular Python & PyTorch implementation of a GPT-style Large Language Model built completely from scratch without external Hugging Face model dependencies, featuring a FastAPI backend, a Next.js web application, and multi-container Docker deployment.

---

## 🌟 Key Capabilities & Features

- **Built Completely From Scratch**:
  - **Byte Pair Encoding (BPE) Tokenizer**: Custom byte-level BPE tokenizer operating on 256 base UTF-8 byte tokens + special tokens (`<pad>`, `<unk>`, `<bos>`, `<eos>`).
  - **PyTorch Dataset Pipeline**: Binary sequence chunking (`train.bin`, `val.bin`), disk memory-mapping (`np.memmap`), and sliding-window target shift ($y = x_{\text{shift\_right}}$).
  - **Token & Positional Embeddings**: Learned positional lookups, fixed sinusoidal positional matrices, and input embedding layers.
  - **Scaled Dot-Product Self-Attention**: Causal masking with upper-triangular boolean masks ($M \in \{0, -\infty\}$).
  - **Multi-Head Causal Attention**: Batched parallel head projections ($Q, K, V \in \mathbb{R}^{B \times n_{\text{head}} \times T \times d_{\text{head}}}$) and output projection $W^O$.
  - **Transformer Decoder Stack**: Pre-LayerNorm residual architecture ($x \leftarrow x + \text{Attn}(\text{LN}_1(x))$ and $x \leftarrow x + \text{MLP}(\text{LN}_2(x))$).
  - **Complete GPT Model Architecture**: Configurable depth, head count, and dimension ($d_{\text{model}}$), weight tying between token embedding and LM head weights (`lm_head.weight = embedding.weight`), and cross-entropy loss computation.

- **Production Training Pipeline**:
  - **Automatic Mixed Precision (AMP)**: `torch.amp.autocast` & `torch.amp.GradScaler` for memory efficiency.
  - **Selective Weight Decay**: AdamW parameter group decay splitting (2D linear/embedding weights decayed, 1D biases/LayerNorm un-decayed).
  - **Cosine LR Scheduler**: Linear warmup followed by cosine annealing decay to `min_lr`.
  - **Checkpoint Management & Early Stopping**: Atomic save/resume for model, optimizer, scaler, epoch, and step state dicts.

- **Advanced Inference Engine**:
  - **Decoding Algorithms**: Temperature scaling, Top-$k$ truncation, Top-$p$ (Nucleus) cumulative probability sampling, deterministic Greedy decoding, and Beam Search.
  - **Real-Time Streaming Generation**: Python generator yielding individual token strings/IDs in real time.

- **Evaluation & Benchmarking Framework**:
  - Perplexity metric ($\text{PPL} = e^{\text{loss}}$), validation loss evaluation, and inference throughput benchmarking (`tokens/sec`, `ms/token`).

- **Enterprise Platform Features**:
  - **User Accounts & Authentication**: JWT token authentication and bcrypt password hashing.
  - **Database Persistence**: SQLite database storage via SQLAlchemy (`User`, `ChatSessionDB`, `ChatMessageDB`, `UploadedFileDB`).
  - **Document RAG Ingestion**: File parser extracting text context from PDF, DOCX, and TXT files.
  - **Multimodal Vision Image Understanding**: `VisionEncoder` patch convolutional feature projector mapping image inputs into LLM sequence embeddings.
  - **Voice STT & Text-To-Speech (TTS)**: Web Speech API voice microphone input and audio synthesis playback.
  - **Multiple AI Personalities (Personas)**: Persona configurations (*General Assistant*, *Senior Code Architect*, *Creative Storyteller*, *Academic Researcher*, *Math Tutor*).
  - **Plugin / Function Calling Framework**: Interactive tool execution for Web Search, Python REPL Code Interpreter, and Calculator tools.

- **Full-Stack Application & Deployment Infrastructure**:
  - **FastAPI Backend Server**: Exposing `/api/v1/generate`, `/api/v1/chat`, `/api/v1/upload`, `/api/v1/vision`, `/api/v1/health`, `/api/v1/info`, and `/api/v1/auth`.
  - **Next.js Web Studio**: Responsive dark-mode frontend interface built with App Router, TypeScript, and Tailwind CSS.
  - **Docker & Compose**: Multi-stage Dockerfiles (`Dockerfile.api`, `Dockerfile.web`), `docker-compose.yml`, and one-click deployment scripts (`deploy.sh`, `deploy.ps1`).

---

## 📂 Project Structure

```text
d:\Projects\my-gpt 2\
├── api/                  # FastAPI Backend API Server, Auth, Database Models, Schemas
│   ├── app.py            # Main FastAPI server entry point
│   ├── auth.py           # Authentication, JWT, and Password Hashing
│   ├── database.py       # SQLAlchemy SQLite Engine & Session setup
│   ├── models_db.py      # Database ORM models (User, ChatSession, ChatMessage, UploadedFile)
│   └── schemas.py        # Pydantic request/response schemas
├── config/               # Model & Training Configuration
│   ├── model_config.py   # GPTConfig dataclass & model presets (gpt_micro, gpt2_small)
│   ├── train_config.py   # TrainConfig dataclass & training hyperparameters
│   └── personas.py       # AI Personalities (General, Code Architect, Writer, Researcher)
├── dataset/              # PyTorch Dataset Pipeline
│   ├── dataloader.py     # DataLoader construction utilities
│   ├── dataset.py        # GPTDataset (input x and right-shifted target y)
│   └── preprocessor.py   # Memory-friendly tokenization & memmap creation
├── docs/                 # Production & Technical Documentation
│   ├── api_guide.md      # API Endpoint Specifications & cURL Guide
│   ├── deployment.md     # Production Docker, Nginx, Systemd, and CUDA Guide
│   └── evaluation_report.md # Benchmark Metrics Report
├── models/               # PyTorch Model Architecture
│   ├── attention.py      # Scaled Dot-Product & Multi-Head Causal Attention
│   ├── embedding.py      # Token, Positional, Sinusoidal, and GPTEmbedding
│   ├── gpt.py            # Complete GPT Model Class & Autoregressive generate()
│   ├── inference.py      # GPTGenerator (Temperature, Top-k, Top-p, Beam Search, Streaming)
│   ├── layers.py         # FeedForward (GELU), LayerNorm, and TransformerBlock
│   └── vision.py         # Multimodal Vision Encoder & MultimodalGPT
├── scripts/              # Automation & Evaluation Scripts
│   ├── deploy.ps1        # PowerShell Deployment Script (Windows)
│   ├── deploy.sh         # POSIX Bash Deployment Script (Linux/macOS)
│   └── evaluate.py       # Model Perplexity & Throughput Evaluation Script
├── tests/                # Comprehensive Test Suite (67 Pytest Unit Tests)
│   ├── test_api.py       # API Endpoint Tests
│   ├── test_dataset.py   # Dataset & DataLoader Tests
│   ├── test_enterprise.py# Auth, DB, File RAG, Vision, Tools & Persona Tests
│   ├── test_inference.py # Sampling, Beam Search & Streaming Tests
│   ├── test_metrics.py   # Perplexity & Speed Benchmark Tests
│   ├── test_model.py     # Embeddings, Attention, Blocks & GPT Model Tests
│   ├── test_tokenizer.py # BPE Tokenizer Tests
│   └── test_trainer.py   # Optimizer Decay Splitting, Scheduler & Checkpoint Tests
├── tokenizer/            # BPE Tokenizer
│   ├── base_tokenizer.py # Abstract Base Class for Tokenizers
│   └── bpe_tokenizer.py  # Byte-Level Byte Pair Encoding Tokenizer
├── training/             # Training Pipeline
│   ├── checkpoint.py     # CheckpointManager (save, load, latest/best tracking)
│   ├── optimizer.py      # Selective AdamW Weight Decay & Cosine Warmup Scheduler
│   └── trainer.py        # Trainer Class with AMP, Gradient Clipping & Early Stopping
├── utils/                # System Utilities & Tool Framework
│   ├── file_parser.py    # Document Parser (PDF, DOCX, TXT)
│   ├── helpers.py        # Random seed setting and CUDA/CPU device detection
│   ├── logger.py         # Structured Logging Utility
│   ├── metrics.py        # Perplexity & Speed Throughput Metrics
│   └── tools.py          # Tool Execution Framework (Web Search, Python REPL, Calculator)
├── web/                  # Next.js Web Studio Application
│   ├── src/app/          # Next.js App Router Pages & Styles
│   ├── src/components/   # Chat, Header, Sidebar, Input, Auth & Config Modal Components
│   ├── src/lib/          # Client API & Storage Utilities
│   └── src/types/        # TypeScript Type Definitions
├── Dockerfile.api        # Multi-stage Dockerfile for FastAPI Python Backend
├── Dockerfile.web        # Multi-stage Dockerfile for Next.js Web Frontend
├── docker-compose.yml    # Multi-container Compose Orchestration
├── requirements.txt      # Python Dependencies
└── README.md             # Project Documentation
```

---

## 🚀 Quickstart & Getting Started

### 1. Local Python Environment

```powershell
# Create & activate virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run pytest unit test suite (67 tests)
python -m pytest tests/ -v
```

### 2. Launch FastAPI Server Locally

```powershell
python -m uvicorn api.app:app --host 127.0.0.1 --port 8000 --reload
```
- **Interactive Swagger UI**: `http://127.0.0.1:8000/docs`

### 3. Launch Next.js Web Studio Locally

```powershell
npm --prefix web run dev
```
- **Web App Interface**: `http://localhost:3000`

---

## 🐳 Docker Deployment

Deploy backend and frontend using Docker Compose:

```bash
# POSIX Bash (Linux/macOS)
chmod +x scripts/deploy.sh
./scripts/deploy.sh

# PowerShell (Windows)
.\scripts\deploy.ps1
```

For detailed production configuration, Nginx SSL proxy setup, and CUDA acceleration, refer to the [Deployment Guide](docs/deployment.md).

---

## 🧪 Testing Summary

Executed full test suite verifying all modules:

```powershell
python -m pytest tests/ -v
```
- **Status**: 67 passed in 32.80s
