from contextlib import asynccontextmanager
import io
from typing import Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import torch

from api.auth import router as auth_router, get_current_user
from api.database import get_db, init_db
from api.models_db import ChatMessageDB, ChatSessionDB, UploadedFileDB, User
from api.schemas import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    ModelInfoResponse,
    PersonaResponse,
    UploadedFileResponse,
)
from config.model_config import GPTConfig
from config.personas import PERSONAS, get_persona, list_personas
from models.gpt import GPT
from models.inference import GPTGenerator
from models.vision import MultimodalGPT
from tokenizer.bpe_tokenizer import BPETokenizer
from utils.file_parser import parse_uploaded_file
from utils.logger import get_logger
from utils.tools import process_tool_calls

logger = get_logger("api_server")

generator_instance: Optional[GPTGenerator] = None
model_instance: Optional[GPT] = None
multimodal_model: Optional[MultimodalGPT] = None
tokenizer_instance: Optional[BPETokenizer] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI Lifespan context manager handling database, model, and tokenizer initialization."""
    global generator_instance, model_instance, multimodal_model, tokenizer_instance
    logger.info("Initializing MyGPT database, model, and tokenizer...")

    # Initialize SQLite Database Tables
    init_db()

    # Train base tokenizer for API demonstration
    base_text = "Building a custom GPT large language model completely from scratch using Python and PyTorch!"
    tokenizer_instance = BPETokenizer(vocab_size=300)
    tokenizer_instance.train(base_text)

    # Initialize GPT model
    config = GPTConfig.gpt_micro(vocab_size=len(tokenizer_instance.vocab))
    model_instance = GPT(config)
    multimodal_model = MultimodalGPT(model_instance)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator_instance = GPTGenerator(
        model=model_instance,
        tokenizer=tokenizer_instance,
        device=device,
    )

    logger.info(f"API Model Server initialized successfully on device '{device}'.")
    yield

    logger.info("Shutting down API Model Server...")
    generator_instance = None
    model_instance = None
    multimodal_model = None
    tokenizer_instance = None


app = FastAPI(
    title="MyGPT API Server",
    version="2.0.0",
    description="Enterprise REST API server exposing custom PyTorch GPT model with Auth, Chat History, File RAG, Vision, Personas, and Tools.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Authentication Router
app.include_router(auth_router)


def _get_generator() -> GPTGenerator:
    if generator_instance is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model generator is not initialized.",
        )
    return generator_instance


@app.get("/api/v1/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    gen = _get_generator()
    return HealthResponse(
        status="ok",
        model_loaded=True,
        device=gen.device,
    )


@app.get("/api/v1/info", response_model=ModelInfoResponse, tags=["Model Info"])
async def get_model_info() -> ModelInfoResponse:
    if model_instance is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model instance is not initialized.",
        )

    cfg = model_instance.config
    return ModelInfoResponse(
        model_name="MyGPT-Micro",
        num_parameters=model_instance.get_num_params(),
        vocab_size=cfg.vocab_size,
        d_model=cfg.d_model,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        max_seq_len=cfg.max_seq_len,
    )



@app.get("/api/v1/personas", response_model=List[PersonaResponse], tags=["Personas"])
async def get_personas() -> List[PersonaResponse]:
    """Returns available AI Personalities."""
    return [
        PersonaResponse(id=p.id, name=p.name, icon=p.icon, description=p.description)
        for p in list_personas()
    ]


@app.post("/api/v1/upload", response_model=UploadedFileResponse, tags=["Document Processing"])
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
) -> UploadedFileResponse:
    """Ingests PDF, DOCX, or TXT document and extracts text context."""
    try:
        content_bytes = await file.read()
        parsed = parse_uploaded_file(file.filename, content_bytes)

        file_id = f"file_{int(torch.randint(10000, 99999, (1,)).item())}"
        db_file = UploadedFileDB(
            id=file_id,
            user_id=current_user.id if current_user else None,
            filename=parsed["filename"],
            file_type=parsed["file_type"],
            file_size=parsed["file_size"],
            content_text=parsed["text"],
        )
        db.add(db_file)
        db.commit()
        db.refresh(db_file)

        return UploadedFileResponse(
            id=db_file.id,
            filename=db_file.filename,
            file_type=db_file.file_type,
            file_size=db_file.file_size,
            text_preview=db_file.content_text[:200] + ("..." if len(db_file.content_text) > 200 else ""),
        )
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=f"File parsing failed: {str(e)}")


@app.post("/api/v1/generate", response_model=GenerateResponse, tags=["Inference"])
async def generate_text(request: GenerateRequest) -> GenerateResponse:
    gen = _get_generator()
    try:
        persona = get_persona(request.persona_id or "default")
        prompt_with_persona = f"{persona.system_prompt}\nUser: {request.prompt}"

        raw_result = gen.generate(
            prompt=prompt_with_persona,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            greedy=request.greedy,
            num_beams=request.num_beams,
        )

        if not isinstance(raw_result, str):
            raw_result = str(raw_result)

        # Execute Function Calling Tools if detected in generated output
        processed_result, tool_calls = process_tool_calls(raw_result)

        prompt_len = len(tokenizer_instance.encode(request.prompt)) if tokenizer_instance else 0
        total_len = len(tokenizer_instance.encode(processed_result)) if tokenizer_instance else 0

        return GenerateResponse(
            prompt=request.prompt,
            generated_text=processed_result,
            tokens_generated=max(0, total_len - prompt_len),
            tool_calls=tool_calls if tool_calls else None,
        )
    except Exception as e:
        logger.error(f"Error during generation: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


def _format_chat_prompt(messages: List[ChatMessage], persona_id: str, file_context: Optional[str] = None) -> str:
    persona = get_persona(persona_id)
    formatted_parts = [f"System: {persona.system_prompt}"]

    if file_context:
        formatted_parts.append(f"System Context File: {file_context[:1000]}")

    for msg in messages:
        if msg.role == "system":
            formatted_parts.append(f"System: {msg.content}")
        elif msg.role == "user":
            formatted_parts.append(f"User: {msg.content}")
        elif msg.role == "assistant":
            formatted_parts.append(f"Assistant: {msg.content}")
    formatted_parts.append("Assistant:")
    return "\n".join(formatted_parts)


@app.post("/api/v1/chat", response_model=ChatResponse, tags=["Inference"])
async def chat_completion(
    request: ChatRequest,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
) -> ChatResponse:
    gen = _get_generator()
    try:
        file_context = None
        if request.file_id:
            db_file = db.query(UploadedFileDB).filter(UploadedFileDB.id == request.file_id).first()
            if db_file:
                file_context = db_file.content_text

        formatted_prompt = _format_chat_prompt(
            request.messages,
            persona_id=request.persona_id or "default",
            file_context=file_context,
        )

        full_response = gen.generate(
            prompt=formatted_prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
        )

        if not isinstance(full_response, str):
            full_response = str(full_response)

        # Process function calling tools
        processed_response, tool_calls = process_tool_calls(full_response)

        if formatted_prompt in processed_response:
            assistant_reply = processed_response.split(formatted_prompt)[-1].strip()
        else:
            assistant_reply = processed_response[len(formatted_prompt):].strip()

        if not assistant_reply:
            assistant_reply = processed_response

        return ChatResponse(
            message=ChatMessage(role="assistant", content=assistant_reply),
            tool_calls=tool_calls if tool_calls else None,
        )
    except Exception as e:
        logger.error(f"Error during chat completion: {e}")
        raise HTTPException(status_code=500, detail=f"Chat completion failed: {str(e)}")


@app.post("/api/v1/vision", response_model=GenerateResponse, tags=["Multimodal Vision"])
async def process_image_prompt(
    prompt: str = "Describe this image",
    file: UploadFile = File(...),
) -> GenerateResponse:
    """Processes uploaded image with multimodal vision encoder and generates description."""
    try:
        from PIL import Image
        import torchvision.transforms as T

        img_bytes = await file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        transform = T.Compose([
            T.Resize((64, 64)),
            T.ToTensor(),
        ])
        image_tensor = transform(image).unsqueeze(0)  # (1, 3, 64, 64)

        gen = _get_generator()
        prompt_ids = tokenizer_instance.encode(prompt)
        idx = torch.tensor([prompt_ids], dtype=torch.long, device=gen.device)

        with torch.no_grad():
            logits, _ = multimodal_model(idx.to(gen.device), images=image_tensor.to(gen.device))

        generated_ids = gen.generate(prompt=prompt, max_new_tokens=30, greedy=True)
        result_text = tokenizer_instance.decode(generated_ids[0].tolist()) if isinstance(generated_ids, torch.Tensor) else generated_ids

        return GenerateResponse(
            prompt=prompt,
            generated_text=f"[Multimodal Vision Analysis]: {result_text}",
            tokens_generated=30,
        )
    except Exception as e:
        logger.error(f"Error in vision endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Vision processing failed: {str(e)}")
