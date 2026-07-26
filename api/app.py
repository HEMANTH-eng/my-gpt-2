from contextlib import asynccontextmanager
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import torch

from api.schemas import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    ModelInfoResponse,
)
from config.model_config import GPTConfig
from models.gpt import GPT
from models.inference import GPTGenerator
from tokenizer.bpe_tokenizer import BPETokenizer
from utils.logger import get_logger

logger = get_logger("api_server")

# Global model state holders
generator_instance: Optional[GPTGenerator] = None
model_instance: Optional[GPT] = None
tokenizer_instance: Optional[BPETokenizer] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI Lifespan context manager handling model and tokenizer startup/shutdown."""
    global generator_instance, model_instance, tokenizer_instance
    logger.info("Initializing MyGPT model and tokenizer for API server...")

    # Train a lightweight base tokenizer for API demonstration
    base_text = "Building a custom GPT large language model completely from scratch using Python and PyTorch!"
    tokenizer_instance = BPETokenizer(vocab_size=300)
    tokenizer_instance.train(base_text)

    # Initialize GPT model
    config = GPTConfig.gpt_micro(vocab_size=len(tokenizer_instance.vocab))
    model_instance = GPT(config)

    # Device selection
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
    tokenizer_instance = None


app = FastAPI(
    title="MyGPT API Server",
    version="1.0.0",
    description="Production REST API server exposing custom PyTorch GPT model for text generation and chat completions.",
    lifespan=lifespan,
)

# Enable Cross-Origin Resource Sharing (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_generator() -> GPTGenerator:
    if generator_instance is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model generator is not initialized.",
        )
    return generator_instance


@app.get("/api/v1/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """Returns API health status, device info, and model initialization state."""
    gen = _get_generator()
    return HealthResponse(
        status="ok",
        model_loaded=True,
        device=gen.device,
    )


@app.get("/api/v1/info", response_model=ModelInfoResponse, tags=["Model Info"])
async def get_model_info() -> ModelInfoResponse:
    """Returns model metadata, parameter counts, and hyperparameter specifications."""
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


@app.post("/api/v1/generate", response_model=GenerateResponse, tags=["Inference"])
async def generate_text(request: GenerateRequest) -> GenerateResponse:
    """Generates text continuation for a given prompt string."""
    gen = _get_generator()
    try:
        result_text = gen.generate(
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            greedy=request.greedy,
            num_beams=request.num_beams,
        )

        if not isinstance(result_text, str):
            result_text = str(result_text)

        # Calculate new tokens generated
        prompt_len = len(tokenizer_instance.encode(request.prompt)) if tokenizer_instance else 0
        total_len = len(tokenizer_instance.encode(result_text)) if tokenizer_instance else 0
        tokens_gen = max(0, total_len - prompt_len)

        return GenerateResponse(
            prompt=request.prompt,
            generated_text=result_text,
            tokens_generated=tokens_gen,
        )
    except Exception as e:
        logger.error(f"Error during generation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Generation failed: {str(e)}",
        )


def _format_chat_prompt(messages: List[ChatMessage]) -> str:
    """Formats a list of chat messages into a single text prompt."""
    formatted_parts = []
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
async def chat_completion(request: ChatRequest) -> ChatResponse:
    """Generates an assistant response for multi-turn chat conversations."""
    gen = _get_generator()
    try:
        formatted_prompt = _format_chat_prompt(request.messages)

        full_response = gen.generate(
            prompt=formatted_prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
        )

        if not isinstance(full_response, str):
            full_response = str(full_response)

        # Extract assistant response portion after prompt
        if formatted_prompt in full_response:
            assistant_reply = full_response.split(formatted_prompt)[-1].strip()
        else:
            assistant_reply = full_response[len(formatted_prompt):].strip()

        if not assistant_reply:
            assistant_reply = full_response

        return ChatResponse(
            message=ChatMessage(role="assistant", content=assistant_reply)
        )
    except Exception as e:
        logger.error(f"Error during chat completion: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat completion failed: {str(e)}",
        )
