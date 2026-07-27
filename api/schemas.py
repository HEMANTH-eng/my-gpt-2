from typing import List, Optional
from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    """Request schema for text generation."""

    prompt: str = Field(..., description="Initial prompt string for text generation.")
    max_new_tokens: int = Field(default=50, ge=1, le=1024, description="Number of tokens to generate.")
    temperature: float = Field(default=1.0, ge=0.0, le=2.0, description="Sampling temperature.")
    top_k: Optional[int] = Field(default=None, ge=1, description="Top-k sampling threshold.")
    top_p: Optional[float] = Field(default=None, gt=0.0, le=1.0, description="Top-p nucleus sampling threshold.")
    greedy: bool = Field(default=False, description="If True, uses deterministic greedy decoding (argmax).")
    num_beams: int = Field(default=1, ge=1, le=8, description="Number of beams for beam search.")
    persona_id: Optional[str] = Field(default="default", description="AI personality ID.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "prompt": "Building a custom GPT model from scratch in PyTorch is",
                "max_new_tokens": 30,
                "temperature": 0.8,
                "top_k": 40,
                "top_p": 0.95,
                "persona_id": "code_architect",
            }
        }
    }


class GenerateResponse(BaseModel):
    """Response schema for text generation."""

    prompt: str
    generated_text: str
    tokens_generated: int
    tool_calls: Optional[List[dict]] = None


class ChatMessage(BaseModel):
    """Single chat message schema."""

    role: str = Field(..., description="Role of the message sender ('user', 'assistant', 'system').")
    content: str = Field(..., description="Message text content.")


class ChatRequest(BaseModel):
    """Request schema for chat completions."""

    messages: List[ChatMessage] = Field(..., min_length=1, description="List of conversation messages.")
    max_new_tokens: int = Field(default=50, ge=1, le=1024, description="Number of response tokens to generate.")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature.")
    top_k: Optional[int] = Field(default=None, ge=1, description="Top-k sampling threshold.")
    top_p: Optional[float] = Field(default=None, gt=0.0, le=1.0, description="Top-p nucleus sampling threshold.")
    persona_id: Optional[str] = Field(default="default", description="AI personality ID.")
    file_id: Optional[str] = Field(default=None, description="Optional uploaded file context ID.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "messages": [
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": "Explain self-attention briefly."},
                ],
                "max_new_tokens": 50,
                "temperature": 0.7,
                "persona_id": "default",
            }
        }
    }


class ChatResponse(BaseModel):
    """Response schema for chat completions."""

    message: ChatMessage
    tool_calls: Optional[List[dict]] = None


class PersonaResponse(BaseModel):
    """Persona schema."""

    id: str
    name: str
    icon: str
    description: str


class UploadedFileResponse(BaseModel):
    """Uploaded file metadata response schema."""

    id: str
    filename: str
    file_type: str
    file_size: int
    text_preview: str



class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str
    model_loaded: bool
    device: str


class ModelInfoResponse(BaseModel):
    """Model metadata response schema."""

    model_name: str
    num_parameters: int
    vocab_size: int
    d_model: int
    n_layer: int
    n_head: int
    max_seq_len: int
