import io
import numpy as np
import pytest
import torch

from api.auth import create_access_token, get_password_hash, verify_password
from api.database import Base, SessionLocal, engine
from api.models_db import ChatMessageDB, ChatSessionDB, UploadedFileDB, User
from config.model_config import GPTConfig
from config.personas import PERSONAS, get_persona, list_personas
from models.gpt import GPT
from models.vision import MultimodalGPT, VisionEncoder
from utils.file_parser import parse_uploaded_file
from utils.tools import calculator, process_tool_calls, python_repl, web_search


@pytest.fixture(autouse=True)
def setup_db():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


def test_auth_password_hashing_and_jwt():
    raw_password = "SecretPassword123"
    hashed = get_password_hash(raw_password)

    assert verify_password(raw_password, hashed)
    assert not verify_password("WrongPassword", hashed)

    token = create_access_token({"sub": "testuser"})
    assert isinstance(token, str)
    assert len(token) > 20


def test_database_user_and_session_persistence():
    db = SessionLocal()
    user = User(username="alice", email="alice@example.com", hashed_password="hashed_pwd")
    db.add(user)
    db.commit()
    db.refresh(user)

    session_db = ChatSessionDB(id="sess_101", user_id=user.id, title="Test Chat Session")
    db.add(session_db)
    db.commit()

    retrieved = db.query(User).filter(User.username == "alice").first()
    assert retrieved is not None
    assert retrieved.email == "alice@example.com"
    assert len(retrieved.sessions) == 1
    assert retrieved.sessions[0].id == "sess_101"
    db.close()


def test_file_parser_txt_pdf_docx():
    # Plain text parsing
    txt_bytes = "Building a custom GPT model completely from scratch!".encode("utf-8")
    parsed_txt = parse_uploaded_file("document.txt", txt_bytes)
    assert parsed_txt["file_type"] == "txt"
    assert "custom GPT" in parsed_txt["text"]

    # PDF fallback parsing
    pdf_bytes = b"%PDF-1.4 (Hello PDF Document) Tj"
    parsed_pdf = parse_uploaded_file("sample.pdf", pdf_bytes)
    assert parsed_pdf["file_type"] == "pdf"
    assert len(parsed_pdf["text"]) > 0


def test_ai_personalities_personas():
    personas_list = list_personas()
    assert len(personas_list) >= 5

    default_persona = get_persona("default")
    assert default_persona.id == "default"

    code_architect = get_persona("code_architect")
    assert code_architect.name == "Senior Code Architect"
    assert "Software" in code_architect.system_prompt or "Code" in code_architect.system_prompt


def test_multimodal_vision_encoder():
    config = GPTConfig.gpt_micro(vocab_size=100)
    gpt_model = GPT(config)
    multimodal_gpt = MultimodalGPT(gpt_model, patch_size=16)

    # Synthetic RGB image tensor (batch_size=2, channels=3, H=64, W=64)
    images = torch.randn(2, 3, 64, 64)
    text_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.long)

    logits, loss = multimodal_gpt(idx=text_ids, images=images)
    assert logits.dim() == 3
    assert logits.size(0) == 2  # batch_size
    assert logits.size(-1) == 100  # vocab_size


def test_plugin_tool_calling_framework():
    search_out = web_search("PyTorch GPT")
    assert "PyTorch" in search_out

    repl_out = python_repl("12 * 12")
    assert "144" in repl_out

    calc_out = calculator("50 + 50")
    assert "100" in calc_out

    sample_text = "Here is the result: [TOOL: calc('10 + 20')]"
    processed_text, tool_calls = process_tool_calls(sample_text)

    assert len(tool_calls) == 1
    assert tool_calls[0]["tool"] == "calc"
    assert "30" in processed_text
