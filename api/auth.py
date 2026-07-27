from datetime import datetime, timedelta
import hashlib
import secrets
from typing import List, Optional
import bcrypt
from fastapi import APIRouter, Depends, Header, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
import jwt
from pydantic import BaseModel, ConfigDict, EmailStr, Field
from sqlalchemy.orm import Session

from api.database import get_db
from api.models_db import ApiKeyDB, User
from utils.audit import global_audit_logger
from utils.logger import get_logger

logger = get_logger("auth")

SECRET_KEY = "mygpt_secret_key_change_in_production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login", auto_error=False)

router = APIRouter(prefix="/api/v1/auth", tags=["Authentication"])


class UserRegister(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=6)
    role: Optional[str] = Field(default="user", description="Role: 'admin', 'user', or 'guest'")


class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    role: str

    model_config = ConfigDict(from_attributes=True)


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


class ApiKeyCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=50)
    scopes: Optional[str] = Field(default="read,write")


class ApiKeyResponse(BaseModel):
    id: str
    name: str
    prefix: str
    scopes: str
    api_key: Optional[str] = None  # Returned only upon creation


def verify_password(plain_password: str, hashed_password: str) -> bool:
    pwd_bytes = plain_password.encode("utf-8")[:72]
    hash_bytes = hashed_password.encode("utf-8")
    return bcrypt.checkpw(pwd_bytes, hash_bytes)


def get_password_hash(password: str) -> str:
    pwd_bytes = password.encode("utf-8")[:72]
    return bcrypt.hashpw(pwd_bytes, bcrypt.gensalt()).decode("utf-8")


def generate_api_key() -> tuple[str, str, str]:
    """Generates raw API key, prefix, and SHA-256 hash.

    Returns:
        (raw_key, prefix, key_hash)
    """
    raw_key = f"mgpt_{secrets.token_hex(24)}"
    prefix = raw_key[:8]
    key_hash = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
    return raw_key, prefix, key_hash


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(
    token: Optional[str] = Depends(oauth2_scheme),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    db: Session = Depends(get_db),
) -> Optional[User]:
    """Authenticates user via JWT Bearer Token or X-API-Key Header."""
    # 1. API Key Auth Check
    if x_api_key:
        key_hash = hashlib.sha256(x_api_key.encode("utf-8")).hexdigest()
        db_key = db.query(ApiKeyDB).filter(ApiKeyDB.key_hash == key_hash).first()
        if db_key and db_key.user:
            return db_key.user

    # 2. JWT Token Auth Check
    if not token:
        return None
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            return None
    except Exception:
        return None

    return db.query(User).filter(User.username == username).first()


ROLE_HIERARCHY = {"admin": 3, "user": 2, "guest": 1}


def require_role(required_role: str):
    """FastAPI Dependency for Role-Based Access Control (RBAC)."""
    def role_checker(current_user: Optional[User] = Depends(get_current_user)):
        if not current_user:
            global_audit_logger.log_event("rbac_check", status="denied", details={"required_role": required_role})
            raise HTTPException(status_code=401, detail="Authentication required")

        user_level = ROLE_HIERARCHY.get(current_user.role, 1)
        req_level = ROLE_HIERARCHY.get(required_role, 2)

        if user_level < req_level:
            global_audit_logger.log_event(
                "rbac_check",
                user_id=current_user.username,
                status="denied",
                details={"required_role": required_role, "user_role": current_user.role},
            )
            raise HTTPException(status_code=403, detail=f"Forbidden: Requires minimum '{required_role}' role")
        return current_user
    return role_checker



@router.post("/register", response_model=Token, status_code=status.HTTP_201_CREATED)
def register(user_in: UserRegister, db: Session = Depends(get_db)):
    if db.query(User).filter(User.username == user_in.username).first():
        raise HTTPException(status_code=400, detail="Username already registered")
    if db.query(User).filter(User.email == user_in.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")

    user = User(
        username=user_in.username,
        email=user_in.email,
        hashed_password=get_password_hash(user_in.password),
        role=user_in.role or "user",
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    global_audit_logger.log_event("user_register", user_id=user.username, status="success")
    access_token = create_access_token(data={"sub": user.username})
    return Token(access_token=access_token, user=UserResponse.model_validate(user))


@router.post("/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        global_audit_logger.log_event("user_login", user_id=form_data.username, status="denied")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    global_audit_logger.log_event("user_login", user_id=user.username, status="success")
    access_token = create_access_token(data={"sub": user.username})
    return Token(access_token=access_token, user=UserResponse.model_validate(user))


@router.get("/me", response_model=UserResponse)
def get_me(current_user: Optional[User] = Depends(get_current_user)):
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return UserResponse.model_validate(current_user)


@router.post("/api-keys", response_model=ApiKeyResponse, status_code=status.HTTP_201_CREATED)
def create_api_key(
    request: ApiKeyCreateRequest,
    current_user: User = Depends(require_role("user")),
    db: Session = Depends(get_db),
):
    """Creates a new API Key for the authenticated user."""
    raw_key, prefix, key_hash = generate_api_key()
    key_id = f"key_{secrets.token_hex(8)}"

    db_key = ApiKeyDB(
        id=key_id,
        user_id=current_user.id,
        name=request.name,
        key_hash=key_hash,
        prefix=prefix,
        scopes=request.scopes or "read,write",
    )
    db.add(db_key)
    db.commit()
    db.refresh(db_key)

    global_audit_logger.log_event("api_key_create", user_id=current_user.username, status="success", details={"key_id": key_id})

    return ApiKeyResponse(
        id=db_key.id,
        name=db_key.name,
        prefix=db_key.prefix,
        scopes=db_key.scopes,
        api_key=raw_key,
    )


@router.get("/api-keys", response_model=List[ApiKeyResponse])
def list_api_keys(
    current_user: User = Depends(require_role("user")),
    db: Session = Depends(get_db),
):
    """Lists all active API Keys for the authenticated user."""
    keys = db.query(ApiKeyDB).filter(ApiKeyDB.user_id == current_user.id).all()
    return [
        ApiKeyResponse(id=k.id, name=k.name, prefix=k.prefix, scopes=k.scopes)
        for k in keys
    ]
