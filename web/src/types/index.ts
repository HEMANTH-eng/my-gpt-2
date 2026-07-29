export interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: string;
  isStreaming?: boolean;
  tool_calls?: Array<{ tool: string; argument: string; output: string }>;
  isLiked?: boolean;
  isDisliked?: boolean;
  tokensCount?: number;
  latencyMs?: number;
}

export interface ChatSession {
  id: string;
  title: string;
  messages: Message[];
  persona_id?: string;
  file_id?: string;
  workspace_id?: string;
  isPinned?: boolean;
  isStarred?: boolean;
  folderId?: string;
  createdAt: string;
  updatedAt: string;
}

export interface Workspace {
  id: string;
  name: string;
  icon: string;
  type: 'personal' | 'work' | 'school' | 'projects';
  description: string;
  chatsCount: number;
  filesCount: number;
}

export interface MemoryItem {
  id: string;
  key: string;
  value: string;
  category: 'preference' | 'fact' | 'instruction';
  createdAt: string;
}

export interface NotificationItem {
  id: string;
  title: string;
  message: string;
  type: 'info' | 'success' | 'warning' | 'error';
  timestamp: string;
  read: boolean;
}

export interface ModelConfig {
  temperature: number;
  top_k: number;
  top_p: number;
  max_new_tokens: number;
  greedy: boolean;
  persona_id: string;
  presence_penalty?: number;
  frequency_penalty?: number;
  streaming?: boolean;
  reasoning_mode?: boolean;
  vision_mode?: boolean;
  voice_mode?: boolean;
  memory_enabled?: boolean;
}

export interface AppearanceSettings {
  theme: 'dark' | 'light' | 'system';
  accentColor: string;
  fontSize: 'sm' | 'md' | 'lg';
  compactMode: boolean;
  sidebarWidth: number;
  chatDensity: 'comfortable' | 'compact';
  animations: boolean;
}

export interface Persona {
  id: string;
  name: string;
  icon: string;
  description: string;
}

export interface UploadedFile {
  id: string;
  filename: string;
  file_type: string;
  file_size: number;
  text_preview: string;
  uploadedAt?: string;
}

export interface User {
  id: number;
  username: string;
  email: string;
  displayName?: string;
  avatarUrl?: string;
  bio?: string;
  website?: string;
  location?: string;
  joinedDate?: string;
  role?: string;
  plan?: 'free' | 'pro' | 'team' | 'enterprise';
  tokensUsed?: number;
  storageUsedMB?: number;
  apiCallsCount?: number;
}

export interface AuthToken {
  access_token: string;
  token_type: string;
  user: User;
}

export interface ModelInfo {
  model_name: string;
  num_parameters: number;
  vocab_size: number;
  d_model: number;
  n_layer: number;
  n_head: number;
  max_seq_len: number;
}

export interface HealthStatus {
  status: string;
  model_loaded: boolean;
  device: string;
}

export interface ApiKeyItem {
  id: string;
  name: string;
  prefix: string;
  created_at: string;
}

