export interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: string;
  isStreaming?: boolean;
  tool_calls?: Array<{ tool: string; argument: string; output: string }>;
}

export interface ChatSession {
  id: string;
  title: string;
  messages: Message[];
  persona_id?: string;
  file_id?: string;
  createdAt: string;
  updatedAt: string;
}

export interface ModelConfig {
  temperature: number;
  top_k: number;
  top_p: number;
  max_new_tokens: number;
  greedy: boolean;
  persona_id: string;
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
}

export interface User {
  id: number;
  username: string;
  email: string;
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
