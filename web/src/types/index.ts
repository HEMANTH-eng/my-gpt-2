export interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: string;
  isStreaming?: boolean;
}

export interface ChatSession {
  id: string;
  title: string;
  messages: Message[];
  createdAt: string;
  updatedAt: string;
}

export interface ModelConfig {
  temperature: number;
  top_k: number;
  top_p: number;
  max_new_tokens: number;
  greedy: boolean;
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
