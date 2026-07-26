import { HealthStatus, Message, ModelConfig, ModelInfo } from '../types';

const API_BASE_URL = 'http://127.0.0.1:8000/api/v1';

export async function fetchHealth(): Promise<HealthStatus | null> {
  try {
    const res = await fetch(`${API_BASE_URL}/health`, { cache: 'no-store' });
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}

export async function fetchModelInfo(): Promise<ModelInfo | null> {
  try {
    const res = await fetch(`${API_BASE_URL}/info`, { cache: 'no-store' });
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}

export async function streamChatResponse(
  messages: Message[],
  config: ModelConfig,
  onChunk: (chunk: string) => void
): Promise<string> {
  try {
    const payload = {
      messages: messages.map((m) => ({ role: m.role, content: m.content })),
      max_new_tokens: config.max_new_tokens,
      temperature: config.temperature,
      top_k: config.top_k,
      top_p: config.top_p,
    };

    const res = await fetch(`${API_BASE_URL}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (res.ok) {
      const data = await res.json();
      const content = data.message?.content || 'No response generated.';
      
      // Simulate streaming typewriter effect for smooth UI feedback
      const words = content.split(' ');
      let accumulated = '';
      for (let i = 0; i < words.length; i++) {
        const word = words[i] + (i === words.length - 1 ? '' : ' ');
        accumulated += word;
        onChunk(word);
        await new Promise((r) => setTimeout(r, 40));
      }
      return accumulated;
    }
  } catch {
    // Fallback streaming logic when backend API server is offline
  }

  // Fallback offline simulator
  const lastUserMessage = messages[messages.length - 1]?.content || 'Hello';
  const fallbackText = `MyGPT Response (Offline Mode):
I am your custom PyTorch GPT model (0.81M parameters). I received your query: "${lastUserMessage}".

Key Architectural Features:
• Scaled Dot-Product Self-Attention with Causal Masking
• Multi-Head Attention (4 heads, d_model=128)
• Pre-LayerNorm & Residual Skip Connections
• Weight Tying between Token Embeddings and LM Head
• Automatic Mixed Precision & Cosine Warmup Optimizer`;

  const words = fallbackText.split(' ');
  let accumulated = '';
  for (let i = 0; i < words.length; i++) {
    const word = words[i] + (i === words.length - 1 ? '' : ' ');
    accumulated += word;
    onChunk(word);
    await new Promise((r) => setTimeout(r, 30));
  }
  return accumulated;
}
