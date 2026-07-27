import { HealthStatus, Message, ModelConfig, ModelInfo, Persona, UploadedFile } from '../types';

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

export async function fetchPersonas(): Promise<Persona[]> {
  try {
    const res = await fetch(`${API_BASE_URL}/personas`, { cache: 'no-store' });
    if (!res.ok) return [];
    return await res.json();
  } catch {
    return [
      { id: 'default', name: 'General Assistant', icon: '🤖', description: 'Balanced AI assistant.' },
      { id: 'code_architect', name: 'Senior Code Architect', icon: '💻', description: 'Expert software engineer.' },
      { id: 'creative_writer', name: 'Creative Storyteller', icon: '🎨', description: 'Imaginative author.' },
      { id: 'research_scientist', name: 'Academic Researcher', icon: '🔬', description: 'Analytical researcher.' },
      { id: 'math_tutor', name: 'Math & Logic Tutor', icon: '🧮', description: 'Step-by-step math tutor.' },
    ];
  }
}

export async function uploadDocumentFile(file: File): Promise<UploadedFile | null> {
  try {
    const formData = new FormData();
    formData.append('file', file);

    const res = await fetch(`${API_BASE_URL}/upload`, {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}

export async function streamChatResponse(
  messages: Message[],
  config: ModelConfig,
  fileId?: string,
  onChunk?: (chunk: string, toolCalls?: any[]) => void
): Promise<string> {
  try {
    const payload = {
      messages: messages.map((m) => ({ role: m.role, content: m.content })),
      max_new_tokens: config.max_new_tokens,
      temperature: config.temperature,
      top_k: config.top_k,
      top_p: config.top_p,
      persona_id: config.persona_id,
      file_id: fileId,
    };

    const res = await fetch(`${API_BASE_URL}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (res.ok) {
      const data = await res.json();
      const content = data.message?.content || 'No response generated.';
      const toolCalls = data.tool_calls || null;
      
      const words = content.split(' ');
      let accumulated = '';
      for (let i = 0; i < words.length; i++) {
        const word = words[i] + (i === words.length - 1 ? '' : ' ');
        accumulated += word;
        if (onChunk) onChunk(word, i === words.length - 1 ? toolCalls : undefined);
        await new Promise((r) => setTimeout(r, 30));
      }
      return accumulated;
    }
  } catch {
    // Offline fallback
  }

  const lastUserMessage = messages[messages.length - 1]?.content || 'Hello';
  const fallbackText = `MyGPT Response (Offline Mode):
I am your custom PyTorch GPT model with Personas & Tool support. I received your prompt: "${lastUserMessage}".

Active Configuration:
• Persona ID: ${config.persona_id}
• Attached File ID: ${fileId || 'None'}`;

  const words = fallbackText.split(' ');
  let accumulated = '';
  for (let i = 0; i < words.length; i++) {
    const word = words[i] + (i === words.length - 1 ? '' : ' ');
    accumulated += word;
    if (onChunk) onChunk(word);
    await new Promise((r) => setTimeout(r, 25));
  }
  return accumulated;
}
