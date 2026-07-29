import { ChatSession, Message } from '../types';

const STORAGE_KEY = 'novexa_chat_sessions_v1';

export function loadChatSessions(): ChatSession[] {
  if (typeof window === 'undefined') return [];
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    return JSON.parse(raw);
  } catch {
    return [];
  }
}

export function saveChatSessions(sessions: ChatSession[]): void {
  if (typeof window === 'undefined') return;
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions));
  } catch {
    // Ignore storage errors
  }
}

export function createNewSession(firstUserMessage?: string): ChatSession {
  const now = new Date().toISOString();
  const id = 'session_' + Date.now();
  const title = firstUserMessage
    ? firstUserMessage.slice(0, 30) + (firstUserMessage.length > 30 ? '...' : '')
    : 'New Conversation';

  return {
    id,
    title,
    messages: [],
    createdAt: now,
    updatedAt: now,
  };
}
