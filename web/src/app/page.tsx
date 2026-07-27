'use client';

import React, { useEffect, useRef, useState } from 'react';
import { AuthModal } from '../components/AuthModal';
import { ChatInput } from '../components/ChatInput';
import { ChatMessage } from '../components/ChatMessage';
import { ConfigModal } from '../components/ConfigModal';
import { Header } from '../components/Header';
import { Sidebar } from '../components/Sidebar';
import {
  fetchHealth,
  fetchModelInfo,
  fetchPersonas,
  streamChatResponse,
  uploadDocumentFile,
} from '../lib/api';
import { createNewSession, loadChatSessions, saveChatSessions } from '../lib/storage';
import { ChatSession, HealthStatus, Message, ModelConfig, ModelInfo, Persona, User } from '../types';

export default function Home() {
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [personas, setPersonas] = useState<Persona[]>([]);
  const [currentUser, setCurrentUser] = useState<User | null>(null);

  const [config, setConfig] = useState<ModelConfig>({
    temperature: 0.7,
    top_k: 40,
    top_p: 0.95,
    max_new_tokens: 100,
    greedy: false,
    persona_id: 'default',
  });

  const [isLoading, setIsLoading] = useState(false);
  const [isConfigOpen, setIsConfigOpen] = useState(false);
  const [isAuthOpen, setIsAuthOpen] = useState(false);
  const [isMobileSidebarOpen, setIsMobileSidebarOpen] = useState(false);

  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const loadedSessions = loadChatSessions();
    if (loadedSessions.length > 0) {
      setSessions(loadedSessions);
      setActiveSessionId(loadedSessions[0].id);
    } else {
      const initial = createNewSession();
      setSessions([initial]);
      setActiveSessionId(initial.id);
      saveChatSessions([initial]);
    }

    fetchHealth().then(setHealth);
    fetchModelInfo().then(setModelInfo);
    fetchPersonas().then(setPersonas);
  }, []);

  const activeSession = sessions.find((s) => s.id === activeSessionId) || sessions[0];
  const messages = activeSession?.messages || [];

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  const handleNewChat = () => {
    const newSess = createNewSession();
    const updated = [newSess, ...sessions];
    setSessions(updated);
    setActiveSessionId(newSess.id);
    saveChatSessions(updated);
  };

  const handleDeleteSession = (id: string) => {
    const updated = sessions.filter((s) => s.id !== id);
    setSessions(updated);
    if (activeSessionId === id) {
      setActiveSessionId(updated[0]?.id || null);
    }
    saveChatSessions(updated);
  };

  const handleClearAll = () => {
    const initial = createNewSession();
    setSessions([initial]);
    setActiveSessionId(initial.id);
    saveChatSessions([initial]);
  };

  const handleFileUpload = async (file: File): Promise<string | null> => {
    const uploaded = await uploadDocumentFile(file);
    if (uploaded) {
      return uploaded.id;
    }
    return 'local_file_' + Date.now();
  };

  const handleSendMessage = async (text: string, fileId?: string) => {
    if (!activeSessionId || isLoading) return;

    const now = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const userMsg: Message = {
      id: 'msg_' + Date.now(),
      role: 'user',
      content: text,
      timestamp: now,
    };

    const isFirstMessage = activeSession.messages.length === 0;
    const updatedTitle = isFirstMessage ? text.slice(0, 25) + '...' : activeSession.title;

    const assistantMsgId = 'msg_' + (Date.now() + 1);
    const initialAssistantMsg: Message = {
      id: assistantMsgId,
      role: 'assistant',
      content: '',
      timestamp: now,
      isStreaming: true,
    };

    const nextMessages = [...activeSession.messages, userMsg, initialAssistantMsg];
    const updatedSessions = sessions.map((s) =>
      s.id === activeSessionId
        ? { ...s, title: updatedTitle, messages: nextMessages, updatedAt: new Date().toISOString() }
        : s
    );

    setSessions(updatedSessions);
    saveChatSessions(updatedSessions);
    setIsLoading(true);

    let accumulatedContent = '';

    await streamChatResponse(
      [...activeSession.messages, userMsg],
      config,
      fileId,
      (chunk, toolCalls) => {
        accumulatedContent += chunk;
        setSessions((prevSessions) =>
          prevSessions.map((s) => {
            if (s.id !== activeSessionId) return s;
            const msgs = s.messages.map((m) =>
              m.id === assistantMsgId
                ? { ...m, content: accumulatedContent, tool_calls: toolCalls || m.tool_calls }
                : m
            );
            return { ...s, messages: msgs };
          })
        );
      }
    );

    setSessions((prevSessions) => {
      const finalized = prevSessions.map((s) => {
        if (s.id !== activeSessionId) return s;
        const msgs = s.messages.map((m) =>
          m.id === assistantMsgId ? { ...m, isStreaming: false } : m
        );
        return { ...s, messages: msgs };
      });
      saveChatSessions(finalized);
      return finalized;
    });

    setIsLoading(false);
  };

  return (
    <div className="flex h-screen bg-zinc-950 text-zinc-100 font-sans overflow-hidden">
      <Sidebar
        sessions={sessions}
        activeSessionId={activeSessionId}
        modelInfo={modelInfo}
        onSelectSession={setActiveSessionId}
        onNewChat={handleNewChat}
        onDeleteSession={handleDeleteSession}
        onClearAll={handleClearAll}
        isOpen={isMobileSidebarOpen}
        onCloseMobile={() => setIsMobileSidebarOpen(false)}
      />

      <div className="flex-1 flex flex-col min-w-0 bg-gradient-to-b from-zinc-950 via-zinc-950 to-zinc-900">
        <Header
          health={health}
          personas={personas}
          selectedPersonaId={config.persona_id}
          onSelectPersona={(id) => setConfig({ ...config, persona_id: id })}
          currentUser={currentUser}
          onOpenAuth={() => setIsAuthOpen(true)}
          onOpenConfig={() => setIsConfigOpen(true)}
          onNewChat={handleNewChat}
        />

        <main className="flex-1 overflow-y-auto px-4 py-6 space-y-4 scrollbar-thin">
          {messages.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-center p-6 max-w-md mx-auto">
              <div className="w-16 h-16 rounded-2xl bg-gradient-to-tr from-cyan-500 to-indigo-600 p-[1px] mb-4 shadow-xl shadow-cyan-950">
                <div className="w-full h-full bg-zinc-950 rounded-[15px] flex items-center justify-center text-cyan-400 font-bold text-xl">
                  GPT
                </div>
              </div>
              <h2 className="text-lg font-semibold text-zinc-100 mb-1">
                MyGPT Multimodal Platform
              </h2>
              <p className="text-xs text-zinc-400 leading-relaxed mb-6 font-mono">
                Featuring User Accounts, File Ingestion (PDF, DOCX, TXT), Image Understanding, Voice Input/TTS, Personas, and Plugin Tools.
              </p>
            </div>
          ) : (
            messages.map((msg) => <ChatMessage key={msg.id} message={msg} />)
          )}
          <div ref={messagesEndRef} />
        </main>

        <footer className="border-t border-zinc-800/60 bg-zinc-950/80 backdrop-blur-md">
          <ChatInput
            onSendMessage={handleSendMessage}
            onFileUpload={handleFileUpload}
            isLoading={isLoading}
            config={config}
          />
        </footer>
      </div>

      <ConfigModal
        isOpen={isConfigOpen}
        onClose={() => setIsConfigOpen(false)}
        config={config}
        onChangeConfig={setConfig}
      />

      <AuthModal
        isOpen={isAuthOpen}
        onClose={() => setIsAuthOpen(false)}
        onLoginSuccess={(user) => setCurrentUser(user)}
      />
    </div>
  );
}
