'use client';

import React, { useEffect, useRef, useState } from 'react';
import { AdminModal } from '../components/AdminModal';
import { AuthModal } from '../components/AuthModal';
import { ChatInput } from '../components/ChatInput';
import { ChatMessage } from '../components/ChatMessage';
import { DashboardModal } from '../components/DashboardModal';
import { FileManagerModal } from '../components/FileManagerModal';
import { LandingPage } from '../components/LandingPage';
import { MemoryModal } from '../components/MemoryModal';
import { Navbar } from '../components/Navbar';
import { NotificationDrawer } from '../components/NotificationDrawer';
import { ProfileModal } from '../components/ProfileModal';
import { SettingsModal } from '../components/SettingsModal';
import { Sidebar } from '../components/Sidebar';
import {
  fetchHealth,
  fetchModelInfo,
  fetchPersonas,
  streamChatResponse,
  uploadDocumentFile,
} from '../lib/api';
import { createNewSession, loadChatSessions, saveChatSessions } from '../lib/storage';
import {
  ChatSession,
  HealthStatus,
  MemoryItem,
  Message,
  ModelConfig,
  ModelInfo,
  NotificationItem,
  Persona,
  UploadedFile,
  User,
  Workspace,
} from '../types';

export default function Home() {
  const [activeView, setActiveView] = useState<'landing' | 'studio'>('landing');

  // Modals & Panels State
  const [isAuthOpen, setIsAuthOpen] = useState(false);
  const [isProfileOpen, setIsProfileOpen] = useState(false);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [isDashboardOpen, setIsDashboardOpen] = useState(false);
  const [isMemoryOpen, setIsMemoryOpen] = useState(false);
  const [isFileManagerOpen, setIsFileManagerOpen] = useState(false);
  const [isNotificationsOpen, setIsNotificationsOpen] = useState(false);
  const [isAdminOpen, setIsAdminOpen] = useState(false);
  const [isMobileSidebarOpen, setIsMobileSidebarOpen] = useState(false);

  // Data & Sessions
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [personas, setPersonas] = useState<Persona[]>([]);
  const [currentUser, setCurrentUser] = useState<User | null>({
    id: 1,
    username: 'novexa_dev',
    email: 'dev@novexa.ai',
    displayName: 'Lead AI Engineer',
    plan: 'pro',
    tokensUsed: 142500,
    storageUsedMB: 24.8,
  });

  // Workspaces State
  const [workspaces] = useState<Workspace[]>([
    { id: 'personal', name: 'Personal Workspace', icon: '👤', type: 'personal', description: 'General research & personal tasks', chatsCount: 8, filesCount: 3 },
    { id: 'work', name: 'Work & Engineering', icon: '💼', type: 'work', description: 'PyTorch models, API code, and agent flows', chatsCount: 14, filesCount: 12 },
    { id: 'school', name: 'Academic & Papers', icon: '🎓', type: 'school', description: 'Transformer literature and math derivations', chatsCount: 4, filesCount: 5 },
    { id: 'projects', name: 'Novexa AI Core', icon: '⚡', type: 'projects', description: 'BPE Tokenizer, SFT training, & REPL tools', chatsCount: 19, filesCount: 8 },
  ]);
  const [activeWorkspaceId, setActiveWorkspaceId] = useState('work');

  // Memories & Files State
  const [memories, setMemories] = useState<MemoryItem[]>([
    { id: '1', key: 'Preferred Stack', value: 'Python, PyTorch, Next.js, FastAPI', category: 'preference', createdAt: '2026-07-29' },
    { id: '2', key: 'Model Target', value: 'Novexa-Micro (0.83M params)', category: 'fact', createdAt: '2026-07-29' },
  ]);

  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([
    { id: 'f1', filename: 'Novexa_AI_Architecture.pdf', file_type: 'pdf', file_size: 1420000, text_preview: 'Causal Self-Attention Layer Implementation' },
    { id: 'f2', filename: 'training_corpus.txt', file_type: 'txt', file_size: 32000, text_preview: 'SFT Training data pairs' },
  ]);

  const [notifications, setNotifications] = useState<NotificationItem[]>([
    { id: 'n1', title: 'Novexa AI Engine Ready', message: 'FastAPI Backend online on http://127.0.0.1:8000', type: 'success', timestamp: '12:50 PM', read: false },
    { id: 'n2', title: 'PyTorch GPU Detection', message: 'Model running on CUDA/CPU high-throughput pool', type: 'info', timestamp: '12:45 PM', read: false },
  ]);

  const [config, setConfig] = useState<ModelConfig>({
    temperature: 0.7,
    top_k: 40,
    top_p: 0.95,
    max_new_tokens: 100,
    greedy: false,
    persona_id: 'default',
    memory_enabled: true,
    streaming: true,
  });

  const [isLoading, setIsLoading] = useState(false);
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
    const newF: UploadedFile = {
      id: uploaded?.id || 'f_' + Date.now(),
      filename: file.name,
      file_type: file.name.split('.').pop() || 'file',
      file_size: file.size,
      text_preview: 'Ingested context file',
    };
    setUploadedFiles((prev) => [newF, ...prev]);
    return newF.id;
  };

  const handleDeleteFile = (id: string) => {
    setUploadedFiles(uploadedFiles.filter((f) => f.id !== id));
  };

  const handleAddMemory = (key: string, value: string, category: 'preference' | 'fact' | 'instruction') => {
    const newM: MemoryItem = {
      id: Date.now().toString(),
      key,
      value,
      category,
      createdAt: new Date().toISOString().split('T')[0],
    };
    setMemories([newM, ...memories]);
  };

  const handleDeleteMemory = (id: string) => {
    setMemories(memories.filter((m) => m.id !== id));
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

  const unreadCount = notifications.filter((n) => !n.read).length;

  return (
    <div className="flex flex-col h-screen bg-[#090D16] text-zinc-100 font-sans overflow-hidden">
      {/* Top Universal Navbar */}
      <Navbar
        activeView={activeView}
        onSelectView={setActiveView}
        workspaces={workspaces}
        activeWorkspaceId={activeWorkspaceId}
        onSelectWorkspace={setActiveWorkspaceId}
        onOpenAuth={() => setIsAuthOpen(true)}
        onOpenProfile={() => setIsProfileOpen(true)}
        onOpenSettings={() => setIsSettingsOpen(true)}
        onOpenDashboard={() => setIsDashboardOpen(true)}
        onOpenNotifications={() => setIsNotificationsOpen(true)}
        unreadNotificationsCount={unreadCount}
        currentUser={currentUser}
      />

      {/* Main View Router */}
      {activeView === 'landing' ? (
        <div className="flex-1 overflow-y-auto">
          <LandingPage
            onLaunchStudio={() => setActiveView('studio')}
            onOpenAuth={() => setIsAuthOpen(true)}
          />
        </div>
      ) : (
        <div className="flex-1 flex min-h-0 overflow-hidden">
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
            workspaces={workspaces}
            activeWorkspaceId={activeWorkspaceId}
            onSelectWorkspace={setActiveWorkspaceId}
            onOpenMemory={() => setIsMemoryOpen(true)}
            onOpenFileManager={() => setIsFileManagerOpen(true)}
          />

          <div className="flex-1 flex flex-col min-w-0 bg-gradient-to-b from-[#090D16] via-zinc-950 to-zinc-950 relative">
            <main className="flex-1 overflow-y-auto px-4 py-6 space-y-4">
              {messages.length === 0 ? (
                <div className="h-full flex flex-col items-center justify-center text-center p-6 max-w-lg mx-auto">
                  <div className="w-16 h-16 rounded-2xl bg-gradient-to-tr from-[#4F46E5] via-[#7C3AED] to-[#06B6D4] p-[1px] mb-4 shadow-2xl shadow-[#4F46E5]/40 animate-pulse-glow">
                    <div className="w-full h-full bg-zinc-950 rounded-[15px] flex items-center justify-center text-[#06B6D4] font-extrabold text-2xl">
                      NA
                    </div>
                  </div>
                  <h2 className="text-xl font-extrabold text-white mb-2">
                    Novexa AI Studio
                  </h2>
                  <p className="text-xs text-zinc-400 leading-relaxed mb-6 font-mono">
                    Build. Think. Create. Powered by PyTorch GPT, RAG Document Parsing, Vision, and Autonomous AI Agents.
                  </p>

                  <div className="grid grid-cols-2 gap-2 text-xs w-full max-w-sm">
                    <button
                      onClick={() => handleSendMessage('Explain causal dot-product self-attention in PyTorch.')}
                      className="p-3 rounded-xl bg-zinc-900/80 hover:bg-zinc-800 border border-zinc-800 text-left text-zinc-300"
                    >
                      💡 Self-Attention Logic
                    </button>
                    <button
                      onClick={() => handleSendMessage('Write a Python function to compute the Fibonacci sequence up to n.')}
                      className="p-3 rounded-xl bg-zinc-900/80 hover:bg-zinc-800 border border-zinc-800 text-left text-zinc-300"
                    >
                      💻 Python Fibonacci REPL
                    </button>
                  </div>
                </div>
              ) : (
                messages.map((msg) => <ChatMessage key={msg.id} message={msg} />)
              )}
              <div ref={messagesEndRef} />
            </main>

            <footer className="border-t border-zinc-800/80 bg-zinc-950/90 backdrop-blur-xl">
              <ChatInput
                onSendMessage={handleSendMessage}
                onFileUpload={handleFileUpload}
                isLoading={isLoading}
                config={config}
              />
            </footer>
          </div>
        </div>
      )}

      {/* Global Application Modals */}
      <ProfileModal
        isOpen={isProfileOpen}
        onClose={() => setIsProfileOpen(false)}
        currentUser={currentUser}
        onUpdateUser={setCurrentUser}
      />

      <SettingsModal
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
        config={config}
        onChangeConfig={setConfig}
      />

      <DashboardModal
        isOpen={isDashboardOpen}
        onClose={() => setIsDashboardOpen(false)}
        health={health}
        modelInfo={modelInfo}
      />

      <MemoryModal
        isOpen={isMemoryOpen}
        onClose={() => setIsMemoryOpen(false)}
        memories={memories}
        onAddMemory={handleAddMemory}
        onDeleteMemory={handleDeleteMemory}
      />

      <FileManagerModal
        isOpen={isFileManagerOpen}
        onClose={() => setIsFileManagerOpen(false)}
        files={uploadedFiles}
        onUploadFile={async (file) => {
          await handleFileUpload(file);
        }}
        onDeleteFile={handleDeleteFile}
      />

      <NotificationDrawer
        isOpen={isNotificationsOpen}
        onClose={() => setIsNotificationsOpen(false)}
        notifications={notifications}
        onMarkAllAsRead={() => setNotifications(notifications.map((n) => ({ ...n, read: true })))}
      />

      <AdminModal
        isOpen={isAdminOpen}
        onClose={() => setIsAdminOpen(false)}
      />

      <AuthModal
        isOpen={isAuthOpen}
        onClose={() => setIsAuthOpen(false)}
        onLoginSuccess={(user) => setCurrentUser(user)}
      />
    </div>
  );
}

