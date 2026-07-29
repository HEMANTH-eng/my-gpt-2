'use client';

import React, { useState } from 'react';
import { ChatSession, ModelInfo, Workspace } from '../types';

interface SidebarProps {
  sessions: ChatSession[];
  activeSessionId: string | null;
  modelInfo: ModelInfo | null;
  onSelectSession: (id: string) => void;
  onNewChat: () => void;
  onDeleteSession: (id: string) => void;
  onClearAll: () => void;
  isOpen: boolean;
  onCloseMobile: () => void;
  workspaces: Workspace[];
  activeWorkspaceId: string;
  onSelectWorkspace: (id: string) => void;
  onOpenMemory: () => void;
  onOpenFileManager: () => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
  sessions,
  activeSessionId,
  modelInfo,
  onSelectSession,
  onNewChat,
  onDeleteSession,
  onClearAll,
  isOpen,
  onCloseMobile,
  workspaces,
  activeWorkspaceId,
  onSelectWorkspace,
  onOpenMemory,
  onOpenFileManager,
}) => {
  const [searchQuery, setSearchQuery] = useState('');

  const filteredSessions = sessions.filter((s) =>
    s.title.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <>
      {/* Mobile Backdrop */}
      {isOpen && (
        <div
          onClick={onCloseMobile}
          className="fixed inset-0 bg-black/60 backdrop-blur-sm z-30 md:hidden"
        />
      )}

      <aside
        className={`fixed md:static inset-y-0 left-0 z-40 w-72 bg-zinc-950/95 md:bg-zinc-950/80 border-r border-zinc-800/80 flex flex-col transition-transform duration-300 ease-in-out ${
          isOpen ? 'translate-x-0' : '-translate-x-full md:translate-x-0'
        }`}
      >
        {/* New Chat & Quick Tools */}
        <div className="p-4 border-b border-zinc-800/60 space-y-2">
          <button
            onClick={() => {
              onNewChat();
              onCloseMobile();
            }}
            className="w-full py-2.5 px-4 rounded-xl glow-button text-white font-semibold text-xs tracking-wide shadow-lg transition-all flex items-center justify-center gap-2"
          >
            <span>+</span>
            <span>New Conversation</span>
          </button>

          {/* Search Bar */}
          <input
            type="text"
            placeholder="Search chats..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full bg-zinc-900 border border-zinc-800 rounded-xl px-3 py-1.5 text-xs text-zinc-100 focus:outline-none focus:border-[#4F46E5]"
          />

          {/* Quick Shortcuts */}
          <div className="grid grid-cols-2 gap-1.5 pt-1 text-[11px]">
            <button
              onClick={onOpenMemory}
              className="py-1.5 px-2 rounded-lg bg-zinc-900 hover:bg-zinc-800 border border-zinc-800 text-zinc-300 flex items-center gap-1.5 justify-center"
            >
              🧠 Memory
            </button>
            <button
              onClick={onOpenFileManager}
              className="py-1.5 px-2 rounded-lg bg-zinc-900 hover:bg-zinc-800 border border-zinc-800 text-zinc-300 flex items-center gap-1.5 justify-center"
            >
              📁 Documents
            </button>
          </div>
        </div>

        {/* Sessions List */}
        <div className="flex-1 overflow-y-auto p-3 space-y-1">
          <div className="text-[10px] font-semibold uppercase tracking-wider text-zinc-400 px-3 py-1 flex justify-between">
            <span>Recent Chats</span>
            <span>({filteredSessions.length})</span>
          </div>

          {filteredSessions.length === 0 ? (
            <div className="text-center py-8 px-4 text-zinc-500 text-xs font-mono">
              No matching chats found.
            </div>
          ) : (
            filteredSessions.map((session) => {
              const isActive = session.id === activeSessionId;
              return (
                <div
                  key={session.id}
                  onClick={() => {
                    onSelectSession(session.id);
                    onCloseMobile();
                  }}
                  className={`group relative flex items-center justify-between px-3 py-2.5 rounded-xl text-xs cursor-pointer transition-all ${
                    isActive
                      ? 'bg-[#4F46E5]/20 text-[#06B6D4] border border-[#06B6D4]/40 font-medium shadow-sm'
                      : 'text-zinc-400 hover:text-zinc-200 hover:bg-zinc-900/60'
                  }`}
                >
                  <div className="flex items-center gap-2.5 truncate pr-6">
                    <span className={isActive ? 'text-[#06B6D4]' : 'text-zinc-500'}>💬</span>
                    <span className="truncate">{session.title}</span>
                  </div>

                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      onDeleteSession(session.id);
                    }}
                    className="opacity-0 group-hover:opacity-100 text-zinc-400 hover:text-rose-400 p-1 transition-opacity"
                    title="Delete Chat"
                  >
                    ✕
                  </button>
                </div>
              );
            })
          )}
        </div>

        {/* Bottom Model Summary Card */}
        <div className="p-3 border-t border-zinc-800/60 bg-zinc-950/90">
          <div className="p-3 rounded-xl bg-zinc-900/80 border border-zinc-800/60 text-xs">
            <div className="flex items-center justify-between text-zinc-200 font-medium mb-1">
              <span className="font-bold text-[#06B6D4]">Novexa-Micro</span>
              <span className="text-[10px] text-zinc-400 font-mono">0.83M PyTorch</span>
            </div>

            <div className="grid grid-cols-2 gap-1 text-[10px] text-zinc-400 font-mono mt-2 pt-2 border-t border-zinc-800/50">
              <div>Layers: {modelInfo?.n_layer || 4}</div>
              <div>Heads: {modelInfo?.n_head || 4}</div>
              <div>Embed: {modelInfo?.d_model || 128}</div>
              <div>Context: {modelInfo?.max_seq_len || 128}</div>
            </div>
          </div>

          {sessions.length > 0 && (
            <button
              onClick={onClearAll}
              className="mt-2 w-full text-center text-[11px] text-zinc-500 hover:text-rose-400 transition-colors py-1"
            >
              Clear All Chat History
            </button>
          )}
        </div>
      </aside>
    </>
  );
};

