'use client';

import React from 'react';
import { ChatSession, ModelInfo } from '../types';

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
}) => {
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
        className={`fixed md:static inset-y-0 left-0 z-40 w-72 bg-zinc-950/95 md:bg-zinc-950/60 border-r border-zinc-800/80 flex flex-col transition-transform duration-300 ease-in-out ${
          isOpen ? 'translate-x-0' : '-translate-x-full md:translate-x-0'
        }`}
      >
        {/* New Chat Section */}
        <div className="p-4 border-b border-zinc-800/60">
          <button
            onClick={() => {
              onNewChat();
              onCloseMobile();
            }}
            className="w-full py-2.5 px-4 rounded-xl bg-gradient-to-r from-cyan-600 to-indigo-600 hover:from-cyan-500 hover:to-indigo-500 text-white font-medium text-xs tracking-wide shadow-lg shadow-cyan-900/30 transition-all flex items-center justify-center gap-2"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
            </svg>
            New Conversation
          </button>
        </div>

        {/* Sessions List */}
        <div className="flex-1 overflow-y-auto p-3 space-y-1">
          <div className="text-[10px] font-semibold uppercase tracking-wider text-zinc-400 px-3 py-1">
            Conversations ({sessions.length})
          </div>

          {sessions.length === 0 ? (
            <div className="text-center py-8 px-4 text-zinc-400 text-xs font-mono">
              No conversations yet. Start a new chat above!
            </div>
          ) : (
            sessions.map((session) => {
              const isActive = session.id === activeSessionId;
              return (
                <div
                  key={session.id}
                  onClick={() => {
                    onSelectSession(session.id);
                    onCloseMobile();
                  }}
                  className={`group relative flex items-center justify-between px-3 py-2.5 rounded-lg text-xs cursor-pointer transition-all ${
                    isActive
                      ? 'bg-zinc-900 text-cyan-400 border border-zinc-800/80 font-medium'
                      : 'text-zinc-400 hover:text-zinc-200 hover:bg-zinc-900/50'
                  }`}
                >
                  <div className="flex items-center gap-2.5 truncate pr-6">
                    <svg
                      className={`w-3.5 h-3.5 shrink-0 ${isActive ? 'text-cyan-400' : 'text-zinc-400'}`}
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z"
                      />
                    </svg>
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
                    <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                      />
                    </svg>
                  </button>
                </div>
              );
            })
          )}
        </div>

        {/* Bottom Model Card */}
        <div className="p-3 border-t border-zinc-800/60 bg-zinc-950/80">
          <div className="p-3 rounded-xl bg-zinc-900/70 border border-zinc-800/60 text-xs">
            <div className="flex items-center justify-between text-zinc-200 font-medium mb-1">
              <span>MyGPT Architecture</span>
              <span className="text-[10px] text-cyan-400 font-mono">0.81M</span>
            </div>
            <div className="grid grid-cols-2 gap-1 text-[11px] text-zinc-400 font-mono mt-2 pt-2 border-t border-zinc-800/40">
              <div>Layers: {modelInfo?.n_layer || 4}</div>
              <div>Heads: {modelInfo?.n_head || 4}</div>
              <div>Embed: {modelInfo?.d_model || 128}</div>
              <div>Context: {modelInfo?.max_seq_len || 64}</div>
            </div>
          </div>

          {sessions.length > 0 && (
            <button
              onClick={onClearAll}
              className="mt-2 w-full text-center text-[11px] text-zinc-400 hover:text-rose-400 transition-colors py-1"
            >
              Clear Chat History
            </button>
          )}
        </div>
      </aside>
    </>
  );
};
