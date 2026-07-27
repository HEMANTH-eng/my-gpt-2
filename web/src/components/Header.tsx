'use client';

import React from 'react';
import { HealthStatus, Persona, User } from '../types';

interface HeaderProps {
  health: HealthStatus | null;
  personas: Persona[];
  selectedPersonaId: string;
  onSelectPersona: (id: string) => void;
  currentUser: User | null;
  onOpenAuth: () => void;
  onOpenConfig: () => void;
  onNewChat: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  health,
  personas,
  selectedPersonaId,
  onSelectPersona,
  currentUser,
  onOpenAuth,
  onOpenConfig,
  onNewChat,
}) => {
  const isOnline = health?.status === 'ok';

  return (
    <header className="h-16 border-b border-zinc-800/80 bg-zinc-950/70 backdrop-blur-md px-4 flex items-center justify-between sticky top-0 z-20">
      {/* Brand & Logo */}
      <div className="flex items-center gap-3">
        <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-cyan-500 via-indigo-500 to-purple-600 p-[1px] shadow-lg shadow-cyan-500/20">
          <div className="w-full h-full bg-zinc-950 rounded-[11px] flex items-center justify-center font-bold text-sm text-cyan-400">
            GPT
          </div>
        </div>
        <div>
          <h1 className="font-semibold text-zinc-100 text-sm tracking-wide flex items-center gap-2">
            MyGPT Studio
            <span className="text-[10px] font-medium px-2 py-0.5 rounded-full bg-cyan-950/80 text-cyan-400 border border-cyan-800/60">
              v2.0 Multimodal
            </span>
          </h1>
          <p className="text-[11px] text-zinc-400 font-mono">Enterprise Platform</p>
        </div>
      </div>

      {/* Center Persona Selector */}
      {personas.length > 0 && (
        <div className="hidden sm:flex items-center gap-2 bg-zinc-900/90 border border-zinc-800 rounded-xl px-3 py-1 text-xs">
          <span className="text-zinc-400 text-[11px]">Persona:</span>
          <select
            value={selectedPersonaId}
            onChange={(e) => onSelectPersona(e.target.value)}
            className="bg-transparent text-zinc-200 focus:outline-none cursor-pointer font-medium"
          >
            {personas.map((p) => (
              <option key={p.id} value={p.id} className="bg-zinc-900 text-zinc-200">
                {p.icon} {p.name}
              </option>
            ))}
          </select>
        </div>
      )}

      {/* Right Controls & Status */}
      <div className="flex items-center gap-3">
        {/* Status Badge */}
        <div
          className={`hidden md:flex items-center gap-2 text-xs font-medium px-3 py-1.5 rounded-full border ${
            isOnline
              ? 'bg-emerald-950/40 text-emerald-400 border-emerald-800/50'
              : 'bg-amber-950/40 text-amber-400 border-amber-800/50'
          }`}
        >
          <span
            className={`w-2 h-2 rounded-full animate-pulse ${
              isOnline ? 'bg-emerald-400 shadow-sm shadow-emerald-400' : 'bg-amber-400 shadow-sm shadow-amber-400'
            }`}
          />
          {isOnline ? `API Online (${health?.device})` : 'Offline Mode'}
        </div>

        {/* User Account / Auth Button */}
        <button
          onClick={onOpenAuth}
          className="text-xs font-medium px-3 py-1.5 rounded-lg bg-zinc-900 hover:bg-zinc-800 text-cyan-400 border border-zinc-800 transition-colors flex items-center gap-1.5"
        >
          <span className="w-2 h-2 rounded-full bg-cyan-400" />
          {currentUser ? currentUser.username : 'Sign In'}
        </button>

        {/* New Chat Button */}
        <button
          onClick={onNewChat}
          className="text-xs font-medium px-3 py-1.5 rounded-lg bg-gradient-to-r from-cyan-600 to-indigo-600 hover:from-cyan-500 hover:to-indigo-500 text-white transition-all flex items-center gap-1.5 shadow-md shadow-cyan-950"
        >
          + New Chat
        </button>

        {/* Config Modal Button */}
        <button
          onClick={onOpenConfig}
          className="p-2 rounded-lg bg-zinc-900 hover:bg-zinc-800 text-zinc-400 hover:text-zinc-200 border border-zinc-800 transition-colors"
          title="Model Controls"
        >
          ⚙️
        </button>
      </div>
    </header>
  );
};
