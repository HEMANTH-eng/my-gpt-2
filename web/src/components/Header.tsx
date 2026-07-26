'use client';

import React from 'react';
import { HealthStatus } from '../types';

interface HeaderProps {
  health: HealthStatus | null;
  onOpenConfig: () => void;
  onNewChat: () => void;
}

export const Header: React.FC<HeaderProps> = ({ health, onOpenConfig, onNewChat }) => {
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
              v1.0
            </span>
          </h1>
          <p className="text-[11px] text-zinc-400 font-mono">PyTorch Language Model</p>
        </div>
      </div>

      {/* Controls & Status */}
      <div className="flex items-center gap-3">
        {/* Status Badge */}
        <div
          className={`flex items-center gap-2 text-xs font-medium px-3 py-1.5 rounded-full border ${
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

        {/* New Chat Button */}
        <button
          onClick={onNewChat}
          className="text-xs font-medium px-3 py-1.5 rounded-lg bg-zinc-900 hover:bg-zinc-800 text-zinc-200 border border-zinc-800 transition-colors flex items-center gap-1.5"
        >
          <svg className="w-3.5 h-3.5 text-zinc-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
          </svg>
          New Chat
        </button>

        {/* Config Modal Button */}
        <button
          onClick={onOpenConfig}
          className="p-2 rounded-lg bg-zinc-900 hover:bg-zinc-800 text-zinc-400 hover:text-zinc-200 border border-zinc-800 transition-colors"
          title="Model Hyperparameters"
        >
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"
            />
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
          </svg>
        </button>
      </div>
    </header>
  );
};
