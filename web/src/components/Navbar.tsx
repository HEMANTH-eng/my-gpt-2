'use client';

import React, { useState } from 'react';
import { Workspace } from '../types';

interface NavbarProps {
  activeView: 'landing' | 'studio';
  onSelectView: (view: 'landing' | 'studio') => void;
  workspaces: Workspace[];
  activeWorkspaceId: string;
  onSelectWorkspace: (id: string) => void;
  onOpenAuth: () => void;
  onOpenProfile: () => void;
  onOpenSettings: () => void;
  onOpenDashboard: () => void;
  onOpenNotifications: () => void;
  unreadNotificationsCount: number;
  currentUser: { username: string; email: string } | null;
}

export const Navbar: React.FC<NavbarProps> = ({
  activeView,
  onSelectView,
  workspaces,
  activeWorkspaceId,
  onSelectWorkspace,
  onOpenAuth,
  onOpenProfile,
  onOpenSettings,
  onOpenDashboard,
  onOpenNotifications,
  unreadNotificationsCount,
  currentUser,
}) => {
  const activeWorkspace = workspaces.find((w) => w.id === activeWorkspaceId) || workspaces[0];

  return (
    <header className="h-16 border-b border-zinc-800/80 bg-zinc-950/80 backdrop-blur-xl px-4 md:px-8 flex items-center justify-between sticky top-0 z-40 transition-all">
      {/* Brand & Logo */}
      <div className="flex items-center gap-4">
        <button
          onClick={() => onSelectView('landing')}
          className="flex items-center gap-3 group text-left focus:outline-none"
        >
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-[#4F46E5] via-[#7C3AED] to-[#06B6D4] p-[1px] shadow-lg shadow-[#4F46E5]/30 group-hover:scale-105 transition-transform">
            <div className="w-full h-full bg-zinc-950 rounded-[11px] flex items-center justify-center font-extrabold text-xs tracking-wider text-[#06B6D4]">
              NA
            </div>
          </div>
          <div>
            <div className="font-extrabold text-zinc-100 text-base tracking-tight flex items-center gap-2">
              Novexa AI
              <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full bg-[#06B6D4]/10 text-[#06B6D4] border border-[#06B6D4]/30">
                PRO
              </span>
            </div>
            <p className="text-[10px] text-zinc-400 font-mono tracking-wider">Build. Think. Create.</p>
          </div>
        </button>

        {/* View Switcher (Landing vs Studio) */}
        <div className="hidden sm:flex items-center bg-zinc-900/80 p-1 rounded-xl border border-zinc-800 text-xs ml-4">
          <button
            onClick={() => onSelectView('landing')}
            className={`px-3 py-1 rounded-lg font-medium transition-all ${
              activeView === 'landing'
                ? 'bg-zinc-800 text-zinc-100 shadow-sm'
                : 'text-zinc-400 hover:text-zinc-200'
            }`}
          >
            Overview
          </button>
          <button
            onClick={() => onSelectView('studio')}
            className={`px-3 py-1 rounded-lg font-medium transition-all ${
              activeView === 'studio'
                ? 'bg-gradient-to-r from-[#4F46E5] to-[#7C3AED] text-white shadow-md'
                : 'text-zinc-400 hover:text-zinc-200'
            }`}
          >
            AI Studio
          </button>
        </div>
      </div>

      {/* Workspace Switcher & Right Controls */}
      <div className="flex items-center gap-3">
        {/* Workspace Dropdown (when in Studio mode) */}
        {activeView === 'studio' && (
          <div className="hidden md:flex items-center gap-2 bg-zinc-900/90 border border-zinc-800/80 rounded-xl px-3 py-1.5 text-xs">
            <span className="text-[11px] text-zinc-400">Workspace:</span>
            <select
              value={activeWorkspaceId}
              onChange={(e) => onSelectWorkspace(e.target.value)}
              className="bg-transparent text-zinc-200 font-semibold focus:outline-none cursor-pointer"
            >
              {workspaces.map((w) => (
                <option key={w.id} value={w.id} className="bg-zinc-900 text-zinc-200">
                  {w.icon} {w.name}
                </option>
              ))}
            </select>
          </div>
        )}

        {/* Dashboard Button */}
        <button
          onClick={onOpenDashboard}
          className="p-2 rounded-xl bg-zinc-900/80 hover:bg-zinc-800 text-zinc-300 border border-zinc-800/80 text-xs font-medium transition-all flex items-center gap-1.5"
          title="System Dashboard"
        >
          📊 <span className="hidden lg:inline">Metrics</span>
        </button>

        {/* Notifications Button */}
        <button
          onClick={onOpenNotifications}
          className="relative p-2 rounded-xl bg-zinc-900/80 hover:bg-zinc-800 text-zinc-300 border border-zinc-800/80 text-xs font-medium transition-all"
          title="Notifications"
        >
          🔔
          {unreadNotificationsCount > 0 && (
            <span className="absolute -top-1 -right-1 w-4 h-4 rounded-full bg-rose-500 text-white text-[9px] font-bold flex items-center justify-center animate-pulse">
              {unreadNotificationsCount}
            </span>
          )}
        </button>

        {/* Settings Button */}
        <button
          onClick={onOpenSettings}
          className="p-2 rounded-xl bg-zinc-900/80 hover:bg-zinc-800 text-zinc-300 border border-zinc-800/80 text-xs font-medium transition-all"
          title="Preferences & Settings"
        >
          ⚙️
        </button>

        {/* Auth / Profile Button */}
        {currentUser ? (
          <button
            onClick={onOpenProfile}
            className="flex items-center gap-2 pl-2 pr-3 py-1 rounded-xl bg-zinc-900/90 hover:bg-zinc-800 border border-zinc-800 text-xs text-zinc-200 font-medium transition-all"
          >
            <div className="w-6 h-6 rounded-lg bg-gradient-to-tr from-[#4F46E5] to-[#06B6D4] flex items-center justify-center text-white font-bold text-[11px]">
              {currentUser.username.charAt(0).toUpperCase()}
            </div>
            <span className="hidden sm:inline">{currentUser.username}</span>
          </button>
        ) : (
          <button
            onClick={onOpenAuth}
            className="px-4 py-1.5 rounded-xl glow-button text-white text-xs font-semibold tracking-wide"
          >
            Sign In
          </button>
        )}
      </div>
    </header>
  );
};
