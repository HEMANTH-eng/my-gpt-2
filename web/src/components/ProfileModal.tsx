'use client';

import React, { useState } from 'react';
import { User } from '../types';

interface ProfileModalProps {
  isOpen: boolean;
  onClose: () => void;
  currentUser: User | null;
  onUpdateUser: (user: User) => void;
}

export const ProfileModal: React.FC<ProfileModalProps> = ({
  isOpen,
  onClose,
  currentUser,
  onUpdateUser,
}) => {
  if (!isOpen) return null;

  const [displayName, setDisplayName] = useState(currentUser?.displayName || currentUser?.username || 'AI Developer');
  const [bio, setBio] = useState(currentUser?.bio || 'Building custom PyTorch AI models & autonomous workflows.');
  const [website, setWebsite] = useState(currentUser?.website || 'https://novexa.ai');
  const [location, setLocation] = useState(currentUser?.location || 'San Francisco, CA');

  const handleSave = () => {
    if (currentUser) {
      onUpdateUser({
        ...currentUser,
        displayName,
        bio,
        website,
        location,
      });
    }
    onClose();
  };

  return (
    <div className="fixed inset-0 bg-black/70 backdrop-blur-md z-50 flex items-center justify-center p-4">
      <div className="glass-modal rounded-2xl max-w-2xl w-full max-h-[90vh] overflow-y-auto border border-zinc-800 shadow-2xl p-6 text-left">
        {/* Banner & Avatar */}
        <div className="relative mb-16">
          <div className="h-28 rounded-xl bg-gradient-to-r from-[#4F46E5] via-[#7C3AED] to-[#06B6D4] opacity-80" />
          <div className="absolute -bottom-10 left-6 w-20 h-20 rounded-2xl bg-zinc-950 border-2 border-[#06B6D4] flex items-center justify-center font-extrabold text-2xl text-[#06B6D4] shadow-xl">
            {currentUser?.username.charAt(0).toUpperCase() || 'N'}
          </div>
        </div>

        {/* User Info Header */}
        <div className="flex justify-between items-start mb-6">
          <div>
            <h2 className="text-xl font-bold text-zinc-100">{displayName}</h2>
            <p className="text-xs text-zinc-400 font-mono">@{currentUser?.username || 'user'} • {currentUser?.email}</p>
          </div>
          <span className="px-3 py-1 rounded-full bg-[#4F46E5]/20 text-[#06B6D4] border border-[#06B6D4]/30 text-xs font-semibold">
            {currentUser?.plan?.toUpperCase() || 'PRO WORKSPACE'}
          </span>
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-6 font-mono text-xs">
          <div className="p-3 rounded-xl bg-zinc-900/80 border border-zinc-800">
            <p className="text-zinc-400 text-[10px]">Tokens Used</p>
            <p className="text-sm font-bold text-[#06B6D4] mt-1">142.5K</p>
          </div>
          <div className="p-3 rounded-xl bg-zinc-900/80 border border-zinc-800">
            <p className="text-zinc-400 text-[10px]">Storage Used</p>
            <p className="text-sm font-bold text-emerald-400 mt-1">24.8 MB</p>
          </div>
          <div className="p-3 rounded-xl bg-zinc-900/80 border border-zinc-800">
            <p className="text-zinc-400 text-[10px]">API Calls</p>
            <p className="text-sm font-bold text-amber-400 mt-1">1,840</p>
          </div>
          <div className="p-3 rounded-xl bg-zinc-900/80 border border-zinc-800">
            <p className="text-zinc-400 text-[10px]">Security Status</p>
            <p className="text-sm font-bold text-purple-400 mt-1">2FA Verified</p>
          </div>
        </div>

        {/* Editable Fields */}
        <div className="space-y-4 text-xs">
          <div>
            <label className="block text-zinc-400 mb-1">Display Name</label>
            <input
              type="text"
              value={displayName}
              onChange={(e) => setDisplayName(e.target.value)}
              className="w-full bg-zinc-900 border border-zinc-800 rounded-xl px-3 py-2 text-zinc-100 focus:outline-none focus:border-[#4F46E5]"
            />
          </div>

          <div>
            <label className="block text-zinc-400 mb-1">Bio</label>
            <textarea
              value={bio}
              onChange={(e) => setBio(e.target.value)}
              rows={2}
              className="w-full bg-zinc-900 border border-zinc-800 rounded-xl px-3 py-2 text-zinc-100 focus:outline-none focus:border-[#4F46E5] resize-none"
            />
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <label className="block text-zinc-400 mb-1">Website</label>
              <input
                type="text"
                value={website}
                onChange={(e) => setWebsite(e.target.value)}
                className="w-full bg-zinc-900 border border-zinc-800 rounded-xl px-3 py-2 text-zinc-100 focus:outline-none focus:border-[#4F46E5]"
              />
            </div>
            <div>
              <label className="block text-zinc-400 mb-1">Location</label>
              <input
                type="text"
                value={location}
                onChange={(e) => setLocation(e.target.value)}
                className="w-full bg-zinc-900 border border-zinc-800 rounded-xl px-3 py-2 text-zinc-100 focus:outline-none focus:border-[#4F46E5]"
              />
            </div>
          </div>

          {/* Connected Accounts */}
          <div>
            <label className="block text-zinc-400 mb-2">Connected Accounts</label>
            <div className="flex flex-wrap gap-2">
              {['Google', 'GitHub', 'Microsoft', 'Discord', 'Apple'].map((acc, idx) => (
                <span
                  key={idx}
                  className="px-3 py-1.5 rounded-lg bg-zinc-900 border border-zinc-800 text-zinc-300 flex items-center gap-1.5 text-[11px]"
                >
                  <span className="text-emerald-400">✓</span> {acc}
                </span>
              ))}
            </div>
          </div>
        </div>

        {/* Buttons */}
        <div className="mt-6 pt-4 border-t border-zinc-800 flex justify-end gap-3">
          <button
            onClick={onClose}
            className="px-4 py-2 rounded-xl bg-zinc-900 hover:bg-zinc-800 text-zinc-300 text-xs transition-all"
          >
            Cancel
          </button>
          <button
            onClick={handleSave}
            className="px-5 py-2 rounded-xl glow-button text-white text-xs font-semibold"
          >
            Save Profile
          </button>
        </div>
      </div>
    </div>
  );
};
