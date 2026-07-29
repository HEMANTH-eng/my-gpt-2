'use client';

import React from 'react';

interface AdminModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const AdminModal: React.FC<AdminModalProps> = ({ isOpen, onClose }) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/70 backdrop-blur-md z-50 flex items-center justify-center p-4">
      <div className="glass-modal rounded-2xl max-w-4xl w-full max-h-[90vh] overflow-y-auto border border-zinc-800 shadow-2xl p-6 text-left">
        <div className="flex items-center justify-between border-b border-zinc-800 pb-3 mb-6">
          <div className="flex items-center gap-2">
            <span className="text-xl">🛡️</span>
            <div>
              <h2 className="text-lg font-bold text-zinc-100">Novexa AI Platform Admin Control Panel</h2>
              <p className="text-xs text-zinc-400 font-mono">User governance, active sessions, moderation, and feature flags</p>
            </div>
          </div>
          <button onClick={onClose} className="text-zinc-400 hover:text-zinc-200 text-xs">✕</button>
        </div>

        {/* System Stats */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-6">
          <div className="p-3 rounded-xl bg-zinc-900 border border-zinc-800 font-mono">
            <p className="text-[10px] text-zinc-400">Total Users</p>
            <p className="text-lg font-bold text-white mt-1">4,120</p>
          </div>
          <div className="p-3 rounded-xl bg-zinc-900 border border-zinc-800 font-mono">
            <p className="text-[10px] text-zinc-400">Active Subscriptions</p>
            <p className="text-lg font-bold text-[#06B6D4] mt-1">1,845</p>
          </div>
          <div className="p-3 rounded-xl bg-zinc-900 border border-zinc-800 font-mono">
            <p className="text-[10px] text-zinc-400">Security Events</p>
            <p className="text-lg font-bold text-emerald-400 mt-1">0 Threats</p>
          </div>
          <div className="p-3 rounded-xl bg-zinc-900 border border-zinc-800 font-mono">
            <p className="text-[10px] text-zinc-400">System Uptime</p>
            <p className="text-lg font-bold text-purple-400 mt-1">99.98%</p>
          </div>
        </div>

        {/* Registered Users Table */}
        <div className="glass-card rounded-xl p-4 border border-zinc-800 mb-6 text-xs">
          <h3 className="font-bold text-zinc-200 mb-3">Registered Platform Accounts</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-left font-mono">
              <thead>
                <tr className="border-b border-zinc-800 text-zinc-400 text-[10px] uppercase">
                  <th className="pb-2">ID</th>
                  <th className="pb-2">User</th>
                  <th className="pb-2">Email</th>
                  <th className="pb-2">Role</th>
                  <th className="pb-2">Plan</th>
                  <th className="pb-2">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-zinc-800/60">
                {[
                  { id: 1, user: 'admin', email: 'admin@novexa.ai', role: 'admin', plan: 'Enterprise' },
                  { id: 2, user: 'developer_user', email: 'dev@novexa.ai', role: 'user', plan: 'Pro' },
                  { id: 3, user: 'guest_user', email: 'guest@novexa.ai', role: 'guest', plan: 'Free' },
                ].map((u) => (
                  <tr key={u.id}>
                    <td className="py-2 text-zinc-400">{u.id}</td>
                    <td className="py-2 font-sans font-semibold text-zinc-100">{u.user}</td>
                    <td className="py-2 text-zinc-400">{u.email}</td>
                    <td className="py-2">
                      <span className="px-2 py-0.5 rounded bg-purple-950 text-purple-300 border border-purple-800 text-[10px]">
                        {u.role}
                      </span>
                    </td>
                    <td className="py-2 text-[#06B6D4]">{u.plan}</td>
                    <td className="py-2">
                      <button className="text-rose-400 hover:underline">Manage</button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div className="flex justify-end pt-3 border-t border-zinc-800">
          <button onClick={onClose} className="px-5 py-2 rounded-xl glow-button text-white text-xs font-semibold">
            Close Control Panel
          </button>
        </div>
      </div>
    </div>
  );
};
