'use client';

import React from 'react';
import { NotificationItem } from '../types';

interface NotificationDrawerProps {
  isOpen: boolean;
  onClose: () => void;
  notifications: NotificationItem[];
  onMarkAllAsRead: () => void;
}

export const NotificationDrawer: React.FC<NotificationDrawerProps> = ({
  isOpen,
  onClose,
  notifications,
  onMarkAllAsRead,
}) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex justify-end">
      <div className="w-full max-w-sm bg-zinc-950/95 border-l border-zinc-800 h-full flex flex-col p-4 shadow-2xl text-left">
        <div className="flex items-center justify-between border-b border-zinc-800 pb-3 mb-4">
          <div className="flex items-center gap-2">
            <span className="text-lg">🔔</span>
            <h3 className="font-bold text-zinc-100 text-sm">Notifications & Updates</h3>
          </div>
          <button onClick={onClose} className="text-zinc-400 hover:text-zinc-200 text-xs">✕</button>
        </div>

        <div className="flex justify-between items-center mb-3">
          <span className="text-[11px] text-zinc-400 font-mono">System Activity Log</span>
          {notifications.length > 0 && (
            <button
              onClick={onMarkAllAsRead}
              className="text-[11px] text-[#06B6D4] hover:underline"
            >
              Mark all as read
            </button>
          )}
        </div>

        <div className="flex-1 overflow-y-auto space-y-2">
          {notifications.length === 0 ? (
            <p className="text-xs text-zinc-500 text-center py-10 font-mono">No new notifications.</p>
          ) : (
            notifications.map((n) => (
              <div
                key={n.id}
                className={`p-3 rounded-xl border text-xs transition-all ${
                  n.read
                    ? 'bg-zinc-900/40 border-zinc-800/60 text-zinc-400'
                    : 'bg-zinc-900 border-[#06B6D4]/40 text-zinc-100 shadow-md'
                }`}
              >
                <div className="flex items-center justify-between font-bold mb-1">
                  <span className={n.type === 'success' ? 'text-emerald-400' : 'text-[#06B6D4]'}>{n.title}</span>
                  <span className="text-[9px] text-zinc-500 font-mono">{n.timestamp}</span>
                </div>
                <p className="text-[11px] leading-relaxed">{n.message}</p>
              </div>
            ))
          )}
        </div>

        <div className="pt-3 border-t border-zinc-800 text-center text-[10px] text-zinc-500 font-mono">
          Novexa AI Platform v2.0
        </div>
      </div>
    </div>
  );
};
