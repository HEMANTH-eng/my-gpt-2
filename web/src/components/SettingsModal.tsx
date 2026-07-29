'use client';

import React, { useState } from 'react';
import { AppearanceSettings, ModelConfig } from '../types';

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  config: ModelConfig;
  onChangeConfig: (newConfig: ModelConfig) => void;
}

export const SettingsModal: React.FC<SettingsModalProps> = ({
  isOpen,
  onClose,
  config,
  onChangeConfig,
}) => {
  if (!isOpen) return null;

  const [activeTab, setActiveTab] = useState<'appearance' | 'ai' | 'voice' | 'security' | 'apiKeys'>('ai');

  const [appearance, setAppearance] = useState<AppearanceSettings>({
    theme: 'dark',
    accentColor: '#4F46E5',
    fontSize: 'md',
    compactMode: false,
    sidebarWidth: 280,
    chatDensity: 'comfortable',
    animations: true,
  });

  const [apiKeys, setApiKeys] = useState([
    { id: '1', name: 'Development Key', prefix: 'nx_live_9a', created_at: '2026-07-29' },
    { id: '2', name: 'Production Agent Key', prefix: 'nx_live_4f', created_at: '2026-07-25' },
  ]);

  const [newKeyName, setNewKeyName] = useState('');

  const handleCreateKey = () => {
    if (!newKeyName.trim()) return;
    const newK = {
      id: Date.now().toString(),
      name: newKeyName,
      prefix: 'nx_live_' + Math.random().toString(36).substring(2, 6),
      created_at: new Date().toISOString().split('T')[0],
    };
    setApiKeys([...apiKeys, newK]);
    setNewKeyName('');
  };

  const handleRevokeKey = (id: string) => {
    setApiKeys(apiKeys.filter((k) => k.id !== id));
  };

  return (
    <div className="fixed inset-0 bg-black/70 backdrop-blur-md z-50 flex items-center justify-center p-4">
      <div className="glass-modal rounded-2xl max-w-3xl w-full max-h-[90vh] overflow-hidden border border-zinc-800 shadow-2xl flex flex-col sm:flex-row text-left">
        {/* Settings Navigation Sidebar */}
        <div className="w-full sm:w-56 bg-zinc-950/80 border-b sm:border-b-0 sm:border-r border-zinc-800/80 p-4 space-y-1">
          <h3 className="text-xs font-bold uppercase tracking-wider text-zinc-400 mb-3 px-2">Settings</h3>
          {[
            { id: 'ai', label: 'AI Model & Controls', icon: '🧠' },
            { id: 'appearance', label: 'Appearance & UI', icon: '🎨' },
            { id: 'voice', label: 'Voice & Audio', icon: '🎙️' },
            { id: 'security', label: 'Security & Auth', icon: '🔒' },
            { id: 'apiKeys', label: 'API Keys & Developer', icon: '🔑' },
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              className={`w-full flex items-center gap-2.5 px-3 py-2 rounded-xl text-xs font-medium transition-all ${
                activeTab === tab.id
                  ? 'bg-gradient-to-r from-[#4F46E5]/30 to-[#7C3AED]/30 text-[#06B6D4] border border-[#06B6D4]/30'
                  : 'text-zinc-400 hover:text-zinc-200 hover:bg-zinc-900/50'
              }`}
            >
              <span>{tab.icon}</span>
              <span>{tab.label}</span>
            </button>
          ))}
        </div>

        {/* Content Panel */}
        <div className="flex-1 p-6 overflow-y-auto max-h-[75vh] space-y-6">
          {/* AI Settings Tab */}
          {activeTab === 'ai' && (
            <div className="space-y-4 text-xs">
              <h4 className="text-sm font-bold text-zinc-100 border-b border-zinc-800 pb-2">
                AI Hyperparameters & Reasoning Modes
              </h4>

              <div>
                <div className="flex justify-between mb-1">
                  <label className="text-zinc-300">Temperature ({config.temperature})</label>
                  <span className="text-zinc-500">Creative vs Deterministic</span>
                </div>
                <input
                  type="range"
                  min="0.0"
                  max="2.0"
                  step="0.05"
                  value={config.temperature}
                  onChange={(e) => onChangeConfig({ ...config, temperature: parseFloat(e.target.value) })}
                  className="w-full accent-[#4F46E5]"
                />
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-zinc-300 mb-1">Top-K Truncation ({config.top_k})</label>
                  <input
                    type="number"
                    value={config.top_k}
                    onChange={(e) => onChangeConfig({ ...config, top_k: parseInt(e.target.value) || 40 })}
                    className="w-full bg-zinc-900 border border-zinc-800 rounded-xl px-3 py-2 text-zinc-100 focus:outline-none"
                  />
                </div>
                <div>
                  <label className="block text-zinc-300 mb-1">Top-P Nucleus ({config.top_p})</label>
                  <input
                    type="number"
                    step="0.05"
                    value={config.top_p}
                    onChange={(e) => onChangeConfig({ ...config, top_p: parseFloat(e.target.value) || 0.95 })}
                    className="w-full bg-zinc-900 border border-zinc-800 rounded-xl px-3 py-2 text-zinc-100 focus:outline-none"
                  />
                </div>
              </div>

              <div>
                <label className="block text-zinc-300 mb-1">Max Generation Tokens ({config.max_new_tokens})</label>
                <input
                  type="number"
                  value={config.max_new_tokens}
                  onChange={(e) => onChangeConfig({ ...config, max_new_tokens: parseInt(e.target.value) || 100 })}
                  className="w-full bg-zinc-900 border border-zinc-800 rounded-xl px-3 py-2 text-zinc-100 focus:outline-none"
                />
              </div>

              <div className="pt-2 border-t border-zinc-800 space-y-3">
                <label className="flex items-center justify-between text-zinc-300 cursor-pointer">
                  <span>Deterministic Greedy Decoding</span>
                  <input
                    type="checkbox"
                    checked={config.greedy}
                    onChange={(e) => onChangeConfig({ ...config, greedy: e.target.checked })}
                    className="accent-[#4F46E5] w-4 h-4 rounded"
                  />
                </label>

                <label className="flex items-center justify-between text-zinc-300 cursor-pointer">
                  <span>Long-Term Memory Engine</span>
                  <input
                    type="checkbox"
                    checked={config.memory_enabled ?? true}
                    onChange={(e) => onChangeConfig({ ...config, memory_enabled: e.target.checked })}
                    className="accent-[#4F46E5] w-4 h-4 rounded"
                  />
                </label>

                <label className="flex items-center justify-between text-zinc-300 cursor-pointer">
                  <span>Real-Time Streaming Response</span>
                  <input
                    type="checkbox"
                    checked={config.streaming ?? true}
                    onChange={(e) => onChangeConfig({ ...config, streaming: e.target.checked })}
                    className="accent-[#4F46E5] w-4 h-4 rounded"
                  />
                </label>
              </div>
            </div>
          )}

          {/* Appearance Settings Tab */}
          {activeTab === 'appearance' && (
            <div className="space-y-4 text-xs">
              <h4 className="text-sm font-bold text-zinc-100 border-b border-zinc-800 pb-2">
                Appearance & Customization
              </h4>

              <div>
                <label className="block text-zinc-300 mb-2">Theme Mode</label>
                <div className="grid grid-cols-3 gap-2">
                  {(['dark', 'light', 'system'] as const).map((t) => (
                    <button
                      key={t}
                      onClick={() => setAppearance({ ...appearance, theme: t })}
                      className={`py-2 rounded-xl border capitalize font-medium ${
                        appearance.theme === t
                          ? 'bg-[#4F46E5]/20 border-[#06B6D4] text-[#06B6D4]'
                          : 'bg-zinc-900 border-zinc-800 text-zinc-400'
                      }`}
                    >
                      {t}
                    </button>
                  ))}
                </div>
              </div>

              <div>
                <label className="block text-zinc-300 mb-2">Accent Palette</label>
                <div className="flex gap-3">
                  {['#4F46E5', '#7C3AED', '#06B6D4', '#10B981', '#F59E0B'].map((c) => (
                    <button
                      key={c}
                      onClick={() => setAppearance({ ...appearance, accentColor: c })}
                      className="w-7 h-7 rounded-full border-2 transition-transform hover:scale-110"
                      style={{ backgroundColor: c, borderColor: appearance.accentColor === c ? '#ffffff' : 'transparent' }}
                    />
                  ))}
                </div>
              </div>

              <div className="pt-2 border-t border-zinc-800 space-y-3">
                <label className="flex items-center justify-between text-zinc-300 cursor-pointer">
                  <span>UI Animations & Micro-transitions</span>
                  <input
                    type="checkbox"
                    checked={appearance.animations}
                    onChange={(e) => setAppearance({ ...appearance, animations: e.target.checked })}
                    className="accent-[#4F46E5] w-4 h-4 rounded"
                  />
                </label>

                <label className="flex items-center justify-between text-zinc-300 cursor-pointer">
                  <span>Compact Chat Layout</span>
                  <input
                    type="checkbox"
                    checked={appearance.compactMode}
                    onChange={(e) => setAppearance({ ...appearance, compactMode: e.target.checked })}
                    className="accent-[#4F46E5] w-4 h-4 rounded"
                  />
                </label>
              </div>
            </div>
          )}

          {/* Voice Tab */}
          {activeTab === 'voice' && (
            <div className="space-y-4 text-xs">
              <h4 className="text-sm font-bold text-zinc-100 border-b border-zinc-800 pb-2">
                Speech-to-Text & Voice Synthesis
              </h4>
              <p className="text-zinc-400">Configure Web Speech API microphone input and text-to-speech audio playback.</p>
              <div>
                <label className="block text-zinc-300 mb-1">Speech Recognition Engine</label>
                <select className="w-full bg-zinc-900 border border-zinc-800 rounded-xl px-3 py-2 text-zinc-100 focus:outline-none">
                  <option>Web Speech API (Native browser)</option>
                  <option>Whisper Speech Recognizer (Backend)</option>
                </select>
              </div>
              <div>
                <label className="block text-zinc-300 mb-1">Text-To-Speech Playback Voice</label>
                <select className="w-full bg-zinc-900 border border-zinc-800 rounded-xl px-3 py-2 text-zinc-100 focus:outline-none">
                  <option>Novexa AI Voice (Natural Assistant)</option>
                  <option>Standard Neural Synthetic Voice</option>
                </select>
              </div>
            </div>
          )}

          {/* Security Tab */}
          {activeTab === 'security' && (
            <div className="space-y-4 text-xs">
              <h4 className="text-sm font-bold text-zinc-100 border-b border-zinc-800 pb-2">
                Security & Authentication
              </h4>
              <div className="p-3 rounded-xl bg-emerald-950/30 border border-emerald-800/50 text-emerald-300">
                ✓ AES-256 Payload Encryption & Threat Sanitizer Enabled
              </div>
              <button className="px-4 py-2 rounded-xl bg-zinc-900 hover:bg-zinc-800 border border-zinc-800 text-zinc-200">
                Enable Two-Factor Authentication (2FA)
              </button>
            </div>
          )}

          {/* API Keys Tab */}
          {activeTab === 'apiKeys' && (
            <div className="space-y-4 text-xs">
              <h4 className="text-sm font-bold text-zinc-100 border-b border-zinc-800 pb-2">
                API Keys & Developer Tokens
              </h4>

              <div className="flex gap-2">
                <input
                  type="text"
                  placeholder="Key description (e.g. My App Key)"
                  value={newKeyName}
                  onChange={(e) => setNewKeyName(e.target.value)}
                  className="flex-1 bg-zinc-900 border border-zinc-800 rounded-xl px-3 py-2 text-zinc-100 focus:outline-none"
                />
                <button
                  onClick={handleCreateKey}
                  className="px-4 py-2 rounded-xl glow-button text-white font-semibold"
                >
                  + Create
                </button>
              </div>

              <div className="space-y-2">
                {apiKeys.map((k) => (
                  <div key={k.id} className="p-3 rounded-xl bg-zinc-900 border border-zinc-800 flex items-center justify-between font-mono">
                    <div>
                      <p className="text-zinc-200 font-sans font-bold">{k.name}</p>
                      <p className="text-zinc-400 text-[10px]">{k.prefix}••••••••••••• (Created {k.created_at})</p>
                    </div>
                    <button
                      onClick={() => handleRevokeKey(k.id)}
                      className="text-rose-400 hover:text-rose-300 text-[11px]"
                    >
                      Revoke
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Modal Footer Close */}
          <div className="pt-4 border-t border-zinc-800 flex justify-end">
            <button
              onClick={onClose}
              className="px-5 py-2 rounded-xl glow-button text-white text-xs font-semibold"
            >
              Done & Close
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
