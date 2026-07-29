'use client';

import React, { useState } from 'react';
import { MemoryItem } from '../types';

interface MemoryModalProps {
  isOpen: boolean;
  onClose: () => void;
  memories: MemoryItem[];
  onAddMemory: (key: string, value: string, category: 'preference' | 'fact' | 'instruction') => void;
  onDeleteMemory: (id: string) => void;
}

export const MemoryModal: React.FC<MemoryModalProps> = ({
  isOpen,
  onClose,
  memories,
  onAddMemory,
  onDeleteMemory,
}) => {
  if (!isOpen) return null;

  const [key, setKey] = useState('');
  const [value, setValue] = useState('');
  const [category, setCategory] = useState<'preference' | 'fact' | 'instruction'>('fact');

  const handleAdd = () => {
    if (!key.trim() || !value.trim()) return;
    onAddMemory(key, value, category);
    setKey('');
    setValue('');
  };

  return (
    <div className="fixed inset-0 bg-black/70 backdrop-blur-md z-50 flex items-center justify-center p-4">
      <div className="glass-modal rounded-2xl max-w-2xl w-full max-h-[90vh] overflow-y-auto border border-zinc-800 shadow-2xl p-6 text-left">
        <div className="flex items-center justify-between border-b border-zinc-800 pb-3 mb-4">
          <div className="flex items-center gap-2">
            <span className="text-xl">🧠</span>
            <div>
              <h2 className="text-base font-bold text-zinc-100">Long-Term AI Memory Manager</h2>
              <p className="text-xs text-zinc-400 font-mono">Facts, preferences, and guidelines remembered by Novexa AI</p>
            </div>
          </div>
          <button onClick={onClose} className="text-zinc-400 hover:text-zinc-200 text-xs">✕</button>
        </div>

        {/* Add Memory Inputs */}
        <div className="bg-zinc-900/80 rounded-xl p-4 border border-zinc-800 space-y-3 mb-6">
          <h3 className="text-xs font-semibold text-zinc-200">Add New Memory Entry</h3>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 text-xs">
            <input
              type="text"
              placeholder="Topic / Key (e.g. Favorite Language)"
              value={key}
              onChange={(e) => setKey(e.target.value)}
              className="bg-zinc-950 border border-zinc-800 rounded-lg px-3 py-2 text-zinc-100 focus:outline-none"
            />
            <input
              type="text"
              placeholder="Fact Value (e.g. Python & TypeScript)"
              value={value}
              onChange={(e) => setValue(e.target.value)}
              className="bg-zinc-950 border border-zinc-800 rounded-lg px-3 py-2 text-zinc-100 focus:outline-none"
            />
            <select
              value={category}
              onChange={(e) => setCategory(e.target.value as any)}
              className="bg-zinc-950 border border-zinc-800 rounded-lg px-3 py-2 text-zinc-100 focus:outline-none"
            >
              <option value="fact">Fact</option>
              <option value="preference">Preference</option>
              <option value="instruction">Instruction</option>
            </select>
          </div>
          <button
            onClick={handleAdd}
            className="px-4 py-1.5 rounded-lg glow-button text-white text-xs font-semibold"
          >
            + Remember Fact
          </button>
        </div>

        {/* Memory List */}
        <div className="space-y-2">
          <h3 className="text-xs font-bold uppercase tracking-wider text-zinc-400 mb-2">
            Remembered Timeline ({memories.length})
          </h3>
          {memories.length === 0 ? (
            <p className="text-xs text-zinc-500 text-center py-6 font-mono">No remembered items yet. Add one above!</p>
          ) : (
            memories.map((m) => (
              <div key={m.id} className="p-3 rounded-xl bg-zinc-900/60 border border-zinc-800 flex items-center justify-between text-xs">
                <div>
                  <span className="font-bold text-[#06B6D4] mr-2">[{m.category.toUpperCase()}]</span>
                  <span className="font-semibold text-zinc-200">{m.key}:</span>{' '}
                  <span className="text-zinc-300">{m.value}</span>
                </div>
                <button
                  onClick={() => onDeleteMemory(m.id)}
                  className="text-zinc-500 hover:text-rose-400 text-xs ml-4"
                  title="Forget Memory"
                >
                  ✕
                </button>
              </div>
            ))
          )}
        </div>

        <div className="mt-6 pt-3 border-t border-zinc-800 flex justify-end">
          <button onClick={onClose} className="px-4 py-2 rounded-xl bg-zinc-900 text-zinc-300 text-xs">
            Close
          </button>
        </div>
      </div>
    </div>
  );
};
