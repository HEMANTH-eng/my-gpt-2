'use client';

import React, { useState } from 'react';
import { UploadedFile } from '../types';

interface FileManagerModalProps {
  isOpen: boolean;
  onClose: () => void;
  files: UploadedFile[];
  onUploadFile: (file: File) => Promise<void>;
  onDeleteFile: (id: string) => void;
}

export const FileManagerModal: React.FC<FileManagerModalProps> = ({
  isOpen,
  onClose,
  files,
  onUploadFile,
  onDeleteFile,
}) => {
  if (!isOpen) return null;

  const [isUploading, setIsUploading] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setIsUploading(true);
      await onUploadFile(e.target.files[0]);
      setIsUploading(false);
    }
  };

  const filteredFiles = files.filter(
    (f) =>
      f.filename.toLowerCase().includes(searchQuery.toLowerCase()) ||
      f.file_type.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div className="fixed inset-0 bg-black/70 backdrop-blur-md z-50 flex items-center justify-center p-4">
      <div className="glass-modal rounded-2xl max-w-3xl w-full max-h-[90vh] overflow-y-auto border border-zinc-800 shadow-2xl p-6 text-left">
        <div className="flex items-center justify-between border-b border-zinc-800 pb-3 mb-4">
          <div className="flex items-center gap-2">
            <span className="text-xl">📁</span>
            <div>
              <h2 className="text-base font-bold text-zinc-100">Document RAG & Media File Manager</h2>
              <p className="text-xs text-zinc-400 font-mono">Upload PDF, DOCX, TXT, and Image files for AI contextual grounding</p>
            </div>
          </div>
          <button onClick={onClose} className="text-zinc-400 hover:text-zinc-200 text-xs">✕</button>
        </div>

        {/* File Drag & Drop Upload Bar */}
        <div className="border-2 border-dashed border-zinc-800 hover:border-[#4F46E5] rounded-2xl p-6 text-center bg-zinc-950/60 transition-colors mb-6 cursor-pointer relative">
          <input
            type="file"
            onChange={handleFileChange}
            className="absolute inset-0 opacity-0 cursor-pointer"
          />
          <div className="text-3xl mb-2">📥</div>
          <p className="text-xs font-semibold text-zinc-200">
            {isUploading ? 'Ingesting document context...' : 'Click or Drag & Drop PDF, DOCX, or TXT files here'}
          </p>
          <p className="text-[10px] text-zinc-500 font-mono mt-1">Supports RAG contextual extraction up to 25 MB</p>
        </div>

        {/* Search */}
        <div className="mb-4">
          <input
            type="text"
            placeholder="Search ingested files..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full bg-zinc-900 border border-zinc-800 rounded-xl px-3 py-2 text-xs text-zinc-100 focus:outline-none"
          />
        </div>

        {/* Files Grid */}
        <div className="space-y-2">
          <h3 className="text-xs font-bold uppercase tracking-wider text-zinc-400 mb-2">
            Ingested Documents ({filteredFiles.length})
          </h3>
          {filteredFiles.length === 0 ? (
            <p className="text-xs text-zinc-500 text-center py-8 font-mono">No files uploaded yet.</p>
          ) : (
            filteredFiles.map((f) => (
              <div key={f.id} className="p-3 rounded-xl bg-zinc-900/70 border border-zinc-800 flex items-center justify-between text-xs">
                <div className="flex items-center gap-3">
                  <span className="text-lg">📄</span>
                  <div>
                    <p className="font-bold text-zinc-100">{f.filename}</p>
                    <p className="text-[10px] text-zinc-400 font-mono">
                      {f.file_type.toUpperCase()} • {(f.file_size / 1024).toFixed(1)} KB
                    </p>
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  <span className="text-[10px] px-2 py-0.5 rounded bg-emerald-950 text-emerald-400 border border-emerald-800">
                    RAG Ready
                  </span>
                  <button
                    onClick={() => onDeleteFile(f.id)}
                    className="text-zinc-500 hover:text-rose-400"
                  >
                    Delete
                  </button>
                </div>
              </div>
            ))
          )}
        </div>

        <div className="mt-6 pt-3 border-t border-zinc-800 flex justify-end">
          <button onClick={onClose} className="px-4 py-2 rounded-xl bg-zinc-900 text-zinc-300 text-xs">
            Done
          </button>
        </div>
      </div>
    </div>
  );
};
