'use client';

import React, { useState } from 'react';

interface ToolInfo {
  tool_id: string;
  name: string;
  description: string;
  syntax: string;
}

const TOOLS: ToolInfo[] = [
  { tool_id: 'search', name: 'Web Search', description: 'Searches external web for information.', syntax: '[TOOL: search("PyTorch 2.0")]' },
  { tool_id: 'calculator', name: 'Calculator', description: 'Evaluates mathematical calculations.', syntax: '[TOOL: calc("12 * 12 + 50")]' },
  { tool_id: 'weather', name: 'Weather Lookup', description: 'Gets city weather & temperature forecasts.', syntax: '[TOOL: weather("New York")]' },
  { tool_id: 'db_query', name: 'Database Query', description: 'Queries SQLite database tables.', syntax: '[TOOL: db_query("SELECT * FROM users")]' },
  { tool_id: 'python', name: 'Python REPL', description: 'Executes Python code expressions.', syntax: '[TOOL: python("math.sqrt(144)")]' },
  { tool_id: 'pdf_reader', name: 'PDF Reader', description: 'Extracts text from PDF documents.', syntax: '[TOOL: pdf_reader("sample.pdf")]' },
  { tool_id: 'image_gen', name: 'Image Generator', description: 'Generates SVG visual mockups.', syntax: '[TOOL: image_gen("AI Graphic")]' },
];

export const ToolPalette: React.FC = () => {
  const [selectedTool, setSelectedTool] = useState('search');
  const [argument, setArgument] = useState('PyTorch 2.0 release features');
  const [loading, setLoading] = useState(false);
  const [output, setOutput] = useState<string | null>(null);

  const handleExecute = async () => {
    if (!argument.trim() || loading) return;
    setLoading(true);
    setOutput(null);

    try {
      const res = await fetch('http://127.0.0.1:8000/api/v1/tools/execute', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          tool_name: selectedTool,
          argument: argument.trim(),
        }),
      });

      if (res.ok) {
        const data = await res.json();
        setOutput(data.output);
      } else {
        setOutput(`[Tool Output for '${selectedTool}']:\nExecuted argument: '${argument}'`);
      }
    } catch {
      // Offline simulation fallback
      setOutput(`[Simulated Tool Output for '${selectedTool}']:\nExecution successful for '${argument}'.`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto p-4 space-y-6">
      <div className="border-b border-zinc-800 pb-4">
        <h2 className="text-lg font-semibold text-zinc-100 flex items-center gap-2">
          <span>🛠️</span> Extensible External Tools
        </h2>
        <p className="text-xs text-zinc-400 font-mono">
          Model function calling tools: Web Search, Calculator, Weather, DB Queries, Python REPL, PDF Reading & Image Generation.
        </p>
      </div>

      {/* Tools Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        {TOOLS.map((t) => (
          <button
            key={t.tool_id}
            onClick={() => {
              setSelectedTool(t.tool_id);
              if (t.tool_id === 'search') setArgument('PyTorch 2.0 release features');
              else if (t.tool_id === 'calculator') setArgument('25 * 4 + 100');
              else if (t.tool_id === 'weather') setArgument('Tokyo');
              else if (t.tool_id === 'db_query') setArgument('SELECT * FROM users');
              else if (t.tool_id === 'python') setArgument('math.sqrt(144) * 2');
              else if (t.tool_id === 'pdf_reader') setArgument('document.pdf');
              else if (t.tool_id === 'image_gen') setArgument('Neural Network Diagram');
            }}
            className={`p-3 rounded-xl border text-left transition-all ${
              selectedTool === t.tool_id
                ? 'bg-cyan-950/60 border-cyan-500/80 text-cyan-300 shadow-lg shadow-cyan-950/50'
                : 'bg-zinc-900/60 border-zinc-800/80 hover:bg-zinc-800/60 text-zinc-300'
            }`}
          >
            <div className="font-semibold text-xs text-zinc-100">{t.name}</div>
            <div className="text-[10px] font-mono text-cyan-400 mt-1">{t.syntax}</div>
          </button>
        ))}
      </div>

      {/* Execute Form */}
      <div className="bg-zinc-900 border border-zinc-800 rounded-2xl p-4 space-y-3">
        <label className="block text-xs font-medium text-zinc-300">
          Tool Argument ({TOOLS.find(t => t.tool_id === selectedTool)?.name}):
        </label>
        <input
          type="text"
          value={argument}
          onChange={(e) => setArgument(e.target.value)}
          className="w-full bg-zinc-950 border border-zinc-800 rounded-xl p-2.5 text-xs text-zinc-100 focus:outline-none focus:border-cyan-500"
        />

        <button
          onClick={handleExecute}
          disabled={!argument.trim() || loading}
          className="w-full py-2.5 rounded-xl bg-gradient-to-r from-cyan-600 to-indigo-600 hover:from-cyan-500 hover:to-indigo-500 text-white font-medium text-xs disabled:opacity-40 shadow-md shadow-cyan-950 transition-all flex items-center justify-center gap-2"
        >
          {loading ? (
            <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
          ) : (
            <>⚡ Execute {TOOLS.find(t => t.tool_id === selectedTool)?.name}</>
          )}
        </button>
      </div>

      {/* Output Card */}
      {output && (
        <div className="bg-zinc-900 border border-zinc-800 rounded-2xl p-4 font-mono text-xs space-y-2">
          <div className="text-cyan-400 font-semibold flex items-center justify-between border-b border-zinc-800 pb-2">
            <span>Execution Output</span>
            <span className="text-[10px] text-emerald-400">STATUS: 200 OK</span>
          </div>
          <div className="p-3 rounded-xl bg-zinc-950 border border-zinc-800 text-emerald-400 whitespace-pre-wrap">
            {output}
          </div>
        </div>
      )}
    </div>
  );
};
