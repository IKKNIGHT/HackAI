import { Brain } from "lucide-react";

export function Header() {
  return (
    <header className="flex items-center justify-between px-8 py-6 border-b border-white/10">
      <div className="flex items-center gap-3">
        <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-[#39FF14] to-[#00F0FF] flex items-center justify-center">
          <Brain className="w-6 h-6 text-black" />
        </div>
        <h1 className="text-2xl font-bold text-white">MindCraft</h1>
      </div>
      <div className="flex items-center gap-2 px-4 py-2 rounded-full bg-white/5 backdrop-blur-sm border border-white/10">
        <div className="w-2 h-2 rounded-full bg-[#39FF14] animate-pulse"></div>
        <span className="text-sm text-gray-300">All in-browser • Private</span>
      </div>
    </header>
  );
}
