import { useState, type FormEvent, type KeyboardEvent } from "react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Loader2 } from "lucide-react";

interface ModelOption {
  id: string;
  name: string;
}

interface PromptInputProps {
  onSubmit: (prompt: string, model: string) => void;
  isLoading: boolean;
  model: string;
  models: ModelOption[];
  onModelChange: (model: string) => void;
}

export function PromptInput({
  onSubmit,
  isLoading,
  model,
  models,
  onModelChange,
}: PromptInputProps) {
  const [value, setValue] = useState("");

  const handleSubmit = (
    e: FormEvent<HTMLFormElement> | KeyboardEvent<HTMLTextAreaElement>,
  ) => {
    e.preventDefault();
    if (value.trim() && !isLoading) {
      onSubmit(value.trim(), model);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="w-full">
      <div className="mb-3">
        <label className="mb-1.5 block text-sm font-medium text-slate-700">
          Select Model
        </label>
        <Select value={model} onValueChange={onModelChange}>
          <SelectTrigger className="w-full bg-white">
            <SelectValue placeholder="Select a model" />
          </SelectTrigger>
          <SelectContent>
            {models.map((m) => (
              <SelectItem key={m.id} value={m.id}>
                {m.name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
      <div className="relative mt-2">
        <textarea
          value={value}
          onChange={(e) => setValue(e.target.value)}
          placeholder="Ask a question or enter a prompt..."
          rows={4}
          disabled={isLoading}
          className="w-full resize-none rounded-xl border border-slate-200 bg-white p-4 text-slate-800 placeholder:text-slate-400 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition-all shadow-sm"
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              handleSubmit(e);
            }
          }}
        />
        <button
          type="submit"
          disabled={!value.trim() || isLoading}
          className="absolute bottom-3 right-3 flex items-center gap-2 rounded-lg bg-slate-900 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-slate-800 disabled:opacity-50 disabled:cursor-not-allowed shadow-sm"
        >
          {isLoading && <Loader2 className="w-4 h-4 animate-spin" />}
          {isLoading ? "Working..." : "Submit"}
        </button>
      </div>
    </form>
  );
}
