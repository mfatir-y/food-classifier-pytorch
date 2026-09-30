import type { HistoryEntry } from "@/lib/history";
import { formatLabel } from "@/lib/format";

interface HistoryPanelProps {
  entries: HistoryEntry[];
  onClear: () => void;
}

/** Recently classified images, stored client-side in localStorage. */
export function HistoryPanel({ entries, onClear }: HistoryPanelProps) {
  if (entries.length === 0) return null;

  return (
    <div className="flex w-full flex-col gap-3 border-t border-zinc-200 pt-6 dark:border-zinc-800">
      <div className="flex items-center justify-between">
        <h2 className="text-xs font-medium uppercase tracking-wide text-zinc-400 dark:text-zinc-500">
          Recent
        </h2>
        <button
          onClick={onClear}
          className="text-xs font-medium text-zinc-400 underline hover:text-zinc-700 dark:text-zinc-500 dark:hover:text-zinc-200"
        >
          Clear
        </button>
      </div>

      <div className="flex flex-wrap gap-3">
        {entries.map((entry) => (
          <div key={entry.id} className="flex w-16 flex-col items-center gap-1">
            {entry.thumbnail ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                src={entry.thumbnail}
                alt={formatLabel(entry.label)}
                className="h-16 w-16 rounded-lg object-cover"
              />
            ) : (
              <div className="h-16 w-16 rounded-lg bg-zinc-200 dark:bg-zinc-800" />
            )}
            <span className="w-full truncate text-center text-[11px] text-zinc-600 dark:text-zinc-400">
              {formatLabel(entry.label)}
            </span>
            <span className="text-[10px] text-zinc-400 dark:text-zinc-600">
              {(entry.confidence * 100).toFixed(0)}%
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
