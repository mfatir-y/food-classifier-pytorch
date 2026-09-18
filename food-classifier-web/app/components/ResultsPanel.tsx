import type { PredictResponse } from "@/lib/api";
import { formatLabel } from "@/lib/format";

interface ResultsPanelProps {
  result: PredictResponse;
}

/** Top prediction card plus a top-5 confidence bar list. */
export function ResultsPanel({ result }: ResultsPanelProps) {
  return (
    <div className="flex w-full flex-col gap-4">
      <div className="flex flex-col items-center gap-1 rounded-2xl bg-zinc-900 px-6 py-5 text-center dark:bg-zinc-50">
        <span className="text-xs font-medium uppercase tracking-wide text-zinc-400 dark:text-zinc-500">
          Top prediction
        </span>
        <span className="text-xl font-semibold text-zinc-50 dark:text-zinc-900">
          {formatLabel(result.top.label)}
        </span>
        <span className="text-sm text-zinc-400 dark:text-zinc-500">
          {(result.top.confidence * 100).toFixed(1)}% confident
        </span>
      </div>

      <div className="flex flex-col gap-2">
        {result.predictions.map((pred) => (
          <div key={pred.label} className="flex items-center gap-3">
            <span className="w-28 shrink-0 truncate text-sm text-zinc-600 dark:text-zinc-400">
              {formatLabel(pred.label)}
            </span>
            <div className="h-2 flex-1 overflow-hidden rounded-full bg-zinc-200 dark:bg-zinc-800">
              <div
                className="h-full rounded-full bg-zinc-900 dark:bg-zinc-50"
                style={{ width: `${pred.confidence * 100}%` }}
              />
            </div>
            <span className="w-12 shrink-0 text-right text-sm text-zinc-500 dark:text-zinc-500">
              {(pred.confidence * 100).toFixed(1)}%
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
