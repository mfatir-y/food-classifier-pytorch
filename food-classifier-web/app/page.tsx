"use client";

import { useRef, useState, type ChangeEvent, type DragEvent } from "react";
import { classifyImage, type PredictResponse } from "@/lib/api";

const ACCEPTED_TYPES = ["image/jpeg", "image/png", "image/webp"];
const MAX_SIZE_BYTES = 10 * 1024 * 1024;

function formatLabel(label: string) {
  return label
    .split("_")
    .map((word) => word[0].toUpperCase() + word.slice(1))
    .join(" ");
}

type Status = "idle" | "loading" | "success" | "error";

export default function Home() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [status, setStatus] = useState<Status>("idle");
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  function reset() {
    setSelectedFile(null);
    setPreviewUrl(null);
    setStatus("idle");
    setResult(null);
    setError(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  }

  function handleFile(file: File) {
    if (!ACCEPTED_TYPES.includes(file.type)) {
      setError("Please choose a JPEG, PNG, or WebP image.");
      setStatus("error");
      return;
    }
    if (file.size > MAX_SIZE_BYTES) {
      setError("That image is over 10MB. Please choose a smaller file.");
      setStatus("error");
      return;
    }

    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setResult(null);
    setError(null);
    setStatus("idle");
  }

  async function handleClassify() {
    if (!selectedFile) return;

    setStatus("loading");
    setError(null);

    try {
      const prediction = await classifyImage(selectedFile);
      setResult(prediction);
      setStatus("success");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Prediction failed.");
      setStatus("error");
    }
  }

  function onInputChange(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  }

  function onDrop(e: DragEvent<HTMLDivElement>) {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file) handleFile(file);
  }

  function onDragOver(e: DragEvent<HTMLDivElement>) {
    e.preventDefault();
    setIsDragging(true);
  }

  function onDragLeave() {
    setIsDragging(false);
  }

  return (
    <div className="flex flex-1 items-center justify-center bg-zinc-50 px-6 py-16 dark:bg-black">
      <main className="flex w-full max-w-md flex-col items-center gap-8">
        <div className="flex flex-col items-center gap-2 text-center">
          <h1 className="text-2xl font-semibold tracking-tight text-zinc-900 dark:text-zinc-50">
            Food Classifier
          </h1>
          <p className="text-sm text-zinc-500 dark:text-zinc-400">
            Upload a photo of food and see what the model thinks it is.
          </p>
        </div>

        {/* Upload zone */}
        <div
          onDrop={onDrop}
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
          onClick={() => fileInputRef.current?.click()}
          className={`flex w-full cursor-pointer flex-col items-center justify-center gap-3 rounded-2xl border-2 border-dashed p-8 text-center transition-colors ${
            isDragging
              ? "border-zinc-900 bg-zinc-100 dark:border-zinc-50 dark:bg-zinc-900"
              : "border-zinc-300 hover:border-zinc-400 dark:border-zinc-700 dark:hover:border-zinc-600"
          }`}
        >
          {previewUrl ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={previewUrl}
              alt="Selected food"
              className="max-h-56 w-full rounded-lg object-contain"
            />
          ) : (
            <>
              <svg
                width="32"
                height="32"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.5"
                className="text-zinc-400 dark:text-zinc-600"
              >
                <path d="M12 16V4M12 4l-4 4M12 4l4 4" strokeLinecap="round" strokeLinejoin="round" />
                <path d="M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
              <p className="text-sm text-zinc-600 dark:text-zinc-400">
                Drag & drop an image, or{" "}
                <span className="font-medium text-zinc-900 underline dark:text-zinc-50">
                  browse
                </span>
              </p>
              <p className="text-xs text-zinc-400 dark:text-zinc-600">
                JPEG, PNG, or WebP — up to 10MB
              </p>
            </>
          )}
        </div>
        <input
          ref={fileInputRef}
          type="file"
          accept={ACCEPTED_TYPES.join(",")}
          onChange={onInputChange}
          className="hidden"
        />

        {/* Submit button */}
        {selectedFile && status !== "success" && (
          <button
            onClick={handleClassify}
            disabled={status === "loading"}
            className="flex w-full items-center justify-center gap-2 rounded-full bg-zinc-900 px-5 py-2.5 text-sm font-medium text-zinc-50 transition-colors hover:bg-zinc-700 disabled:cursor-not-allowed disabled:opacity-60 dark:bg-zinc-50 dark:text-zinc-900 dark:hover:bg-zinc-300"
          >
            {status === "loading" && (
              <span className="h-4 w-4 animate-spin rounded-full border-2 border-zinc-500 border-t-zinc-50 dark:border-zinc-400 dark:border-t-zinc-900" />
            )}
            {status === "loading" ? "Classifying..." : "Classify"}
          </button>
        )}

        {/* Error state */}
        {status === "error" && error && (
          <div className="w-full rounded-xl bg-red-50 px-4 py-3 text-sm text-red-700 dark:bg-red-950/40 dark:text-red-400">
            {error}
          </div>
        )}

        {/* Results */}
        {status === "success" && result && (
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
        )}

        {(status === "success" || status === "error") && (
          <button
            onClick={reset}
            className="text-sm font-medium text-zinc-500 underline hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-zinc-50"
          >
            Try another image
          </button>
        )}
      </main>
    </div>
  );
}
