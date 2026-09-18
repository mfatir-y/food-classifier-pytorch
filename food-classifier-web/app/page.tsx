"use client";

import { useState } from "react";
import { classifyImage, type PredictResponse } from "@/lib/api";
import { validateImageFile } from "@/lib/validation";
import { UploadZone } from "./components/UploadZone";
import { ResultsPanel } from "./components/ResultsPanel";

type Status = "idle" | "loading" | "success" | "error";

export default function Home() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [status, setStatus] = useState<Status>("idle");
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  // Bumped on reset to remount UploadZone, clearing its internal file input.
  const [uploadKey, setUploadKey] = useState(0);

  function reset() {
    setSelectedFile(null);
    setPreviewUrl(null);
    setStatus("idle");
    setResult(null);
    setError(null);
    setUploadKey((key) => key + 1);
  }

  function handleFileSelect(file: File) {
    const validationError = validateImageFile(file);
    if (validationError) {
      setError(validationError);
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

        <UploadZone key={uploadKey} previewUrl={previewUrl} onFileSelect={handleFileSelect} />

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

        {status === "error" && error && (
          <div className="w-full rounded-xl bg-red-50 px-4 py-3 text-sm text-red-700 dark:bg-red-950/40 dark:text-red-400">
            {error}
          </div>
        )}

        {status === "success" && result && <ResultsPanel result={result} />}

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
