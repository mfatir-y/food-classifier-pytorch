import { useRef, useState, type ChangeEvent, type DragEvent } from "react";
import { ACCEPTED_TYPES } from "@/lib/validation";

interface UploadZoneProps {
  previewUrl: string | null;
  onFileSelect: (file: File) => void;
}

/** Click-to-browse / drag-and-drop image picker. Shows a preview once a file is chosen. */
export function UploadZone({ previewUrl, onFileSelect }: UploadZoneProps) {
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  function onInputChange(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (file) onFileSelect(file);
  }

  function onDrop(e: DragEvent<HTMLDivElement>) {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file) onFileSelect(file);
  }

  function onDragOver(e: DragEvent<HTMLDivElement>) {
    e.preventDefault();
    setIsDragging(true);
  }

  function onDragLeave() {
    setIsDragging(false);
  }

  return (
    <>
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
    </>
  );
}
