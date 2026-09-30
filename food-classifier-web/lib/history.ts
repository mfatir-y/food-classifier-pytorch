import { useSyncExternalStore } from "react";

export interface HistoryEntry {
  id: string;
  label: string;
  confidence: number;
  thumbnail: string; // small JPEG data URL, or "" if it couldn't be generated
  timestamp: number;
}

const STORAGE_KEY = "food-classifier-history";
const MAX_ENTRIES = 10;
const EMPTY_HISTORY: HistoryEntry[] = [];

// History lives in localStorage, so it's modeled as an external store rather than
// component state: a module-level cache, a set of subscribers, and reads/writes that
// stay in sync with both. useSyncExternalStore below is what lets components read it
// safely without a server/client hydration mismatch on first paint.
let cache: HistoryEntry[] | null = null;
const listeners = new Set<() => void>();

function readFromStorage(): HistoryEntry[] {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    return raw ? (JSON.parse(raw) as HistoryEntry[]) : [];
  } catch {
    // Private browsing, disabled storage, or corrupted data — history is a nice-to-have, fail quietly.
    return [];
  }
}

function writeToStorage(entries: HistoryEntry[]) {
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(entries));
  } catch {
    // Quota exceeded or storage unavailable — drop the write, nothing else depends on it.
  }
}

function setHistory(entries: HistoryEntry[]) {
  cache = entries;
  writeToStorage(entries);
  listeners.forEach((listener) => listener());
}

function subscribe(listener: () => void) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

function getSnapshot(): HistoryEntry[] {
  if (cache === null) cache = readFromStorage();
  return cache;
}

function getServerSnapshot(): HistoryEntry[] {
  return EMPTY_HISTORY;
}

/** Subscribes a component to the persisted classification history. */
export function useHistory(): HistoryEntry[] {
  return useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot);
}

/** Adds an entry to the front of history and caps the list. */
export function addHistoryEntry(entry: HistoryEntry) {
  setHistory([entry, ...getSnapshot()].slice(0, MAX_ENTRIES));
}

export function clearHistory() {
  setHistory([]);
}

/** Downscales an image file into a small square JPEG data URL, cheap enough to keep in localStorage. */
export function createThumbnail(file: File, size = 64): Promise<string> {
  return new Promise((resolve, reject) => {
    const img = new window.Image();
    const objectUrl = URL.createObjectURL(file);

    img.onload = () => {
      URL.revokeObjectURL(objectUrl);

      const canvas = document.createElement("canvas");
      canvas.width = size;
      canvas.height = size;
      const ctx = canvas.getContext("2d");
      if (!ctx) {
        reject(new Error("Canvas not supported"));
        return;
      }

      // Cover-crop to a square so thumbnails line up regardless of source aspect ratio.
      const scale = Math.max(size / img.width, size / img.height);
      const w = img.width * scale;
      const h = img.height * scale;
      ctx.drawImage(img, (size - w) / 2, (size - h) / 2, w, h);

      resolve(canvas.toDataURL("image/jpeg", 0.6));
    };

    img.onerror = () => {
      URL.revokeObjectURL(objectUrl);
      reject(new Error("Failed to load image for thumbnail"));
    };

    img.src = objectUrl;
  });
}
