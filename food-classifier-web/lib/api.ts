const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

export interface Prediction {
  label: string;
  confidence: number;
}

export interface PredictResponse {
  top: Prediction;
  predictions: Prediction[];
}

export async function classifyImage(blob: Blob): Promise<PredictResponse> {
  const form = new FormData();
  form.append("file", blob, "capture.jpg");

  const res = await fetch(`${API_URL}/predict`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail ?? "Prediction failed");
  }

  return res.json();
}

export async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${API_URL}/health`);
    return res.ok;
  } catch {
    return false;
  }
}