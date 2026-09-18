export const ACCEPTED_TYPES = ["image/jpeg", "image/png", "image/webp"];
export const MAX_SIZE_BYTES = 10 * 1024 * 1024;

/** Returns an error message if the file is invalid, or null if it's fine. */
export function validateImageFile(file: File): string | null {
  if (!ACCEPTED_TYPES.includes(file.type)) {
    return "Please choose a JPEG, PNG, or WebP image.";
  }
  if (file.size > MAX_SIZE_BYTES) {
    return "That image is over 10MB. Please choose a smaller file.";
  }
  return null;
}
