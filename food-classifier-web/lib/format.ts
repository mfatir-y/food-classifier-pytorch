/** "apple_pie" -> "Apple Pie" */
export function formatLabel(label: string) {
  return label
    .split("_")
    .map((word) => word[0].toUpperCase() + word.slice(1))
    .join(" ");
}
