/**
 * Evenly-spaced downsampling for chart data. Training histories can have up
 * to 25,000+ rows (see `rl/results/`); rendering that many SVG points would
 * be slow and visually indistinguishable from a downsampled curve, so chart
 * components should always downsample before passing data to Recharts.
 */
export function downsample<T>(data: T[], maxPoints: number): T[] {
  if (data.length <= maxPoints || maxPoints <= 0) return data;

  const step = data.length / maxPoints;
  const result: T[] = [];
  for (let i = 0; i < maxPoints; i++) {
    result.push(data[Math.floor(i * step)]);
  }

  // Always keep the true final point (the end-of-training state) rather
  // than whatever rounding happened to land there.
  const last = data[data.length - 1];
  if (result[result.length - 1] !== last) {
    result[result.length - 1] = last;
  }
  return result;
}
