export function normalizeF5TTSSpeechText(input) {
  let text = String(input || '').replace(/\s+/g, ' ').trim();
  if (!text) return text;

  const replacements = [
    [/\bF5\s*-?\s*TTS\b/gi, 'eff five tee tee ess'],
    [/\bF5TTS\b/gi, 'eff five tee tee ess'],
    [/\bF5\b/g, 'F five'],
    [/\bTTS\b/g, 'T T S'],
    [/\bWebGPU\b/g, 'Web G P U'],
    [/\bGPU\b/g, 'G P U'],
    [/\bWASM\b/g, 'Web Assembly'],
    [/\bWebAssembly\b/g, 'Web Assembly'],
    [/\bint4\b/gi, 'four bit'],
    [/\bq4\b/gi, 'four bit'],
    [/\b24\s*kHz\b/gi, 'twenty four kilo hertz'],
    [/\b24\s*khz\b/gi, 'twenty four kilo hertz'],
    [/\bkHz\b/gi, 'kilo hertz'],
    [/\bVocos\b/gi, 'voh cohs'],
    [/\bvoice cloning quality\b/gi, 'voice clone quality'],
  ];
  for (const [pattern, replacement] of replacements) {
    text = text.replace(pattern, replacement);
  }
  return text.replace(/\s+/g, ' ').trim();
}
