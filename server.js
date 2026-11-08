import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import rateLimit from 'express-rate-limit';
import multer from 'multer';
import pdfParse from 'pdf-parse/lib/pdf-parse.js'; // direct path avoids pdf-parse's debug-mode bug
import { z } from 'zod';
import Anthropic from '@anthropic-ai/sdk';

// ---------- Config ----------
const PORT = process.env.PORT || 5000;
const MODEL = process.env.MODEL || 'claude-haiku-5-5';
const MAX_PDF_MB = Number(process.env.MAX_PDF_MB || 20);
const ORIGINS = (process.env.CLIENT_ORIGIN || 'http://localhost:5173').split(',').map(s => s.trim());
const CHUNK_CHARS = 12000;      // size of each piece sent to the model
const MAX_CHARS = 600000;       // hard cap on total text per request
const CONCURRENCY = 3;          // parallel chunk calls

const client = process.env.ANTHROPIC_API_KEY
  ? new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY })
  : null;
if (!client) console.warn('⚠ No ANTHROPIC_API_KEY set: running in MOCK mode.');

// ---------- Prompts ----------
const STYLES = {
  short: 'Write a concise summary in 1-2 short paragraphs capturing only the main ideas.',
  detailed: 'Write a thorough summary with clear markdown headings (##) for each major topic, covering key concepts, definitions, and examples.',
  bullets: 'Summarize as clear markdown bullet points grouped under short headings (##). Keep each bullet to one line.',
  exam: 'Create exam revision notes in markdown: ## headings per topic, key definitions, formulas, important points as bullets, and a short "Quick Revision" list at the end.',
};
const SYSTEM = 'You are an expert study assistant for students. Be accurate and only use information from the provided text. Do not invent facts. Output clean markdown.';

// ---------- Helpers ----------
function chunkText(text, size = CHUNK_CHARS) {
  const paras = text.split(/\n{2,}/);
  const chunks = [];
  let cur = '';
  for (let p of paras) {
    while (p.length > size) {            // split oversized paragraphs
      if (cur) { chunks.push(cur); cur = ''; }
      chunks.push(p.slice(0, size));
      p = p.slice(size);
    }
    if ((cur + '\n\n' + p).length > size) { chunks.push(cur); cur = p; }
    else cur = cur ? cur + '\n\n' + p : p;
  }
  if (cur.trim()) chunks.push(cur);
  return chunks;
}

async function ask(userPrompt, maxTokens = 1800) {
  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      const res = await client.messages.create({
        model: MODEL,
        max_tokens: maxTokens,
        system: SYSTEM,
        messages: [{ role: 'user', content: userPrompt }],
      });
      return res.content.filter(b => b.type === 'text').map(b => b.text).join('\n').trim();
    } catch (err) {
      const retryable = [429, 500, 502, 503, 529].includes(err?.status);
      if (!retryable || attempt === 2) throw err;
      await new Promise(r => setTimeout(r, 1000 * 2 ** attempt));
    }
  }
}

async function mapLimit(items, limit, fn) {
  const out = new Array(items.length);
  let i = 0;
  await Promise.all(Array.from({ length: Math.min(limit, items.length) }, async () => {
    while (i < items.length) { const idx = i++; out[idx] = await fn(items[idx], idx); }
  }));
  return out;
}

function mockSummary(text, style) {
  const sentences = text.replace(/\s+/g, ' ').match(/[^.!?]+[.!?]/g) || [text.slice(0, 300)];
  const pick = sentences.slice(0, style === 'short' ? 3 : 8).map(s => s.trim());
  const body = style === 'short' ? pick.join(' ') : pick.map(s => `- ${s}`).join('\n');
  return `## Mock summary\n_(Add ANTHROPIC_API_KEY to get real AI summaries.)_\n\n${body}`;
}

async function summarize(text, style) {
  if (!client) return { summary: mockSummary(text, style), chunks: 1, mock: true };

  let chunks = chunkText(text);
  if (chunks.length === 1) {
    return { summary: await ask(`${STYLES[style]}\n\nTEXT:\n${chunks[0]}`), chunks: 1 };
  }

  const total = chunks.length;
  // MAP: condense each chunk into dense notes
  let notes = await mapLimit(chunks, CONCURRENCY, (c, i) =>
    ask(`This is part ${i + 1} of ${total} of a document. Extract all key points, definitions, formulas and examples as compact notes.\n\nTEXT:\n${c}`, 1200));

  // Recursively merge if the notes are still too big
  let combined = notes.join('\n\n---\n\n');
  while (combined.length > CHUNK_CHARS * 1.5) {
    notes = await mapLimit(chunkText(combined), CONCURRENCY, c =>
      ask(`Condense these notes while keeping every key point:\n\n${c}`, 1200));
    combined = notes.join('\n\n');
  }

  // REDUCE: final summary in requested style
  const summary = await ask(`${STYLES[style]}\n\nThese are notes from a full document, in order. Produce one coherent summary.\n\nNOTES:\n${combined}`);
  return { summary, chunks: total };
}

// ---------- App ----------
const app = express();
app.set('trust proxy', 1);
app.use(helmet());
   app.use(cors());
app.use(express.json({ limit: '2mb' }));
app.use('/api/', rateLimit({ windowMs: 60_000, limit: 15, standardHeaders: true, legacyHeaders: false,
  message: { error: 'Too many requests. Please wait a minute and try again.' } }));

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: MAX_PDF_MB * 1024 * 1024, files: 1 },
});

const bodySchema = z.object({
  text: z.string().min(50, 'Not enough text to summarize.').max(MAX_CHARS, 'Document is too long.'),
  style: z.enum(['short', 'detailed', 'bullets', 'exam']).default('detailed'),
});

app.get('/api/health', (_req, res) => res.json({ ok: true, mock: !client, model: MODEL }));

// Option A: frontend extracts text with pdfjs-dist and sends JSON
app.post('/api/summarize', async (req, res, next) => {
  try {
    const parsed = bodySchema.safeParse(req.body);
    if (!parsed.success) return res.status(400).json({ error: parsed.error.issues[0].message });
    const { text, style } = parsed.data;
    res.json(await summarize(text.trim(), style));
  } catch (e) { next(e); }
});

// Option B: upload the PDF itself (multipart field name: "file")
app.post('/api/summarize-pdf', upload.single('file'), async (req, res, next) => {
  try {
    if (!req.file) return res.status(400).json({ error: 'No file uploaded.' });
    if (req.file.buffer.subarray(0, 5).toString() !== '%PDF-')
      return res.status(400).json({ error: 'File is not a valid PDF.' });

    const style = z.enum(['short', 'detailed', 'bullets', 'exam']).catch('detailed').parse(req.body.style);
    let data;
    try { data = await pdfParse(req.file.buffer); }
    catch { return res.status(422).json({ error: 'Could not read this PDF (it may be corrupted or password-protected).' }); }

    const text = (data.text || '').trim();
    if (text.length < 50)
      return res.status(422).json({ error: 'No readable text found. This looks like a scanned/image PDF.' });
    if (text.length > MAX_CHARS)
      return res.status(413).json({ error: 'Document is too long. Try a smaller PDF.' });

    res.json({ ...(await summarize(text, style)), pages: data.numpages });
  } catch (e) { next(e); }
});

// ---------- Errors ----------
app.use((_req, res) => res.status(404).json({ error: 'Not found' }));
app.use((err, _req, res, _next) => {
  if (err instanceof multer.MulterError)
    return res.status(413).json({ error: err.code === 'LIMIT_FILE_SIZE' ? `File too large (max ${MAX_PDF_MB} MB).` : err.message });
  if (err?.status === 429) return res.status(503).json({ error: 'AI service is busy. Please try again shortly.' });
  console.error(err);
  res.status(500).json({ error: 'Something went wrong while summarizing. Please try again.' });
});

app.listen(PORT, () => console.log(`StudySnap API running on http://localhost:${PORT}`));
