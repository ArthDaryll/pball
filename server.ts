import 'dotenv/config';
import express, { Request, Response } from 'express';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';
import { createServer as createViteServer } from 'vite';
import { GoogleGenAI } from '@google/genai';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PICKLEBOT_SYSTEM_INSTRUCTION = `# ROLE & PERSONALITY
You are "PickleBot," the official AI concierge embedded inside "Zero Zero Two Cebu" (Tagline: "Unsay Score? Court Finder Ni!").
Your goal is to help Cebu-based pickleball players find courts, compare rates across all 9 partner venues, compute total costs, and explain basic rules.
Maintain a friendly, energetic, and helpful local Cebuano/Bislish tone. Use light local phrases naturally (e.g., "Bay", "Bai", "Unsay score!").

# GROUNDED KNOWLEDGE BASE (ALL 9 METRO CEBU PARTNER COURTS)
1. State of Mind Pickleball Court (Hernan Cortes St., Mandaue City) | 4 Indoor Air-conditioned Courts | Open 24/7 | Rate: ₱750/hr
2. The Pickle Point Cebu (Centro, Mandaue City) | 2 Covered Courts | 8 AM - 10 PM | Rate: ₱350/hr Day (8 AM - 5 PM) / ₱500/hr Night (5 PM - 10 PM)
3. Kitchenline Pickleball Center (Mandaue City - Subangdaku/A.S. Fortuna) | 6 Covered Courts | 7 AM - 10 PM | Rate: ₱400/hr
4. SweetSpot Pickleball Courts (HVG IT Park, Subangdaku, Mandaue) | 5 Covered Courts | 8 AM - 8 PM | Rate: ₱450/hr
5. Game Changer Sports Facility (Banilad - Mandaue/Cebu City border) | 7 Covered Courts | 6 AM - 10 PM | Rate: ₱350/hr
6. IT Park Dink & Volley Studio (Salinas Dr. / Lahug, Cebu City) | 4 Indoor Air-conditioned Courts | 6 AM - 11 PM | Rate: ₱600/hr
7. Northgate Baseline Courts (Talamban Rd., Cebu City) | 4 Covered Courts | 6 AM - 10 PM | Rate: ₱350/hr
8. SRP Seaside Pickleball Deck (South Road Properties, Cebu City) | 5 Covered Courts | 5 AM - 10 PM | Rate: ₱300/hr Day / ₱400/hr Night
9. Mactan Island Pickleball Hub (Pueblo Verde, Basak, Lapu-Lapu City) | 4 Covered Courts | 7 AM - 10 PM | Rate: ₱400/hr

# GENERAL PRICING & SERVICES
- Paddle Rentals: ₱50 per paddle/session across all locations.
- Open Play Rotations: Daily 6:00 PM – 9:00 PM (₱150 entry per player, drop-in queue).
- Beginner Coaching Clinics: ₱600/hr (Includes instructor & paddle).

# BEHAVIOR & FORMATTING GUIDELINES
1. Direct Calculations: Calculate exact costs step-by-step when requested (e.g., "How much for 2 hours at SRP Seaside at night with 3 paddle rentals?"). Show clear itemized PHP math and total.
2. Location Matching: Recommend courts based on proximity (e.g., suggest SRP Seaside for users near Talisay/Pardo, or IT Park Dink & Volley for users in Lahug).
3. Concise Formatting: Keep answers clear, well-structured, and easy to read inside a compact chat window using short paragraphs or bullet points.

# RESPONSIBLE AI GUARDRAILS & FALLBACKS
1. Payment & Reservation Guardrail: You CANNOT process GCash/credit card payments or directly lock court bookings. Direct users to the front desk:
   "I can help you calculate rates and find court details, but for direct payments and locked reservations, please contact our front desk team directly at 0917-123-4567 or reach out to the court admin!"
2. Out-of-Scope Queries: Refuse non-pickleball/non-Cebu queries politely:
   "Unsay score, bay! I am programmed to assist only with pickleball courts, rates, schedules, and rules in Metro Cebu. For other inquiries, check with our admin team!"
3. Strict Accuracy: Rely ONLY on the 9 venues listed above. Do not invent fake locations or rates.`;

interface ChatTurn {
  role: 'user' | 'model';
  text: string;
}

function getGeminiClient(): GoogleGenAI {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    throw new Error('MISSING_GEMINI_API_KEY');
  }
  return new GoogleGenAI({
    apiKey,
    httpOptions: {
      headers: {
        'User-Agent': 'aistudio-build',
      },
    },
  });
}

async function startServer() {
  const app = express();
  const PORT = Number(process.env.PORT) || 3000;

  app.use(express.json({ limit: '1mb' }));

  app.post('/api/chat', async (req: Request, res: Response) => {
    try {
      const rawMessages: unknown = req.body?.messages;
      if (!Array.isArray(rawMessages) || rawMessages.length === 0) {
        res.status(400).json({ error: 'Please enter a message to ask PickleBot.' });
        return;
      }

      const sanitizedTurns: ChatTurn[] = rawMessages
        .filter(
          (item): item is ChatTurn =>
            Boolean(item) &&
            typeof item === 'object' &&
            ((item as ChatTurn).role === 'user' || (item as ChatTurn).role === 'model') &&
            typeof (item as ChatTurn).text === 'string' &&
            (item as ChatTurn).text.trim().length > 0
        )
        .slice(-20);

      // Ensure conversation starts with a user turn as expected by Gemini contents
      while (sanitizedTurns.length > 0 && sanitizedTurns[0].role !== 'user') {
        sanitizedTurns.shift();
      }

      if (sanitizedTurns.length === 0) {
        res.status(400).json({ error: 'Please enter a valid message for PickleBot.' });
        return;
      }

      const ai = getGeminiClient();

      const contents = sanitizedTurns.map((turn) => ({
        role: turn.role,
        parts: [{ text: turn.text.trim() }],
      }));

      const response = await ai.models.generateContent({
        model: 'gemini-3.8-flash',
        contents,
        config: {
          systemInstruction: PICKLEBOT_SYSTEM_INSTRUCTION,
          temperature: 0.4,
        },
      });

      const replyText =
        response.text?.trim() ||
        'Pasensya bay, I could not generate a response right now. Please try asking again!';

      res.json({ reply: replyText });
    } catch (err: unknown) {
      const rawMsg = err instanceof Error ? err.message : String(err);
      console.error('PickleBot /api/chat error:', rawMsg);

      if (
        rawMsg.includes('MISSING_GEMINI_API_KEY') ||
        rawMsg.includes('API_KEY_INVALID') ||
        rawMsg.includes('PERMISSION_DENIED') ||
        rawMsg.includes('403')
      ) {
        res.status(500).json({
          error:
            'PickleBot could not authenticate with Gemini. Please verify your GEMINI_API_KEY in the Settings > Secrets panel.',
        });
        return;
      }

      if (rawMsg.includes('RESOURCE_EXHAUSTED') || rawMsg.includes('429')) {
        res.status(429).json({
          error:
            'PickleBot hit a temporary rate limit. Please wait a few seconds or check your quota in Settings > Secrets.',
        });
        return;
      }

      res.status(500).json({
        error: 'PickleBot encountered a temporary connection issue. Please try again in a moment!',
      });
    }
  });

  if (process.env.NODE_ENV !== 'production') {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(__dirname, 'dist');
    app.use(express.static(distPath));
    app.get('*', (_req: Request, res: Response) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`Zero Zero Two Cebu server listening on http://0.0.0.0:${PORT}`);
  });
}

startServer();
