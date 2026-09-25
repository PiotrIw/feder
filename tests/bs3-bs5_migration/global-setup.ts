import { chromium, type FullConfig } from '@playwright/test';
import * as fs from 'fs';
import * as path from 'path';

// Minimal .env reader: the project's .env is just two KEY="VALUE" lines,
// not worth pulling in the `dotenv` package for.
function loadEnvFile(filePath: string): Record<string, string> {
  const env: Record<string, string> = {};
  const content = fs.readFileSync(filePath, 'utf-8');
  for (const line of content.split('\n')) {
    const match = line.match(/^([A-Z_][A-Z0-9_]*)=(.*)$/);
    if (!match) continue;
    env[match[1]] = match[2].trim().replace(/^["']|["']$/g, '');
  }
  return env;
}

export default async function globalSetup(config: FullConfig) {
  const env = loadEnvFile(path.resolve(__dirname, '../../.env'));
  const { CLAUDE_USER, CLAUDE_PASS } = env;
  if (!CLAUDE_USER || !CLAUDE_PASS) {
    throw new Error('CLAUDE_USER / CLAUDE_PASS not found in .env');
  }

  const baseURL = config.projects[0].use.baseURL as string;
  const browser = await chromium.launch();
  const page = await browser.newPage();

  await page.goto(`${baseURL}/accounts/login/`);
  await page.fill('input[name="login"]', CLAUDE_USER);
  await page.fill('input[name="password"]', CLAUDE_PASS);
  await page.click('form.login button[type="submit"]');
  await page.waitForURL((url) => !url.pathname.startsWith('/accounts/login'));

  const authDir = path.resolve(__dirname, '.auth');
  fs.mkdirSync(authDir, { recursive: true });
  await page.context().storageState({ path: path.join(authDir, 'user.json') });

  await browser.close();
}
