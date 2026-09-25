import { defineConfig } from '@playwright/test';
import * as path from 'path';

// Self-contained in tests/bs3-bs5_migration (not the repo-wide tests/ dir),
// since tests/cypress.config.js already owns tests/cypress/e2e/**/*.cy.{js,ts}
// as a separate, unrelated e2e suite.
export default defineConfig({
  testDir: '.',
  snapshotDir: './snapshots',
  globalSetup: require.resolve('./global-setup.ts'),
  outputDir: './test-results',
  reporter: [['html', { outputFolder: 'playwright-report' }]],
  use: {
    baseURL: 'http://localhost:8000',
    storageState: path.resolve(__dirname, '.auth/user.json'),
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
  },
  expect: {
    toHaveScreenshot: {
      maxDiffPixelRatio: 0.001,
    },
  },
});
