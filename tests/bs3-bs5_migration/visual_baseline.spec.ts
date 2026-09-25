import { test, expect } from '@playwright/test';
import { PAGES } from './pages';
import { VIEWPORTS } from './viewports';

for (const [name, size] of Object.entries(VIEWPORTS)) {
  test.describe(`${name} (${size.width}px)`, () => {
    test.use({ viewport: size });

    for (const page of PAGES) {
      test(`${page.name} - visual baseline`, async ({ page: pw }) => {
        await pw.goto(page.path);
        await pw.waitForLoadState('networkidle');
        await expect(pw).toHaveScreenshot(`${page.name}-${name}.png`, {
          maxDiffPixelRatio: 0.001,
          fullPage: true,
        });
      });
    }
  });
}
