import { test, expect } from '@playwright/test';
import { PAGES } from './pages';
import { VIEWPORTS } from './viewports';

// Bootstrap 3's `.row` negative margins (-15px) routinely push scrollWidth ~4px past
// clientWidth on every page even with no visible scrollbar - that's cosmetic BS3 grid
// noise, not real overflow. A genuinely overflowing wide table measured ~278px over.
// This tolerance separates the two instead of flagging every single page.
const OVERFLOW_TOLERANCE_PX = 20;

test.describe('Layout sanity - desktop', () => {
  test.use({ viewport: VIEWPORTS.desktop });

  for (const page of PAGES) {
    test(`${page.name} - no horizontal overflow`, async ({ page: pw }) => {
      await pw.goto(page.path);
      await pw.waitForLoadState('networkidle');
      const overflowPx = await pw.evaluate(() =>
        document.documentElement.scrollWidth - document.documentElement.clientWidth
      );
      expect(overflowPx, 'Page has horizontal scroll').toBeLessThanOrEqual(OVERFLOW_TOLERANCE_PX);
    });
  }
});

test.describe('Layout sanity - mobile', () => {
  test.use({ viewport: VIEWPORTS.mobile });

  for (const page of PAGES) {
    test(`${page.name} - no horizontal overflow on mobile`, async ({ page: pw }) => {
      await pw.goto(page.path);
      await pw.waitForLoadState('networkidle');
      const overflowPx = await pw.evaluate(() =>
        document.documentElement.scrollWidth - document.documentElement.clientWidth
      );
      expect(overflowPx, 'Mobile layout has horizontal scroll').toBeLessThanOrEqual(OVERFLOW_TOLERANCE_PX);
    });
  }
});

// This app's desktop layout (feder/main/templates/base.html) is a permanent left
// `.sidebar` next to `.content`, not a top navbar - `.navbar` is `display: none` above
// the mobile breakpoint (it only reappears, with `.navbar-toggle`, on small screens).
// So "nav above content" doesn't apply on desktop; the real desktop invariant is
// "sidebar sits to the left of content", checked below instead.
test.describe('Navigation structure', () => {
  test.use({ viewport: VIEWPORTS.desktop });

  test('sidebar is left of main content on desktop', async ({ page: pw }) => {
    await pw.goto('/');
    await pw.waitForLoadState('networkidle');
    const sidebarBox = await pw.locator('.sidebar').first().boundingBox();
    const contentBox = await pw.locator('.content').first().boundingBox();
    expect(sidebarBox).toBeTruthy();
    expect(contentBox).toBeTruthy();
    expect(sidebarBox!.x + sidebarBox!.width).toBeLessThanOrEqual(contentBox!.x + 5);
  });

  test('navbar collapses on mobile', async ({ page: pw }) => {
    await pw.setViewportSize(VIEWPORTS.mobile);
    await pw.goto('/');
    await pw.waitForLoadState('networkidle');
    const toggle = pw.locator('.navbar-toggle, .navbar-toggler');
    await expect(toggle).toBeVisible();
  });
});
