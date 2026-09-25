# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: layout_assertions.spec.ts >> Layout sanity - mobile >> letters-spam - no horizontal overflow on mobile
- Location: tests/bs3-bs5_migration/layout_assertions.spec.ts:30:9

# Error details

```
Error: Mobile layout has horizontal scroll

expect(received).toBeLessThanOrEqual(expected)

Expected: <= 20
Received:    180
```

# Page snapshot

```yaml
- generic [active] [ref=e1]:
  - banner [ref=e2]:
    - heading "Page not found (404)" [level=1] [ref=e3]
    - generic [ref=e4]: Nie znaleziono List spełniających wybrane kryteria
    - table [ref=e5]:
      - rowgroup [ref=e6]:
        - row [ref=e7]:
          - rowheader "Request Method:" [ref=e8]
          - cell "GET" [ref=e9]
        - row [ref=e10]:
          - rowheader "Request URL:" [ref=e11]
          - cell "http://localhost:8000/listy/7302/~spam" [ref=e12]
        - row [ref=e13]:
          - rowheader "Raised by:" [ref=e14]
          - cell "feder.letters.views.LetterReportSpamView" [ref=e15]
  - main [ref=e16]:
    - paragraph [ref=e17]:
      - text: Using the URLconf defined in
      - code [ref=e18]: feder.main.urls
      - text: ", Django tried these URL patterns, in this order:"
    - list [ref=e19]:
      - listitem [ref=e20]:
        - code [ref=e21]: __debug__/
      - listitem [ref=e22]:
        - code [ref=e23]: ^$ [name='home']
      - listitem [ref=e24]:
        - code [ref=e25]: ^o-stronie/ [name='about']
      - listitem [ref=e26]:
        - code [ref=e27]: ^admin/
      - listitem [ref=e28]:
        - code [ref=e29]: ^uzytkownik/
      - listitem [ref=e30]:
        - code [ref=e31]: accounts/
      - listitem [ref=e32]:
        - code [ref=e33]: ^instytucje/
      - listitem [ref=e34]:
        - code [ref=e35]: ^monitoringi/
      - listitem [ref=e36]:
        - code [ref=e37]: ^sprawy/
      - listitem [ref=e38]:
        - code [ref=e39]: ^sprawy/tagi/
      - listitem [ref=e40]:
        - code [ref=e41]: ^alerty/
      - listitem [ref=e42]:
        - code [ref=e43]: ^listy/
        - code [ref=e44]: ^$ [name='list']
      - listitem [ref=e45]:
        - code [ref=e46]: ^listy/
        - code [ref=e47]: ^feed$ [name='rss']
      - listitem [ref=e48]:
        - code [ref=e49]: ^listy/
        - code [ref=e50]: ^feed/atom$ [name='atom']
      - listitem [ref=e51]:
        - code [ref=e52]: ^listy/
        - code [ref=e53]: ^kanal/monitoring-(?P<monitoring_pk>[\d-]+)/$ [name='rss']
      - listitem [ref=e54]:
        - code [ref=e55]: ^listy/
        - code [ref=e56]: ^kanal/monitoring-(?P<monitoring_pk>[\d-]+)/atom$ [name='atom']
      - listitem [ref=e57]:
        - code [ref=e58]: ^listy/
        - code [ref=e59]: ^kanal/sprawa-(?P<case_pk>[\d-]+)/$ [name='rss']
      - listitem [ref=e60]:
        - code [ref=e61]: ^listy/
        - code [ref=e62]: ^kanal/sprawa-(?P<case_pk>[\d-]+)/atom$ [name='atom']
      - listitem [ref=e63]:
        - code [ref=e64]: ^listy/
        - code [ref=e65]: ^~utworz-(?P<case_pk>[\d-]+)$ [name='create']
      - listitem [ref=e66]:
        - code [ref=e67]: ^listy/
        - code [ref=e68]: ^(?P<pk>[\d-]+)$ [name='details']
      - listitem [ref=e69]:
        - code [ref=e70]: ^listy/
        - code [ref=e71]: ^(?P<pk>[\d-]+)-msg$ [name='download']
      - listitem [ref=e72]:
        - code [ref=e73]: ^listy/
        - code [ref=e74]: ^attachment/(?P<pk>[\d-]+)/(?P<letter_pk>[\d-]+)/~scan [name='scan']
      - listitem [ref=e75]:
        - code [ref=e76]: ^listy/
        - code [ref=e77]: ^zalacznik/(?P<pk>[\d-]+)/(?P<letter_pk>[\d-]+)$ [name='attachment']
      - listitem [ref=e78]:
        - code [ref=e79]: ^listy/
        - code [ref=e80]: ^attachment/(?P<pk>[\d-]+)/(?P<letter_pk>[\d-]+)$
      - listitem [ref=e81]:
        - code [ref=e82]: ^listy/
        - code [ref=e83]: ^(?P<pk>[\d-]+)/~edytuj$ [name='update']
      - listitem [ref=e84]:
        - code [ref=e85]: ^listy/
        - code [ref=e86]: ^(?P<pk>[\d-]+)/~wyslij [name='send']
      - listitem [ref=e87]:
        - code [ref=e88]: ^listy/
        - code [ref=e89]: ^(?P<pk>[\d-]+)/~usun$ [name='delete']
      - listitem [ref=e90]:
        - code [ref=e91]: ^listy/
        - code [ref=e92]: ^(?P<pk>[\d-]+)/~odpowiedz$ [name='reply']
      - listitem [ref=e93]:
        - code [ref=e94]: ^listy/
        - code [ref=e95]: ^(?P<pk>[\d-]+)/~resend$ [name='resend']
      - listitem [ref=e96]:
        - code [ref=e97]: ^listy/
        - code [ref=e98]: ^(?P<pk>[\d-]+)/~spam [name='spam']
    - paragraph [ref=e99]:
      - text: The current path,
      - code [ref=e100]: listy/7302/~spam
      - text: ", matched the last one."
  - contentinfo [ref=e101]:
    - paragraph [ref=e102]:
      - text: You’re seeing this error because you have
      - code [ref=e103]: DEBUG = True
      - text: in your Django settings file. Change that to
      - code [ref=e104]: "False"
      - text: ", and Django will display a standard 404 page."
  - list [ref=e107]:
    - listitem [ref=e108]:
      - link "Ukryj »" [ref=e109] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e110]:
      - link "Toggle Theme" [ref=e111] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e114]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e115]
      - link "Historia /listy/7302/~spam" [ref=e116] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e117]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e118]
      - link "Wersje Django 5.2.17" [ref=e119] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e120]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e121]
      - 'link "Czas CPU: 55.67ms (57.27ms)" [ref=e122] [cursor=pointer]':
        - /url: "#"
    - listitem [ref=e123]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e124]
      - link "Ustawienia" [ref=e125] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e126]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e127]
      - link "Nagłówki" [ref=e128] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e129]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e130]
      - link "Zapytania LetterReportSpamView" [ref=e131] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e132]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e133]
      - link "SQL 5 queries in 1.26ms" [ref=e134] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e135]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e136]
      - link "Pliki statyczne 0 użytych plików" [ref=e137] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e138]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e139]
      - link "Templatki" [ref=e140] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e141]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e142]
      - link "Alerty" [ref=e143] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e144]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e145]
      - link "Cache 0 wywołań w 0.00ms" [ref=e146] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e147]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e148]
      - link "Sygnały 88 odbiorców 15 sygnałów" [ref=e149] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e150]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e151]
      - link "Gmina" [ref=e152] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e153]:
      - checkbox "Enable for next and successive requests" [ref=e154]
      - generic [ref=e155]: Przechwycone przekierowania
    - listitem [ref=e156]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e157]
      - link "Profilowanie" [ref=e158] [cursor=pointer]:
        - /url: "#"
```

# Test source

```ts
  1  | import { test, expect } from '@playwright/test';
  2  | import { PAGES } from './pages';
  3  | import { VIEWPORTS } from './viewports';
  4  | 
  5  | // Bootstrap 3's `.row` negative margins (-15px) routinely push scrollWidth ~4px past
  6  | // clientWidth on every page even with no visible scrollbar - that's cosmetic BS3 grid
  7  | // noise, not real overflow. A genuinely overflowing wide table measured ~278px over.
  8  | // This tolerance separates the two instead of flagging every single page.
  9  | const OVERFLOW_TOLERANCE_PX = 20;
  10 | 
  11 | test.describe('Layout sanity - desktop', () => {
  12 |   test.use({ viewport: VIEWPORTS.desktop });
  13 | 
  14 |   for (const page of PAGES) {
  15 |     test(`${page.name} - no horizontal overflow`, async ({ page: pw }) => {
  16 |       await pw.goto(page.path);
  17 |       await pw.waitForLoadState('networkidle');
  18 |       const overflowPx = await pw.evaluate(() =>
  19 |         document.documentElement.scrollWidth - document.documentElement.clientWidth
  20 |       );
  21 |       expect(overflowPx, 'Page has horizontal scroll').toBeLessThanOrEqual(OVERFLOW_TOLERANCE_PX);
  22 |     });
  23 |   }
  24 | });
  25 | 
  26 | test.describe('Layout sanity - mobile', () => {
  27 |   test.use({ viewport: VIEWPORTS.mobile });
  28 | 
  29 |   for (const page of PAGES) {
  30 |     test(`${page.name} - no horizontal overflow on mobile`, async ({ page: pw }) => {
  31 |       await pw.goto(page.path);
  32 |       await pw.waitForLoadState('networkidle');
  33 |       const overflowPx = await pw.evaluate(() =>
  34 |         document.documentElement.scrollWidth - document.documentElement.clientWidth
  35 |       );
> 36 |       expect(overflowPx, 'Mobile layout has horizontal scroll').toBeLessThanOrEqual(OVERFLOW_TOLERANCE_PX);
     |                                                                 ^ Error: Mobile layout has horizontal scroll
  37 |     });
  38 |   }
  39 | });
  40 | 
  41 | // This app's desktop layout (feder/main/templates/base.html) is a permanent left
  42 | // `.sidebar` next to `.content`, not a top navbar - `.navbar` is `display: none` above
  43 | // the mobile breakpoint (it only reappears, with `.navbar-toggle`, on small screens).
  44 | // So "nav above content" doesn't apply on desktop; the real desktop invariant is
  45 | // "sidebar sits to the left of content", checked below instead.
  46 | test.describe('Navigation structure', () => {
  47 |   test.use({ viewport: VIEWPORTS.desktop });
  48 | 
  49 |   test('sidebar is left of main content on desktop', async ({ page: pw }) => {
  50 |     await pw.goto('/');
  51 |     await pw.waitForLoadState('networkidle');
  52 |     const sidebarBox = await pw.locator('.sidebar').first().boundingBox();
  53 |     const contentBox = await pw.locator('.content').first().boundingBox();
  54 |     expect(sidebarBox).toBeTruthy();
  55 |     expect(contentBox).toBeTruthy();
  56 |     expect(sidebarBox!.x + sidebarBox!.width).toBeLessThanOrEqual(contentBox!.x + 5);
  57 |   });
  58 | 
  59 |   test('navbar collapses on mobile', async ({ page: pw }) => {
  60 |     await pw.setViewportSize(VIEWPORTS.mobile);
  61 |     await pw.goto('/');
  62 |     await pw.waitForLoadState('networkidle');
  63 |     const toggle = pw.locator('.navbar-toggle, .navbar-toggler');
  64 |     await expect(toggle).toBeVisible();
  65 |   });
  66 | });
  67 | 
```