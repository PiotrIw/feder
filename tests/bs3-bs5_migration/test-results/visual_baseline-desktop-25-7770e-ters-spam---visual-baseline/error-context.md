# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: visual_baseline.spec.ts >> desktop (2560px) >> letters-spam - visual baseline
- Location: tests/bs3-bs5_migration/visual_baseline.spec.ts:10:11

# Error details

```
Error: expect(page).toHaveScreenshot(expected) failed

  334615 pixels (ratio 0.10 of all image pixels) are different.

  Snapshot: letters-spam-desktop.png

Call log:
  - Expect "toHaveScreenshot(letters-spam-desktop.png)" with timeout 5000ms
    - verifying given screenshot expectation
  - taking page screenshot
    - disabled all CSS animations
  - waiting for fonts to load...
  - fonts loaded
  - 334615 pixels (ratio 0.10 of all image pixels) are different.
  - waiting 100ms before taking screenshot
  - taking page screenshot
    - disabled all CSS animations
  - waiting for fonts to load...
  - fonts loaded
  - captured a stable screenshot
  - 334615 pixels (ratio 0.10 of all image pixels) are different.

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
      - 'link "Czas CPU: 50.54ms (52.52ms)" [ref=e122] [cursor=pointer]':
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
      - link "SQL 5 queries in 1.58ms" [ref=e134] [cursor=pointer]:
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
  5  | for (const [name, size] of Object.entries(VIEWPORTS)) {
  6  |   test.describe(`${name} (${size.width}px)`, () => {
  7  |     test.use({ viewport: size });
  8  | 
  9  |     for (const page of PAGES) {
  10 |       test(`${page.name} - visual baseline`, async ({ page: pw }) => {
  11 |         await pw.goto(page.path);
  12 |         await pw.waitForLoadState('networkidle');
> 13 |         await expect(pw).toHaveScreenshot(`${page.name}-${name}.png`, {
     |                          ^ Error: expect(page).toHaveScreenshot(expected) failed
  14 |           maxDiffPixelRatio: 0.001,
  15 |           fullPage: true,
  16 |         });
  17 |       });
  18 |     }
  19 |   });
  20 | }
  21 | 
```