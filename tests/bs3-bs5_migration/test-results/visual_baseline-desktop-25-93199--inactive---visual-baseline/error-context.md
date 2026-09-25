# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: visual_baseline.spec.ts >> desktop (2560px) >> account-inactive - visual baseline
- Location: tests/bs3-bs5_migration/visual_baseline.spec.ts:10:11

# Error details

```
Error: expect(page).toHaveScreenshot(expected) failed

  305942 pixels (ratio 0.09 of all image pixels) are different.

  Snapshot: account-inactive-desktop.png

Call log:
  - Expect "toHaveScreenshot(account-inactive-desktop.png)" with timeout 5000ms
    - verifying given screenshot expectation
  - taking page screenshot
    - disabled all CSS animations
  - waiting for fonts to load...
  - fonts loaded
  - 305942 pixels (ratio 0.09 of all image pixels) are different.
  - waiting 100ms before taking screenshot
  - taking page screenshot
    - disabled all CSS animations
  - waiting for fonts to load...
  - fonts loaded
  - captured a stable screenshot
  - 305942 pixels (ratio 0.09 of all image pixels) are different.

```

# Page snapshot

```yaml
- generic [active] [ref=e1]:
  - generic [ref=e2]:
    - strong [ref=e3]: "Menu:"
    - list [ref=e4]:
      - listitem [ref=e5]:
        - link "Zmień e-mail" [ref=e6] [cursor=pointer]:
          - /url: /accounts/email/
      - listitem [ref=e7]:
        - link "Zmień hasło" [ref=e8] [cursor=pointer]:
          - /url: /accounts/password/change/
      - listitem [ref=e9]:
        - link "Połączone konta" [ref=e10] [cursor=pointer]:
          - /url: /accounts/3rdparty/
      - listitem [ref=e11]:
        - link "Wyloguj się" [ref=e12] [cursor=pointer]:
          - /url: /accounts/logout/
  - heading "Konto nieaktywne" [level=1] [ref=e13]
  - paragraph [ref=e14]: To konto jest nieaktywne.
  - list [ref=e16]:
    - listitem [ref=e17]:
      - link "Ukryj »" [ref=e18] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e19]:
      - link "Toggle Theme" [ref=e20] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e23]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e24]
      - link "Historia /accounts/inactive/" [ref=e25] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e26]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e27]
      - link "Wersje Django 5.2.17" [ref=e28] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e29]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e30]
      - 'link "Czas CPU: 41.31ms (42.57ms)" [ref=e31] [cursor=pointer]':
        - /url: "#"
    - listitem [ref=e32]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e33]
      - link "Ustawienia" [ref=e34] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e35]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e36]
      - link "Nagłówki" [ref=e37] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e38]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e39]
      - link "Zapytania AccountInactiveView" [ref=e40] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e41]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e42]
      - link "SQL 4 queries in 1.15ms" [ref=e43] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e44]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e45]
      - link "Pliki statyczne 0 użytych plików" [ref=e46] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e47]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e48]
      - link "Templatki account/account_inactive.html" [ref=e49] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e50]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e51]
      - link "Alerty" [ref=e52] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e53]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e54]
      - link "Cache 0 wywołań w 0.00ms" [ref=e55] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e56]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e57]
      - link "Sygnały 88 odbiorców 15 sygnałów" [ref=e58] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e59]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e60]
      - link "Gmina" [ref=e61] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e62]:
      - checkbox "Enable for next and successive requests" [ref=e63]
      - generic [ref=e64]: Przechwycone przekierowania
    - listitem [ref=e65]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e66]
      - link "Profilowanie" [ref=e67] [cursor=pointer]:
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