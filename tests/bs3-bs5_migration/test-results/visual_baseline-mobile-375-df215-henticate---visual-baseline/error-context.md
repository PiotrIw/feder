# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: visual_baseline.spec.ts >> mobile (375px) >> account-reauthenticate - visual baseline
- Location: tests/bs3-bs5_migration/visual_baseline.spec.ts:10:11

# Error details

```
Error: expect(page).toHaveScreenshot(expected) failed

  168719 pixels (ratio 0.56 of all image pixels) are different.

  Snapshot: account-reauthenticate-mobile.png

Call log:
  - Expect "toHaveScreenshot(account-reauthenticate-mobile.png)" with timeout 5000ms
    - verifying given screenshot expectation
  - taking page screenshot
    - disabled all CSS animations
  - waiting for fonts to load...
  - fonts loaded
  - 168719 pixels (ratio 0.56 of all image pixels) are different.
  - waiting 100ms before taking screenshot
  - taking page screenshot
    - disabled all CSS animations
  - waiting for fonts to load...
  - fonts loaded
  - captured a stable screenshot
  - 168719 pixels (ratio 0.56 of all image pixels) are different.

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
  - heading "Potwierdź dostęp" [level=1] [ref=e13]
  - paragraph [ref=e14]: Aby zabezpieczyć swoje konto, dokonaj ponownego uwierzytelnienia.
  - paragraph [ref=e15]: "Wprowadż hasło:"
  - generic [ref=e16]:
    - paragraph [ref=e17]:
      - text: "Hasło:"
      - textbox "Hasło:" [ref=e18]:
        - /placeholder: Hasło
      - link "Zapomniałeś hasła?" [ref=e20] [cursor=pointer]:
        - /url: /accounts/password/reset/
    - button "Potwierdź" [ref=e21]
  - list [ref=e23]:
    - listitem [ref=e24]:
      - link "Ukryj »" [ref=e25] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e26]:
      - link "Toggle Theme" [ref=e27] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e30]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e31]
      - link "Historia /accounts/reauthenticate/" [ref=e32] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e33]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e34]
      - link "Wersje Django 5.2.17" [ref=e35] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e36]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e37]
      - 'link "Czas CPU: 56.80ms (58.13ms)" [ref=e38] [cursor=pointer]':
        - /url: "#"
    - listitem [ref=e39]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e40]
      - link "Ustawienia" [ref=e41] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e42]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e43]
      - link "Nagłówki" [ref=e44] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e45]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e46]
      - link "Zapytania ReauthenticateView" [ref=e47] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e48]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e49]
      - link "SQL 4 queries in 0.85ms" [ref=e50] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e51]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e52]
      - link "Pliki statyczne 0 użytych plików" [ref=e53] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e54]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e55]
      - link "Templatki account/reauthenticate.html" [ref=e56] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e57]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e58]
      - link "Alerty" [ref=e59] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e60]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e61]
      - link "Cache 0 wywołań w 0.00ms" [ref=e62] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e63]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e64]
      - link "Sygnały 88 odbiorców 15 sygnałów" [ref=e65] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e66]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e67]
      - link "Gmina" [ref=e68] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e69]:
      - checkbox "Enable for next and successive requests" [ref=e70]
      - generic [ref=e71]: Przechwycone przekierowania
    - listitem [ref=e72]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e73]
      - link "Profilowanie" [ref=e74] [cursor=pointer]:
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