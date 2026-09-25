# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: visual_baseline.spec.ts >> mobile (375px) >> account-logout - visual baseline
- Location: tests/bs3-bs5_migration/visual_baseline.spec.ts:10:11

# Error details

```
Error: expect(page).toHaveScreenshot(expected) failed

  168652 pixels (ratio 0.56 of all image pixels) are different.

  Snapshot: account-logout-mobile.png

Call log:
  - Expect "toHaveScreenshot(account-logout-mobile.png)" with timeout 5000ms
    - verifying given screenshot expectation
  - taking page screenshot
    - disabled all CSS animations
  - waiting for fonts to load...
  - fonts loaded
  - 168652 pixels (ratio 0.56 of all image pixels) are different.
  - waiting 100ms before taking screenshot
  - taking page screenshot
    - disabled all CSS animations
  - waiting for fonts to load...
  - fonts loaded
  - captured a stable screenshot
  - 168652 pixels (ratio 0.56 of all image pixels) are different.

```

# Page snapshot

```yaml
- generic [active] [ref=e1]:
  - navigation [ref=e2]:
    - generic [ref=e3]:
      - button "Przełącz nawigacje" [ref=e4] [cursor=pointer]
      - link [ref=e7] [cursor=pointer]:
        - /url: /
        - img "Fedrowanie" [ref=e8]
      - heading "Obywatelskie fedrowanie danych" [level=1] [ref=e9]
  - generic [ref=e10]:
    - text: )
    - generic [ref=e11]:
      - generic [ref=e12]:
        - heading "DEV" [level=1] [ref=e13]
        - link [ref=e14] [cursor=pointer]:
          - /url: /
          - img "Fedrowanie" [ref=e16]
          - paragraph [ref=e17]: Fedrowanie
        - paragraph [ref=e18]:
          - link "Sieci Watchdog" [ref=e19] [cursor=pointer]:
            - /url: http://siecobywatelska.pl
        - paragraph [ref=e20]:
          - link "Klauzula RODO" [ref=e21] [cursor=pointer]:
            - /url: https://fedrowanie.siecobywatelska.pl/media_internal/tinycontent/uploads/11_KLAUZULA.-.dane.z.udip.na.stronach.www.i.w.mediach.spol.pdf
      - list [ref=e23]:
        - listitem [ref=e24]:
          - paragraph [ref=e25]
        - listitem [ref=e26]:
          - link "Strona główna" [ref=e27] [cursor=pointer]:
            - /url: /
        - listitem [ref=e29]:
          - link "O stronie" [ref=e30] [cursor=pointer]:
            - /url: /o-stronie/
        - listitem [ref=e32]:
          - paragraph [ref=e33]
        - generic [ref=e34]: Szukaj
        - listitem [ref=e36]:
          - link "Sprawy" [ref=e37] [cursor=pointer]:
            - /url: /sprawy/
        - listitem [ref=e39]:
          - link "Monitoringi" [ref=e40] [cursor=pointer]:
            - /url: /monitoringi/
        - listitem [ref=e42]:
          - link "Tabela monitoringów" [ref=e43] [cursor=pointer]:
            - /url: /monitoringi/table/
        - listitem [ref=e45]:
          - link "Listy przypisane do spraw" [ref=e46] [cursor=pointer]:
            - /url: /listy/
        - listitem [ref=e48]:
          - link "Listy nieprzypisane do spraw" [ref=e49] [cursor=pointer]:
            - /url: /listy/przypisz
        - listitem [ref=e51]:
          - link "Instytucje" [ref=e52] [cursor=pointer]:
            - /url: /instytucje/
        - listitem [ref=e54]:
          - paragraph [ref=e55]
      - generic [ref=e56]:
        - generic [ref=e57]: Użytkownik / użytkowniczka
        - listitem [ref=e58]:
          - link "Mój profil" [ref=e59] [cursor=pointer]:
            - /url: /uzytkownik/claude_ai/
        - listitem [ref=e61]:
          - link "Panel administracyjny" [ref=e62] [cursor=pointer]:
            - /url: /admin/
        - listitem [ref=e64]:
          - link "Wyloguj" [ref=e65] [cursor=pointer]:
            - /url: /accounts/logout/
    - generic [ref=e68]:
      - generic [ref=e71]:
        - heading "Wyloguj się" [level=2] [ref=e72]
        - paragraph [ref=e73]: Jesteś pewny, że chcesz się wylogować ?
        - button "Wyloguj się" [ref=e75] [cursor=pointer]
      - generic [ref=e76]:
        - generic [ref=e77]:
          - text: Sieć Obywatelska - Watchdog Polska ul. Szpitalna 5/5 00-031 Warszawa
          - paragraph [ref=e78]:
            - link "Klauzula RODO" [ref=e79] [cursor=pointer]:
              - /url: https://fedrowanie.siecobywatelska.pl/media_internal/tinycontent/uploads/11_KLAUZULA.-.dane.z.udip.na.stronach.www.i.w.mediach.spol.pdf
        - generic [ref=e80]:
          - text: "silnik:"
          - link "jawne.info.pl" [ref=e81] [cursor=pointer]:
            - /url: http://jawne.info.pl
          - text: "| v1.5.77.deps |"
          - link "GitHub" [ref=e82] [cursor=pointer]:
            - /url: https://github.com/watchdogpolska/feder
          - text: –
          - link "efd8a3b" [ref=e84] [cursor=pointer]:
            - /url: https://github.com/watchdogpolska/feder/compare/efd8a3b2...master
          - text: "|"
          - link "API" [ref=e85] [cursor=pointer]:
            - /url: /api/
        - generic [ref=e87]: Ta strona wykorzystuje cookies.
  - list [ref=e89]:
    - listitem [ref=e90]:
      - link "Ukryj »" [ref=e91] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e92]:
      - link "Toggle Theme" [ref=e93] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e96]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e97]
      - link "Historia /accounts/logout/" [ref=e98] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e99]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e100]
      - link "Wersje Django 5.2.17" [ref=e101] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e102]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e103]
      - 'link "Czas CPU: 56.19ms (57.67ms)" [ref=e104] [cursor=pointer]':
        - /url: "#"
    - listitem [ref=e105]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e106]
      - link "Ustawienia" [ref=e107] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e108]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e109]
      - link "Nagłówki" [ref=e110] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e111]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e112]
      - link "Zapytania LogoutView" [ref=e113] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e114]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e115]
      - link "SQL 4 queries in 1.01ms" [ref=e116] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e117]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e118]
      - link "Pliki statyczne 3 użyte plików" [ref=e119] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e120]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e121]
      - link "Templatki account/logout.html" [ref=e122] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e123]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e124]
      - link "Alerty" [ref=e125] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e126]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e127]
      - link "Cache 2 wywołania w 0.22ms" [ref=e128] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e129]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e130]
      - link "Sygnały 88 odbiorców 15 sygnałów" [ref=e131] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e132]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e133]
      - link "Gmina" [ref=e134] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e135]:
      - checkbox "Enable for next and successive requests" [ref=e136]
      - generic [ref=e137]: Przechwycone przekierowania
    - listitem [ref=e138]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e139]
      - link "Profilowanie" [ref=e140] [cursor=pointer]:
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