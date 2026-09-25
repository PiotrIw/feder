# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: visual_baseline.spec.ts >> desktop (2560px) >> users-redirect - visual baseline
- Location: tests/bs3-bs5_migration/visual_baseline.spec.ts:10:11

# Error details

```
Error: expect(page).toHaveScreenshot(expected) failed

  337473 pixels (ratio 0.10 of all image pixels) are different.

  Snapshot: users-redirect-desktop.png

Call log:
  - Expect "toHaveScreenshot(users-redirect-desktop.png)" with timeout 5000ms
    - verifying given screenshot expectation
  - taking page screenshot
    - disabled all CSS animations
  - waiting for fonts to load...
  - fonts loaded
  - 337473 pixels (ratio 0.10 of all image pixels) are different.
  - waiting 100ms before taking screenshot
  - taking page screenshot
    - disabled all CSS animations
  - waiting for fonts to load...
  - fonts loaded
  - captured a stable screenshot
  - 337473 pixels (ratio 0.10 of all image pixels) are different.

```

# Page snapshot

```yaml
- generic [active] [ref=e1]:
  - generic [ref=e2]:
    - text: )
    - generic [ref=e3]:
      - generic [ref=e4]:
        - heading "DEV" [level=1] [ref=e5]
        - link [ref=e6] [cursor=pointer]:
          - /url: /
          - img "Fedrowanie" [ref=e8]
          - paragraph [ref=e9]: Fedrowanie
        - paragraph [ref=e10]:
          - link "Sieci Watchdog" [ref=e11] [cursor=pointer]:
            - /url: http://siecobywatelska.pl
        - paragraph [ref=e12]:
          - link "Klauzula RODO" [ref=e13] [cursor=pointer]:
            - /url: https://fedrowanie.siecobywatelska.pl/media_internal/tinycontent/uploads/11_KLAUZULA.-.dane.z.udip.na.stronach.www.i.w.mediach.spol.pdf
      - list [ref=e15]:
        - listitem [ref=e16]:
          - paragraph [ref=e17]
        - listitem [ref=e18]:
          - link "Strona główna" [ref=e19] [cursor=pointer]:
            - /url: /
        - listitem [ref=e21]:
          - link "O stronie" [ref=e22] [cursor=pointer]:
            - /url: /o-stronie/
        - listitem [ref=e24]:
          - paragraph [ref=e25]
        - generic [ref=e26]: Szukaj
        - listitem [ref=e28]:
          - link "Sprawy" [ref=e29] [cursor=pointer]:
            - /url: /sprawy/
        - listitem [ref=e31]:
          - link "Monitoringi" [ref=e32] [cursor=pointer]:
            - /url: /monitoringi/
        - listitem [ref=e34]:
          - link "Tabela monitoringów" [ref=e35] [cursor=pointer]:
            - /url: /monitoringi/table/
        - listitem [ref=e37]:
          - link "Listy przypisane do spraw" [ref=e38] [cursor=pointer]:
            - /url: /listy/
        - listitem [ref=e40]:
          - link "Listy nieprzypisane do spraw" [ref=e41] [cursor=pointer]:
            - /url: /listy/przypisz
        - listitem [ref=e43]:
          - link "Instytucje" [ref=e44] [cursor=pointer]:
            - /url: /instytucje/
        - listitem [ref=e46]:
          - paragraph [ref=e47]
      - generic [ref=e48]:
        - generic [ref=e49]: Użytkownik / użytkowniczka
        - listitem [ref=e50]:
          - link "Mój profil" [ref=e51] [cursor=pointer]:
            - /url: /uzytkownik/claude_ai/
        - listitem [ref=e53]:
          - link "Panel administracyjny" [ref=e54] [cursor=pointer]:
            - /url: /admin/
        - listitem [ref=e56]:
          - link "Wyloguj" [ref=e57] [cursor=pointer]:
            - /url: /accounts/logout/
    - generic [ref=e60]:
      - generic [ref=e61]:
        - heading "claude_ai" [level=2] [ref=e64]
        - generic [ref=e66]:
          - link "O mnie" [ref=e67] [cursor=pointer]:
            - /url: /uzytkownik/~aktualizuj/
          - link "E-mail" [ref=e68] [cursor=pointer]:
            - /url: /accounts/email/
          - link "Zmiana hasła" [ref=e69] [cursor=pointer]:
            - /url: /accounts/password/change/
          - link "Połącz konto Google" [ref=e70] [cursor=pointer]:
            - /url: /accounts/google/login/?process=connect
      - generic [ref=e72]:
        - generic [ref=e73]:
          - text: Sieć Obywatelska - Watchdog Polska ul. Szpitalna 5/5 00-031 Warszawa
          - paragraph [ref=e74]:
            - link "Klauzula RODO" [ref=e75] [cursor=pointer]:
              - /url: https://fedrowanie.siecobywatelska.pl/media_internal/tinycontent/uploads/11_KLAUZULA.-.dane.z.udip.na.stronach.www.i.w.mediach.spol.pdf
        - generic [ref=e76]:
          - text: "silnik:"
          - link "jawne.info.pl" [ref=e77] [cursor=pointer]:
            - /url: http://jawne.info.pl
          - text: "| v1.5.77.deps |"
          - link "GitHub" [ref=e78] [cursor=pointer]:
            - /url: https://github.com/watchdogpolska/feder
          - text: –
          - link "efd8a3b" [ref=e80] [cursor=pointer]:
            - /url: https://github.com/watchdogpolska/feder/compare/efd8a3b2...master
          - text: "|"
          - link "API" [ref=e81] [cursor=pointer]:
            - /url: /api/
        - generic [ref=e83]: Ta strona wykorzystuje cookies.
  - list [ref=e85]:
    - listitem [ref=e86]:
      - link "Ukryj »" [ref=e87] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e88]:
      - link "Toggle Theme" [ref=e89] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e92]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e93]
      - link "Historia /uzytkownik/claude_ai/" [ref=e94] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e95]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e96]
      - link "Wersje Django 5.2.17" [ref=e97] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e98]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e99]
      - 'link "Czas CPU: 77.91ms (80.62ms)" [ref=e100] [cursor=pointer]':
        - /url: "#"
    - listitem [ref=e101]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e102]
      - link "Ustawienia" [ref=e103] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e104]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e105]
      - link "Nagłówki" [ref=e106] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e107]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e108]
      - link "Zapytania UserDetailView" [ref=e109] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e110]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e111]
      - link "SQL 7 queries in 2.96ms" [ref=e112] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e113]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e114]
      - link "Pliki statyczne 3 użyte plików" [ref=e115] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e116]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e117]
      - link "Templatki users/user_detail.html" [ref=e118] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e119]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e120]
      - link "Alerty" [ref=e121] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e122]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e123]
      - link "Cache 2 wywołania w 0.14ms" [ref=e124] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e125]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e126]
      - link "Sygnały 88 odbiorców 15 sygnałów" [ref=e127] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e128]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e129]
      - link "Gmina" [ref=e130] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e131]:
      - checkbox "Enable for next and successive requests" [ref=e132]
      - generic [ref=e133]: Przechwycone przekierowania
    - listitem [ref=e134]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e135]
      - link "Profilowanie" [ref=e136] [cursor=pointer]:
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