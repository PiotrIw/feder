# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: visual_baseline.spec.ts >> desktop (2560px) >> account-reset-password-done - visual baseline
- Location: tests/bs3-bs5_migration/visual_baseline.spec.ts:10:11

# Error details

```
Error: expect(page).toHaveScreenshot(expected) failed

  332747 pixels (ratio 0.10 of all image pixels) are different.

  Snapshot: account-reset-password-done-desktop.png

Call log:
  - Expect "toHaveScreenshot(account-reset-password-done-desktop.png)" with timeout 5000ms
    - verifying given screenshot expectation
  - taking page screenshot
    - disabled all CSS animations
  - waiting for fonts to load...
  - fonts loaded
  - 332747 pixels (ratio 0.10 of all image pixels) are different.
  - waiting 100ms before taking screenshot
  - taking page screenshot
    - disabled all CSS animations
  - waiting for fonts to load...
  - fonts loaded
  - captured a stable screenshot
  - 332747 pixels (ratio 0.10 of all image pixels) are different.

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
      - generic [ref=e63]:
        - heading "Resetowanie hasła" [level=2] [ref=e64]
        - paragraph [ref=e65]:
          - strong [ref=e66]: "Uwaga:"
          - text: Jesteś już zalogowany/-a jako claude_ai.
        - paragraph [ref=e67]: Wysłaliśmy Tobie e-mail. Proszę skontaktuj się z nami, jeśli go nie otrzymasz w ciągu paru minut.
      - generic [ref=e68]:
        - generic [ref=e69]:
          - text: Sieć Obywatelska - Watchdog Polska ul. Szpitalna 5/5 00-031 Warszawa
          - paragraph [ref=e70]:
            - link "Klauzula RODO" [ref=e71] [cursor=pointer]:
              - /url: https://fedrowanie.siecobywatelska.pl/media_internal/tinycontent/uploads/11_KLAUZULA.-.dane.z.udip.na.stronach.www.i.w.mediach.spol.pdf
        - generic [ref=e72]:
          - text: "silnik:"
          - link "jawne.info.pl" [ref=e73] [cursor=pointer]:
            - /url: http://jawne.info.pl
          - text: "| v1.5.77.deps |"
          - link "GitHub" [ref=e74] [cursor=pointer]:
            - /url: https://github.com/watchdogpolska/feder
          - text: –
          - link "efd8a3b" [ref=e76] [cursor=pointer]:
            - /url: https://github.com/watchdogpolska/feder/compare/efd8a3b2...master
          - text: "|"
          - link "API" [ref=e77] [cursor=pointer]:
            - /url: /api/
        - generic [ref=e79]: Ta strona wykorzystuje cookies.
  - list [ref=e81]:
    - listitem [ref=e82]:
      - link "Ukryj »" [ref=e83] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e84]:
      - link "Toggle Theme" [ref=e85] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e88]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e89]
      - link "Historia /accounts/password/reset/done/" [ref=e90] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e91]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e92]
      - link "Wersje Django 5.2.17" [ref=e93] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e94]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e95]
      - 'link "Czas CPU: 60.62ms (62.08ms)" [ref=e96] [cursor=pointer]':
        - /url: "#"
    - listitem [ref=e97]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e98]
      - link "Ustawienia" [ref=e99] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e100]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e101]
      - link "Nagłówki" [ref=e102] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e103]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e104]
      - link "Zapytania PasswordResetDoneView" [ref=e105] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e106]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e107]
      - link "SQL 4 queries in 1.12ms" [ref=e108] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e109]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e110]
      - link "Pliki statyczne 3 użyte plików" [ref=e111] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e112]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e113]
      - link "Templatki account/password_reset_done.html" [ref=e114] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e115]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e116]
      - link "Alerty" [ref=e117] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e118]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e119]
      - link "Cache 2 wywołania w 0.13ms" [ref=e120] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e121]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e122]
      - link "Sygnały 88 odbiorców 15 sygnałów" [ref=e123] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e124]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e125]
      - link "Gmina" [ref=e126] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e127]:
      - checkbox "Enable for next and successive requests" [ref=e128]
      - generic [ref=e129]: Przechwycone przekierowania
    - listitem [ref=e130]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e131]
      - link "Profilowanie" [ref=e132] [cursor=pointer]:
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