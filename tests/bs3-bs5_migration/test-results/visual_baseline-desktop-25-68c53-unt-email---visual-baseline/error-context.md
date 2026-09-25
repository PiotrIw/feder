# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: visual_baseline.spec.ts >> desktop (2560px) >> account-email - visual baseline
- Location: tests/bs3-bs5_migration/visual_baseline.spec.ts:10:11

# Error details

```
Error: expect(page).toHaveScreenshot(expected) failed

  347982 pixels (ratio 0.10 of all image pixels) are different.

  Snapshot: account-email-desktop.png

Call log:
  - Expect "toHaveScreenshot(account-email-desktop.png)" with timeout 5000ms
    - verifying given screenshot expectation
  - taking page screenshot
    - disabled all CSS animations
  - waiting for fonts to load...
  - fonts loaded
  - 347982 pixels (ratio 0.10 of all image pixels) are different.
  - waiting 100ms before taking screenshot
  - taking page screenshot
    - disabled all CSS animations
  - waiting for fonts to load...
  - fonts loaded
  - captured a stable screenshot
  - 347982 pixels (ratio 0.10 of all image pixels) are different.

```

# Page snapshot

```yaml
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
  - generic [ref=e61]:
    - generic [ref=e62]:
      - generic [ref=e64]:
        - heading "Adresy e-mail" [level=2] [ref=e65]
        - paragraph [ref=e66]: "Poniższe adresy e-mail są powiązane z Twoim kontem:"
        - group [ref=e68]:
          - generic [ref=e69]:
            - radio "claude_ai@example.com Zweryfikowany Podstawowy" [checked] [ref=e70]
            - generic [ref=e71]: claude_ai@example.com Zweryfikowany Podstawowy
          - generic [ref=e72]:
            - button "Uczyń podstawowym" [ref=e73] [cursor=pointer]
            - button "Prześlij ponownie wiadomość weryfikacyjną" [ref=e74] [cursor=pointer]
            - button "Usuń" [ref=e75] [cursor=pointer]
      - generic [ref=e77]:
        - heading "Dodaj adres e-mail" [level=2] [ref=e78]
        - generic [ref=e79]:
          - generic [ref=e80]:
            - generic [ref=e81]: Adres e-mail*
            - textbox "Adres e-mail*" [ref=e82]:
              - /placeholder: Adres e-mail
          - button "Dodaj e-mail" [ref=e83] [cursor=pointer]
    - generic [ref=e84]:
      - generic [ref=e85]:
        - text: Sieć Obywatelska - Watchdog Polska ul. Szpitalna 5/5 00-031 Warszawa
        - paragraph [ref=e86]:
          - link "Klauzula RODO" [ref=e87] [cursor=pointer]:
            - /url: https://fedrowanie.siecobywatelska.pl/media_internal/tinycontent/uploads/11_KLAUZULA.-.dane.z.udip.na.stronach.www.i.w.mediach.spol.pdf
      - generic [ref=e88]:
        - text: "silnik:"
        - link "jawne.info.pl" [ref=e89] [cursor=pointer]:
          - /url: http://jawne.info.pl
        - text: "| v1.5.77.deps |"
        - link "GitHub" [ref=e90] [cursor=pointer]:
          - /url: https://github.com/watchdogpolska/feder
        - text: –
        - link "efd8a3b" [ref=e92] [cursor=pointer]:
          - /url: https://github.com/watchdogpolska/feder/compare/efd8a3b2...master
        - text: "|"
        - link "API" [ref=e93] [cursor=pointer]:
          - /url: /api/
      - generic [ref=e95]: Ta strona wykorzystuje cookies.
  - list [ref=e97]:
    - listitem [ref=e98]:
      - link "Ukryj »" [ref=e99] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e100]:
      - link "Toggle Theme" [ref=e101] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e104]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e105]
      - link "Historia /accounts/email/" [ref=e106] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e107]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e108]
      - link "Wersje Django 5.2.17" [ref=e109] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e110]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e111]
      - 'link "Czas CPU: 234.60ms (237.37ms)" [ref=e112] [cursor=pointer]':
        - /url: "#"
    - listitem [ref=e113]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e114]
      - link "Ustawienia" [ref=e115] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e116]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e117]
      - link "Nagłówki" [ref=e118] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e119]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e120]
      - link "Zapytania EmailView" [ref=e121] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e122]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e123]
      - link "SQL 8 queries in 2.97ms" [ref=e124] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e125]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e126]
      - link "Pliki statyczne 3 użyte plików" [ref=e127] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e128]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e129]
      - link "Templatki account/email.html" [ref=e130] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e131]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e132]
      - link "Alerty" [ref=e133] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e134]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e135]
      - link "Cache 2 wywołania w 0.13ms" [ref=e136] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e137]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e138]
      - link "Sygnały 88 odbiorców 15 sygnałów" [ref=e139] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e140]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e141]
      - link "Gmina" [ref=e142] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e143]:
      - checkbox "Enable for next and successive requests" [ref=e144]
      - generic [ref=e145]: Przechwycone przekierowania
    - listitem [ref=e146]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e147]
      - link "Profilowanie" [ref=e148] [cursor=pointer]:
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