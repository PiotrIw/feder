# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: visual_baseline.spec.ts >> desktop (2560px) >> letters-delete - visual baseline
- Location: tests/bs3-bs5_migration/visual_baseline.spec.ts:10:11

# Error details

```
Error: expect(page).toHaveScreenshot(expected) failed

  354271 pixels (ratio 0.10 of all image pixels) are different.

  Snapshot: letters-delete-desktop.png

Call log:
  - Expect "toHaveScreenshot(letters-delete-desktop.png)" with timeout 5000ms
    - verifying given screenshot expectation
  - taking page screenshot
    - disabled all CSS animations
  - waiting for fonts to load...
  - fonts loaded
  - 354271 pixels (ratio 0.10 of all image pixels) are different.
  - waiting 100ms before taking screenshot
  - taking page screenshot
    - disabled all CSS animations
  - waiting for fonts to load...
  - fonts loaded
  - captured a stable screenshot
  - 354271 pixels (ratio 0.10 of all image pixels) are different.

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
      - list [ref=e61]:
        - listitem [ref=e62]:
          - link "Monitoring sądów apelacyjnych" [ref=e63] [cursor=pointer]:
            - /url: /monitoringi/monitoring-sadow-apelacyjnych
        - listitem [ref=e64]:
          - text: /
          - 'link "Monitoring sądów apelacyjnych #1" [ref=e65] [cursor=pointer]':
            - /url: /sprawy/monitoring-sadow-apelacyjnych-1
        - listitem [ref=e66]:
          - text: /
          - link "Wniosek o udostępnienie informacji publicznej" [ref=e67] [cursor=pointer]:
            - /url: /listy/7302
        - listitem [ref=e68]: / Potwierdź usuwanie
      - generic [ref=e72]:
        - link "Edytuj" [ref=e73] [cursor=pointer]:
          - /url: /listy/7302/~edytuj
        - link "Usuń" [ref=e74] [cursor=pointer]:
          - /url: /listy/7302/~usun
        - link "Odpowiedź" [ref=e75] [cursor=pointer]:
          - /url: /listy/7302/~odpowiedz
        - link "Wyślij ponownie" [ref=e77] [cursor=pointer]:
          - /url: /listy/7302/~resend
      - generic [ref=e80]:
        - heading [level=2] [ref=e82]:
          - link "Wniosek o udostępnienie informacji publicznej" [ref=e84] [cursor=pointer]:
            - /url: /listy/7302
          - generic [ref=e85]:
            - text: przez
            - link "adobrawy" [ref=e86] [cursor=pointer]:
              - /url: /uzytkownik/adobrawy/
            - time [ref=e87]: 11 sierpnia 2017 02:48
            - text: w sprawie
            - 'link "Monitoring sądów apelacyjnych #1" [ref=e88] [cursor=pointer]':
              - /url: /sprawy/monitoring-sadow-apelacyjnych-1
            - text: z
            - link "Sąd Apelacyjny w Białymstoku" [ref=e89] [cursor=pointer]:
              - /url: /instytucje/sad-apelacyjny-w-bialymstoku
        - generic [ref=e90]:
          - text: Czy na pewno chcesz usunąć "Wniosek o udostępnienie informacji publicznej"?
          - button "Usuń" [ref=e92] [cursor=pointer]
      - generic [ref=e93]:
        - generic [ref=e94]:
          - text: Sieć Obywatelska - Watchdog Polska ul. Szpitalna 5/5 00-031 Warszawa
          - paragraph [ref=e95]:
            - link "Klauzula RODO" [ref=e96] [cursor=pointer]:
              - /url: https://fedrowanie.siecobywatelska.pl/media_internal/tinycontent/uploads/11_KLAUZULA.-.dane.z.udip.na.stronach.www.i.w.mediach.spol.pdf
        - generic [ref=e97]:
          - text: "silnik:"
          - link "jawne.info.pl" [ref=e98] [cursor=pointer]:
            - /url: http://jawne.info.pl
          - text: "| v1.5.77.deps |"
          - link "GitHub" [ref=e99] [cursor=pointer]:
            - /url: https://github.com/watchdogpolska/feder
          - text: –
          - link "efd8a3b" [ref=e101] [cursor=pointer]:
            - /url: https://github.com/watchdogpolska/feder/compare/efd8a3b2...master
          - text: "|"
          - link "API" [ref=e102] [cursor=pointer]:
            - /url: /api/
        - generic [ref=e104]: Ta strona wykorzystuje cookies.
  - list [ref=e106]:
    - listitem [ref=e107]:
      - link "Ukryj »" [ref=e108] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e109]:
      - link "Toggle Theme" [ref=e110] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e113]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e114]
      - link "Historia /listy/7302/~usun" [ref=e115] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e116]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e117]
      - link "Wersje Django 5.2.17" [ref=e118] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e119]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e120]
      - 'link "Czas CPU: 98.43ms (102.17ms)" [ref=e121] [cursor=pointer]':
        - /url: "#"
    - listitem [ref=e122]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e123]
      - link "Ustawienia" [ref=e124] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e125]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e126]
      - link "Nagłówki" [ref=e127] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e128]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e129]
      - link "Zapytania LetterDeleteView" [ref=e130] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e131]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e132]
      - link "SQL 13 queries in 4.93ms" [ref=e133] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e134]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e135]
      - link "Pliki statyczne 3 użyte plików" [ref=e136] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e137]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e138]
      - link "Templatki letters/letter_confirm_delete.html" [ref=e139] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e140]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e141]
      - link "Alerty" [ref=e142] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e143]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e144]
      - link "Cache 2 wywołania w 0.14ms" [ref=e145] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e146]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e147]
      - link "Sygnały 88 odbiorców 15 sygnałów" [ref=e148] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e149]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e150]
      - link "Gmina" [ref=e151] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e152]:
      - checkbox "Enable for next and successive requests" [ref=e153]
      - generic [ref=e154]: Przechwycone przekierowania
    - listitem [ref=e155]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e156]
      - link "Profilowanie" [ref=e157] [cursor=pointer]:
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