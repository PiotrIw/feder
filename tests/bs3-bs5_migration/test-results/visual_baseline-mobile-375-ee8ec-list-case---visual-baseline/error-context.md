# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: visual_baseline.spec.ts >> mobile (375px) >> letters-logs-list-case - visual baseline
- Location: tests/bs3-bs5_migration/visual_baseline.spec.ts:10:11

# Error details

```
Error: expect(page).toHaveScreenshot(expected) failed

  Expected an image 386px by 855px, received 383px by 812px. 170740 pixels (ratio 0.52 of all image pixels) are different.

  Snapshot: letters-logs-list-case-mobile.png

Call log:
  - Expect "toHaveScreenshot(letters-logs-list-case-mobile.png)" with timeout 5000ms
    - verifying given screenshot expectation
  - taking page screenshot
    - disabled all CSS animations
  - waiting for fonts to load...
  - fonts loaded
  - Expected an image 386px by 855px, received 383px by 812px. 170740 pixels (ratio 0.52 of all image pixels) are different.
  - waiting 100ms before taking screenshot
  - taking page screenshot
    - disabled all CSS animations
  - waiting for fonts to load...
  - fonts loaded
  - captured a stable screenshot
  - Expected an image 386px by 855px, received 383px by 812px. 170740 pixels (ratio 0.52 of all image pixels) are different.

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
      - list [ref=e69]:
        - listitem [ref=e70]:
          - link "Monitoring sądów apelacyjnych" [ref=e71] [cursor=pointer]:
            - /url: /monitoringi/monitoring-sadow-apelacyjnych
        - listitem [ref=e72]:
          - text: /
          - 'link "Monitoring sądów apelacyjnych #1" [ref=e73] [cursor=pointer]':
            - /url: /sprawy/monitoring-sadow-apelacyjnych-1
        - listitem [ref=e74]: / Dziennik
      - generic [ref=e76]:
        - link "Edytuj" [ref=e77] [cursor=pointer]:
          - /url: /sprawy/monitoring-sadow-apelacyjnych-1/~edytuj
        - link "Usuń" [ref=e78] [cursor=pointer]:
          - /url: /sprawy/monitoring-sadow-apelacyjnych-1/~usun
        - link "Zobacz dzienniki" [ref=e79] [cursor=pointer]:
          - /url: /listy/logi/spraw-2684
        - button "Dodaj przesyłkę pocztową" [ref=e81] [cursor=pointer]
        - link "Dodaj list" [ref=e83] [cursor=pointer]:
          - /url: /listy/~utworz-2684
      - 'heading "Monitoring sądów apelacyjnych #1" [level=1] [ref=e86]'
      - generic [ref=e88]:
        - table [ref=e89]:
          - rowgroup [ref=e90]:
            - row [ref=e91]:
              - columnheader "ID" [ref=e92]
              - columnheader "Sprawa" [ref=e93]
              - columnheader "Status" [ref=e94]
              - columnheader "List" [ref=e95]
              - columnheader "Liczba wpisów" [ref=e96]
          - rowgroup
        - list [ref=e97]:
          - listitem [ref=e98]:
            - generic [aria-hidden]: ←
          - listitem [ref=e99]:
            - generic "Current Page" [ref=e100]: "1"
          - listitem [ref=e101]:
            - generic [aria-hidden]: →
      - generic [ref=e102]:
        - generic [ref=e103]:
          - text: Sieć Obywatelska - Watchdog Polska ul. Szpitalna 5/5 00-031 Warszawa
          - paragraph [ref=e104]:
            - link "Klauzula RODO" [ref=e105] [cursor=pointer]:
              - /url: https://fedrowanie.siecobywatelska.pl/media_internal/tinycontent/uploads/11_KLAUZULA.-.dane.z.udip.na.stronach.www.i.w.mediach.spol.pdf
        - generic [ref=e106]:
          - text: "silnik:"
          - link "jawne.info.pl" [ref=e107] [cursor=pointer]:
            - /url: http://jawne.info.pl
          - text: "| v1.5.77.deps |"
          - link "GitHub" [ref=e108] [cursor=pointer]:
            - /url: https://github.com/watchdogpolska/feder
          - text: –
          - link "efd8a3b" [ref=e110] [cursor=pointer]:
            - /url: https://github.com/watchdogpolska/feder/compare/efd8a3b2...master
          - text: "|"
          - link "API" [ref=e111] [cursor=pointer]:
            - /url: /api/
        - generic [ref=e113]: Ta strona wykorzystuje cookies.
  - list [ref=e115]:
    - listitem [ref=e116]:
      - link "Ukryj »" [ref=e117] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e118]:
      - link "Toggle Theme" [ref=e119] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e122]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e123]
      - link "Historia /listy/logi/spraw-2684" [ref=e124] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e125]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e126]
      - link "Wersje Django 5.2.17" [ref=e127] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e128]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e129]
      - 'link "Czas CPU: 86.51ms (92.63ms)" [ref=e130] [cursor=pointer]':
        - /url: "#"
    - listitem [ref=e131]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e132]
      - link "Ustawienia" [ref=e133] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e134]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e135]
      - link "Nagłówki" [ref=e136] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e137]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e138]
      - link "Zapytania EmailLogCaseListView" [ref=e139] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e140]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e141]
      - link "SQL 8 queries in 6.17ms" [ref=e142] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e143]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e144]
      - link "Pliki statyczne 3 użyte plików" [ref=e145] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e146]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e147]
      - link "Templatki logs/emaillog_list_for_case.html" [ref=e148] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e149]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e150]
      - link "Alerty" [ref=e151] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e152]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e153]
      - link "Cache 2 wywołania w 0.12ms" [ref=e154] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e155]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e156]
      - link "Sygnały 88 odbiorców 15 sygnałów" [ref=e157] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e158]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e159]
      - link "Gmina" [ref=e160] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e161]:
      - checkbox "Enable for next and successive requests" [ref=e162]
      - generic [ref=e163]: Przechwycone przekierowania
    - listitem [ref=e164]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e165]
      - link "Profilowanie" [ref=e166] [cursor=pointer]:
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