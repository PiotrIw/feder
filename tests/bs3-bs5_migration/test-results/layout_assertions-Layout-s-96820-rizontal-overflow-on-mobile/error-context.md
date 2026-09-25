# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: layout_assertions.spec.ts >> Layout sanity - mobile >> letters-logs-list-monitoring - no horizontal overflow on mobile
- Location: tests/bs3-bs5_migration/layout_assertions.spec.ts:30:9

# Error details

```
Error: Mobile layout has horizontal scroll

expect(received).toBeLessThanOrEqual(expected)

Expected: <= 20
Received:    226
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
        - listitem [ref=e72]: / Dziennik
      - generic [ref=e74]:
        - link "Edytuj" [ref=e75] [cursor=pointer]:
          - /url: /monitoringi/monitoring-sadow-apelacyjnych/~edytuj
        - link "Aktualizuj wyniki" [ref=e76] [cursor=pointer]:
          - /url: /monitoringi/monitoring-sadow-apelacyjnych/~results-update
        - link "Przypisz" [ref=e77] [cursor=pointer]:
          - /url: /monitoringi/monitoring-sadow-apelacyjnych/~przypisz
        - link "Usuń" [ref=e78] [cursor=pointer]:
          - /url: /monitoringi/monitoring-sadow-apelacyjnych/~usun
        - link "Utwórz sprawę" [ref=e79] [cursor=pointer]:
          - /url: /sprawy/~utworz-5
        - link "Wiadomość masowa" [ref=e80] [cursor=pointer]:
          - /url: /monitoringi/monitoring-sadow-apelacyjnych/~wiadomosc-masowa
        - link "Uprawnienia" [ref=e81] [cursor=pointer]:
          - /url: /monitoringi/monitoring-sadow-apelacyjnych/~uprawnienia
        - link "Lista alertów" [ref=e82] [cursor=pointer]:
          - /url: /alerty/monitoring-5
        - link "Zobacz dzienniki" [ref=e83] [cursor=pointer]:
          - /url: /listy/logi/monitoring-5
        - link "Zobacz tagi" [ref=e85] [cursor=pointer]:
          - /url: /sprawy/tagi/monitoring-5
        - link "Zobacz raport" [ref=e87] [cursor=pointer]:
          - /url: /monitoringi/monitoring-sadow-apelacyjnych/raport
        - link "Zobacz tabelę spraw" [ref=e89] [cursor=pointer]:
          - /url: /monitoringi/monitoring-sadow-apelacyjnych/monitoring_cases_table
      - heading [level=2] [ref=e92]:
        - text: Monitoring sądów apelacyjnych
        - generic [ref=e94]:
          - text: przez
          - link "adobrawy" [ref=e95] [cursor=pointer]:
            - /url: /uzytkownik/adobrawy/
          - time [ref=e96]: 11 sierpnia 2017 02:47
      - link "Pobierz .csv" [ref=e98] [cursor=pointer]:
        - /url: /listy/logi/monitoring-5/eksport
      - generic [ref=e100]:
        - table [ref=e101]:
          - rowgroup [ref=e102]:
            - row [ref=e103]:
              - columnheader "ID" [ref=e104]
              - columnheader "Sprawa" [ref=e105]
              - columnheader "Status" [ref=e106]
              - columnheader "List" [ref=e107]
              - columnheader "Liczba wpisów" [ref=e108]
          - rowgroup [ref=e109]:
            - row [ref=e110]:
              - cell [ref=e111]:
                - link "59a127fa42cf33b253a32932" [ref=e112] [cursor=pointer]:
                  - /url: /listy/logi/wpis-430
              - cell [ref=e113]:
                - 'link "Monitoring sądów apelacyjnych #2" [ref=e114] [cursor=pointer]':
                  - /url: /sprawy/monitoring-sadow-apelacyjnych-2
              - cell "Dostarczony" [ref=e115]
              - cell "None" [ref=e116]
              - cell "1" [ref=e117]
        - list [ref=e118]:
          - listitem [ref=e119]:
            - generic [aria-hidden]: ←
          - listitem [ref=e120]:
            - generic "Current Page" [ref=e121]: "1"
          - listitem [ref=e122]:
            - generic [aria-hidden]: →
      - generic [ref=e123]:
        - generic [ref=e124]:
          - text: Sieć Obywatelska - Watchdog Polska ul. Szpitalna 5/5 00-031 Warszawa
          - paragraph [ref=e125]:
            - link "Klauzula RODO" [ref=e126] [cursor=pointer]:
              - /url: https://fedrowanie.siecobywatelska.pl/media_internal/tinycontent/uploads/11_KLAUZULA.-.dane.z.udip.na.stronach.www.i.w.mediach.spol.pdf
        - generic [ref=e127]:
          - text: "silnik:"
          - link "jawne.info.pl" [ref=e128] [cursor=pointer]:
            - /url: http://jawne.info.pl
          - text: "| v1.5.77.deps |"
          - link "GitHub" [ref=e129] [cursor=pointer]:
            - /url: https://github.com/watchdogpolska/feder
          - text: –
          - link "efd8a3b" [ref=e131] [cursor=pointer]:
            - /url: https://github.com/watchdogpolska/feder/compare/efd8a3b2...master
          - text: "|"
          - link "API" [ref=e132] [cursor=pointer]:
            - /url: /api/
        - generic [ref=e134]: Ta strona wykorzystuje cookies.
  - list [ref=e136]:
    - listitem [ref=e137]:
      - link "Ukryj »" [ref=e138] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e139]:
      - link "Toggle Theme" [ref=e140] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e143]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e144]
      - link "Historia /listy/logi/monitoring-5" [ref=e145] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e146]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e147]
      - link "Wersje Django 5.2.17" [ref=e148] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e149]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e150]
      - 'link "Czas CPU: 107.28ms (110.78ms)" [ref=e151] [cursor=pointer]':
        - /url: "#"
    - listitem [ref=e152]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e153]
      - link "Ustawienia" [ref=e154] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e155]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e156]
      - link "Nagłówki" [ref=e157] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e158]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e159]
      - link "Zapytania EmailLogMonitoringListView" [ref=e160] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e161]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e162]
      - link "SQL 10 queries in 3.89ms" [ref=e163] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e164]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e165]
      - link "Pliki statyczne 3 użyte plików" [ref=e166] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e167]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e168]
      - link "Templatki logs/emaillog_list_for_monitoring.html" [ref=e169] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e170]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e171]
      - link "Alerty" [ref=e172] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e173]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e174]
      - link "Cache 2 wywołania w 0.15ms" [ref=e175] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e176]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e177]
      - link "Sygnały 88 odbiorców 15 sygnałów" [ref=e178] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e179]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e180]
      - link "Gmina" [ref=e181] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e182]:
      - checkbox "Enable for next and successive requests" [ref=e183]
      - generic [ref=e184]: Przechwycone przekierowania
    - listitem [ref=e185]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e186]
      - link "Profilowanie" [ref=e187] [cursor=pointer]:
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