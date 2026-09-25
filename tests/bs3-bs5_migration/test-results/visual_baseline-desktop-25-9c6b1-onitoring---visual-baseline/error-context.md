# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: visual_baseline.spec.ts >> desktop (2560px) >> letters-logs-list-monitoring - visual baseline
- Location: tests/bs3-bs5_migration/visual_baseline.spec.ts:10:11

# Error details

```
Error: expect(page).toHaveScreenshot(expected) failed

  355052 pixels (ratio 0.10 of all image pixels) are different.

  Snapshot: letters-logs-list-monitoring-desktop.png

Call log:
  - Expect "toHaveScreenshot(letters-logs-list-monitoring-desktop.png)" with timeout 5000ms
    - verifying given screenshot expectation
  - taking page screenshot
    - disabled all CSS animations
  - waiting for fonts to load...
  - fonts loaded
  - 355052 pixels (ratio 0.10 of all image pixels) are different.
  - waiting 100ms before taking screenshot
  - taking page screenshot
    - disabled all CSS animations
  - waiting for fonts to load...
  - fonts loaded
  - captured a stable screenshot
  - 355052 pixels (ratio 0.10 of all image pixels) are different.

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
        - listitem [ref=e64]: / Dziennik
      - generic [ref=e66]:
        - link "Edytuj" [ref=e67] [cursor=pointer]:
          - /url: /monitoringi/monitoring-sadow-apelacyjnych/~edytuj
        - link "Aktualizuj wyniki" [ref=e68] [cursor=pointer]:
          - /url: /monitoringi/monitoring-sadow-apelacyjnych/~results-update
        - link "Przypisz" [ref=e69] [cursor=pointer]:
          - /url: /monitoringi/monitoring-sadow-apelacyjnych/~przypisz
        - link "Usuń" [ref=e70] [cursor=pointer]:
          - /url: /monitoringi/monitoring-sadow-apelacyjnych/~usun
        - link "Utwórz sprawę" [ref=e71] [cursor=pointer]:
          - /url: /sprawy/~utworz-5
        - link "Wiadomość masowa" [ref=e72] [cursor=pointer]:
          - /url: /monitoringi/monitoring-sadow-apelacyjnych/~wiadomosc-masowa
        - link "Uprawnienia" [ref=e73] [cursor=pointer]:
          - /url: /monitoringi/monitoring-sadow-apelacyjnych/~uprawnienia
        - link "Lista alertów" [ref=e74] [cursor=pointer]:
          - /url: /alerty/monitoring-5
        - link "Zobacz dzienniki" [ref=e75] [cursor=pointer]:
          - /url: /listy/logi/monitoring-5
        - link "Zobacz tagi" [ref=e77] [cursor=pointer]:
          - /url: /sprawy/tagi/monitoring-5
        - link "Zobacz raport" [ref=e79] [cursor=pointer]:
          - /url: /monitoringi/monitoring-sadow-apelacyjnych/raport
        - link "Zobacz tabelę spraw" [ref=e81] [cursor=pointer]:
          - /url: /monitoringi/monitoring-sadow-apelacyjnych/monitoring_cases_table
      - heading [level=2] [ref=e84]:
        - text: Monitoring sądów apelacyjnych
        - generic [ref=e86]:
          - text: przez
          - link "adobrawy" [ref=e87] [cursor=pointer]:
            - /url: /uzytkownik/adobrawy/
          - time [ref=e88]: 11 sierpnia 2017 02:47
      - link "Pobierz .csv" [ref=e90] [cursor=pointer]:
        - /url: /listy/logi/monitoring-5/eksport
      - generic [ref=e92]:
        - table [ref=e93]:
          - rowgroup [ref=e94]:
            - row [ref=e95]:
              - columnheader "ID" [ref=e96]
              - columnheader "Sprawa" [ref=e97]
              - columnheader "Status" [ref=e98]
              - columnheader "List" [ref=e99]
              - columnheader "Liczba wpisów" [ref=e100]
          - rowgroup [ref=e101]:
            - row [ref=e102]:
              - cell [ref=e103]:
                - link "59a127fa42cf33b253a32932" [ref=e104] [cursor=pointer]:
                  - /url: /listy/logi/wpis-430
              - cell [ref=e105]:
                - 'link "Monitoring sądów apelacyjnych #2" [ref=e106] [cursor=pointer]':
                  - /url: /sprawy/monitoring-sadow-apelacyjnych-2
              - cell "Dostarczony" [ref=e107]
              - cell "None" [ref=e108]
              - cell "1" [ref=e109]
        - list [ref=e110]:
          - listitem [ref=e111]:
            - generic [aria-hidden]: ←
          - listitem [ref=e112]:
            - generic "Current Page" [ref=e113]: "1"
          - listitem [ref=e114]:
            - generic [aria-hidden]: →
      - generic [ref=e115]:
        - generic [ref=e116]:
          - text: Sieć Obywatelska - Watchdog Polska ul. Szpitalna 5/5 00-031 Warszawa
          - paragraph [ref=e117]:
            - link "Klauzula RODO" [ref=e118] [cursor=pointer]:
              - /url: https://fedrowanie.siecobywatelska.pl/media_internal/tinycontent/uploads/11_KLAUZULA.-.dane.z.udip.na.stronach.www.i.w.mediach.spol.pdf
        - generic [ref=e119]:
          - text: "silnik:"
          - link "jawne.info.pl" [ref=e120] [cursor=pointer]:
            - /url: http://jawne.info.pl
          - text: "| v1.5.77.deps |"
          - link "GitHub" [ref=e121] [cursor=pointer]:
            - /url: https://github.com/watchdogpolska/feder
          - text: –
          - link "efd8a3b" [ref=e123] [cursor=pointer]:
            - /url: https://github.com/watchdogpolska/feder/compare/efd8a3b2...master
          - text: "|"
          - link "API" [ref=e124] [cursor=pointer]:
            - /url: /api/
        - generic [ref=e126]: Ta strona wykorzystuje cookies.
  - list [ref=e128]:
    - listitem [ref=e129]:
      - link "Ukryj »" [ref=e130] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e131]:
      - link "Toggle Theme" [ref=e132] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e135]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e136]
      - link "Historia /listy/logi/monitoring-5" [ref=e137] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e138]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e139]
      - link "Wersje Django 5.2.17" [ref=e140] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e141]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e142]
      - 'link "Czas CPU: 99.36ms (102.67ms)" [ref=e143] [cursor=pointer]':
        - /url: "#"
    - listitem [ref=e144]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e145]
      - link "Ustawienia" [ref=e146] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e147]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e148]
      - link "Nagłówki" [ref=e149] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e150]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e151]
      - link "Zapytania EmailLogMonitoringListView" [ref=e152] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e153]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e154]
      - link "SQL 10 queries in 4.05ms" [ref=e155] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e156]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e157]
      - link "Pliki statyczne 3 użyte plików" [ref=e158] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e159]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e160]
      - link "Templatki logs/emaillog_list_for_monitoring.html" [ref=e161] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e162]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e163]
      - link "Alerty" [ref=e164] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e165]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e166]
      - link "Cache 2 wywołania w 0.13ms" [ref=e167] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e168]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e169]
      - link "Sygnały 88 odbiorców 15 sygnałów" [ref=e170] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e171]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e172]
      - link "Gmina" [ref=e173] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e174]:
      - checkbox "Enable for next and successive requests" [ref=e175]
      - generic [ref=e176]: Przechwycone przekierowania
    - listitem [ref=e177]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e178]
      - link "Profilowanie" [ref=e179] [cursor=pointer]:
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