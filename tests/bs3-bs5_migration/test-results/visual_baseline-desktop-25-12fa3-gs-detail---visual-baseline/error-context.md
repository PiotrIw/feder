# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: visual_baseline.spec.ts >> desktop (2560px) >> letters-logs-detail - visual baseline
- Location: tests/bs3-bs5_migration/visual_baseline.spec.ts:10:11

# Error details

```
Error: expect(page).toHaveScreenshot(expected) failed

  357926 pixels (ratio 0.10 of all image pixels) are different.

  Snapshot: letters-logs-detail-desktop.png

Call log:
  - Expect "toHaveScreenshot(letters-logs-detail-desktop.png)" with timeout 5000ms
    - verifying given screenshot expectation
  - taking page screenshot
    - disabled all CSS animations
  - waiting for fonts to load...
  - fonts loaded
  - 357926 pixels (ratio 0.10 of all image pixels) are different.
  - waiting 100ms before taking screenshot
  - taking page screenshot
    - disabled all CSS animations
  - waiting for fonts to load...
  - fonts loaded
  - captured a stable screenshot
  - 357926 pixels (ratio 0.10 of all image pixels) are different.

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
          - link "Monitoring nadleśnictw" [ref=e63] [cursor=pointer]:
            - /url: /monitoringi/monitoring-nadlesnictw
        - listitem [ref=e64]:
          - text: /
          - 'link "Monitoring nadleśnictw #14" [ref=e65] [cursor=pointer]':
            - /url: /sprawy/monitoring-nadlesnictw-14
        - listitem [ref=e66]: "/ Email #1 (599ef08c42cf33b253fdc5f6)"
      - generic [ref=e68]:
        - link "Edytuj" [ref=e69] [cursor=pointer]:
          - /url: /sprawy/monitoring-nadlesnictw-14/~edytuj
        - link "Usuń" [ref=e70] [cursor=pointer]:
          - /url: /sprawy/monitoring-nadlesnictw-14/~usun
        - link "Zobacz dzienniki" [ref=e71] [cursor=pointer]:
          - /url: /listy/logi/spraw-3070
        - button "Dodaj przesyłkę pocztową" [ref=e73] [cursor=pointer]
        - link "Dodaj list" [ref=e75] [cursor=pointer]:
          - /url: /listy/~utworz-3070
      - heading [level=2] [ref=e77]:
        - 'link "Email #1 (599ef08c42cf33b253fdc5f6)" [ref=e79] [cursor=pointer]':
          - /url: /listy/logi/wpis-1
        - time [ref=e81]: 24 sierpnia 2017 17:30
      - generic [ref=e82]:
        - heading "24 sierpnia 2017 17:30" [level=3] [ref=e83]
        - generic [ref=e84]: "{ \"ok_desc\": \"250 2.0.0 Ok: queued as A3B925BF18\", \"account\": \"1.siecobywatelska.smtp\", \"tracking\": [], \"from\": \"sprawa-3070@fedrowanie.siecobywatelska.pl\", \"open_time\": null, \"vps\": \"smtp2-87\", \"tags\": [], \"injected_time\": \"2017-08-24 17:25:50\", \"created_at\": null, \"updated_at\": null, \"id\": \"599ef08c42cf33b253fdc5f6\", \"to\": \"bialowieza@bialystok.lasy.gov.pl\", \"postfix_id\": [ \"3xdSmZ0kpMz6jsBt\", \"3xdSmZ2ZvWz6Q7V0\" ], \"ok_time\": \"2017-08-24 17:25:50\", \"open_desc\": null, \"subject\": \"Wniosek o udost\\u0119pnienie informacji publicznej\", \"message_id\": \"20170824152549.2577.77274@localhost\", \"uid\": \"b1db7556ea65065c69d86b81ef248eb5\" }"
      - generic [ref=e85]:
        - generic [ref=e86]:
          - text: Sieć Obywatelska - Watchdog Polska ul. Szpitalna 5/5 00-031 Warszawa
          - paragraph [ref=e87]:
            - link "Klauzula RODO" [ref=e88] [cursor=pointer]:
              - /url: https://fedrowanie.siecobywatelska.pl/media_internal/tinycontent/uploads/11_KLAUZULA.-.dane.z.udip.na.stronach.www.i.w.mediach.spol.pdf
        - generic [ref=e89]:
          - text: "silnik:"
          - link "jawne.info.pl" [ref=e90] [cursor=pointer]:
            - /url: http://jawne.info.pl
          - text: "| v1.5.77.deps |"
          - link "GitHub" [ref=e91] [cursor=pointer]:
            - /url: https://github.com/watchdogpolska/feder
          - text: –
          - link "efd8a3b" [ref=e93] [cursor=pointer]:
            - /url: https://github.com/watchdogpolska/feder/compare/efd8a3b2...master
          - text: "|"
          - link "API" [ref=e94] [cursor=pointer]:
            - /url: /api/
        - generic [ref=e96]: Ta strona wykorzystuje cookies.
  - list [ref=e98]:
    - listitem [ref=e99]:
      - link "Ukryj »" [ref=e100] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e101]:
      - link "Toggle Theme" [ref=e102] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e105]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e106]
      - link "Historia /listy/logi/wpis-1" [ref=e107] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e108]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e109]
      - link "Wersje Django 5.2.17" [ref=e110] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e111]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e112]
      - 'link "Czas CPU: 87.25ms (90.75ms)" [ref=e113] [cursor=pointer]':
        - /url: "#"
    - listitem [ref=e114]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e115]
      - link "Ustawienia" [ref=e116] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e117]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e118]
      - link "Nagłówki" [ref=e119] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e120]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e121]
      - link "Zapytania EmailLogDetailView" [ref=e122] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e123]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e124]
      - link "SQL 10 queries in 4.02ms" [ref=e125] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e126]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e127]
      - link "Pliki statyczne 3 użyte plików" [ref=e128] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e129]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e130]
      - link "Templatki logs/emaillog_detail.html" [ref=e131] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e132]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e133]
      - link "Alerty" [ref=e134] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e135]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e136]
      - link "Cache 2 wywołania w 0.14ms" [ref=e137] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e138]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e139]
      - link "Sygnały 88 odbiorców 15 sygnałów" [ref=e140] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e141]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e142]
      - link "Gmina" [ref=e143] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e144]:
      - checkbox "Enable for next and successive requests" [ref=e145]
      - generic [ref=e146]: Przechwycone przekierowania
    - listitem [ref=e147]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e148]
      - link "Profilowanie" [ref=e149] [cursor=pointer]:
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