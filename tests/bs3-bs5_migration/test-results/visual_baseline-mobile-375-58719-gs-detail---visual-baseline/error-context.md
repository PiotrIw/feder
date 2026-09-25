# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: visual_baseline.spec.ts >> mobile (375px) >> letters-logs-detail - visual baseline
- Location: tests/bs3-bs5_migration/visual_baseline.spec.ts:10:11

# Error details

```
Error: expect(page).toHaveScreenshot(expected) failed

  Expected an image 397px by 1202px, received 383px by 1124px. 175071 pixels (ratio 0.37 of all image pixels) are different.

  Snapshot: letters-logs-detail-mobile.png

Call log:
  - Expect "toHaveScreenshot(letters-logs-detail-mobile.png)" with timeout 5000ms
    - verifying given screenshot expectation
  - taking page screenshot
    - disabled all CSS animations
  - waiting for fonts to load...
  - fonts loaded
  - Expected an image 397px by 1202px, received 383px by 1124px. 175071 pixels (ratio 0.37 of all image pixels) are different.
  - waiting 100ms before taking screenshot
  - taking page screenshot
    - disabled all CSS animations
  - waiting for fonts to load...
  - fonts loaded
  - captured a stable screenshot
  - Expected an image 397px by 1202px, received 383px by 1124px. 175071 pixels (ratio 0.37 of all image pixels) are different.

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
          - link "Monitoring nadleśnictw" [ref=e71] [cursor=pointer]:
            - /url: /monitoringi/monitoring-nadlesnictw
        - listitem [ref=e72]:
          - text: /
          - 'link "Monitoring nadleśnictw #14" [ref=e73] [cursor=pointer]':
            - /url: /sprawy/monitoring-nadlesnictw-14
        - listitem [ref=e74]: "/ Email #1 (599ef08c42cf33b253fdc5f6)"
      - generic [ref=e76]:
        - link "Edytuj" [ref=e77] [cursor=pointer]:
          - /url: /sprawy/monitoring-nadlesnictw-14/~edytuj
        - link "Usuń" [ref=e78] [cursor=pointer]:
          - /url: /sprawy/monitoring-nadlesnictw-14/~usun
        - link "Zobacz dzienniki" [ref=e79] [cursor=pointer]:
          - /url: /listy/logi/spraw-3070
        - button "Dodaj przesyłkę pocztową" [ref=e81] [cursor=pointer]
        - link "Dodaj list" [ref=e83] [cursor=pointer]:
          - /url: /listy/~utworz-3070
      - heading [level=2] [ref=e85]:
        - 'link "Email #1 (599ef08c42cf33b253fdc5f6)" [ref=e87] [cursor=pointer]':
          - /url: /listy/logi/wpis-1
        - time [ref=e89]: 24 sierpnia 2017 17:30
      - generic [ref=e90]:
        - heading "24 sierpnia 2017 17:30" [level=3] [ref=e91]
        - generic [ref=e92]: "{ \"ok_desc\": \"250 2.0.0 Ok: queued as A3B925BF18\", \"account\": \"1.siecobywatelska.smtp\", \"tracking\": [], \"from\": \"sprawa-3070@fedrowanie.siecobywatelska.pl\", \"open_time\": null, \"vps\": \"smtp2-87\", \"tags\": [], \"injected_time\": \"2017-08-24 17:25:50\", \"created_at\": null, \"updated_at\": null, \"id\": \"599ef08c42cf33b253fdc5f6\", \"to\": \"bialowieza@bialystok.lasy.gov.pl\", \"postfix_id\": [ \"3xdSmZ0kpMz6jsBt\", \"3xdSmZ2ZvWz6Q7V0\" ], \"ok_time\": \"2017-08-24 17:25:50\", \"open_desc\": null, \"subject\": \"Wniosek o udost\\u0119pnienie informacji publicznej\", \"message_id\": \"20170824152549.2577.77274@localhost\", \"uid\": \"b1db7556ea65065c69d86b81ef248eb5\" }"
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
      - link "Historia /listy/logi/wpis-1" [ref=e115] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e116]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e117]
      - link "Wersje Django 5.2.17" [ref=e118] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e119]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e120]
      - 'link "Czas CPU: 87.41ms (90.44ms)" [ref=e121] [cursor=pointer]':
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
      - link "Zapytania EmailLogDetailView" [ref=e130] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e131]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e132]
      - link "SQL 10 queries in 3.53ms" [ref=e133] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e134]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e135]
      - link "Pliki statyczne 3 użyte plików" [ref=e136] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e137]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e138]
      - link "Templatki logs/emaillog_detail.html" [ref=e139] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e140]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e141]
      - link "Alerty" [ref=e142] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e143]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e144]
      - link "Cache 2 wywołania w 0.13ms" [ref=e145] [cursor=pointer]:
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