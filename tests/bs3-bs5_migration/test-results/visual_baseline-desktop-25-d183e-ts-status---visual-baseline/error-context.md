# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: visual_baseline.spec.ts >> desktop (2560px) >> alerts-status - visual baseline
- Location: tests/bs3-bs5_migration/visual_baseline.spec.ts:10:11

# Error details

```
Error: expect(page).toHaveScreenshot(expected) failed

  338639 pixels (ratio 0.10 of all image pixels) are different.

  Snapshot: alerts-status-desktop.png

Call log:
  - Expect "toHaveScreenshot(alerts-status-desktop.png)" with timeout 5000ms
    - verifying given screenshot expectation
  - taking page screenshot
    - disabled all CSS animations
  - waiting for fonts to load...
  - fonts loaded
  - 338639 pixels (ratio 0.10 of all image pixels) are different.
  - waiting 100ms before taking screenshot
  - taking page screenshot
    - disabled all CSS animations
  - waiting for fonts to load...
  - fonts loaded
  - captured a stable screenshot
  - 338639 pixels (ratio 0.10 of all image pixels) are different.

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
          - link "Samorządy o liczbę wniosków 2017" [ref=e63] [cursor=pointer]:
            - /url: /monitoringi/o-liczbe-wnioskow-2017
        - listitem [ref=e64]:
          - text: /
          - link "Wykaz alertów" [ref=e65] [cursor=pointer]:
            - /url: /alerty/monitoring-4
        - listitem [ref=e66]: / 2017-10-18 20:41:18.641909+00:00
      - generic [ref=e68]:
        - link "Edytuj" [ref=e69] [cursor=pointer]:
          - /url: /alerty/1/~aktualizuj
        - link "Przestaw status" [ref=e70] [cursor=pointer]:
          - /url: /alerty/1/~status
        - link "Usuń" [ref=e71] [cursor=pointer]:
          - /url: /alerty/1/~usun
      - heading "2017-10-18 20:41:18.641909+00:00" [level=1] [ref=e73]
      - button "Przestaw" [ref=e78] [cursor=pointer]
      - generic [ref=e79]:
        - generic [ref=e80]:
          - text: Sieć Obywatelska - Watchdog Polska ul. Szpitalna 5/5 00-031 Warszawa
          - paragraph [ref=e81]:
            - link "Klauzula RODO" [ref=e82] [cursor=pointer]:
              - /url: https://fedrowanie.siecobywatelska.pl/media_internal/tinycontent/uploads/11_KLAUZULA.-.dane.z.udip.na.stronach.www.i.w.mediach.spol.pdf
        - generic [ref=e83]:
          - text: "silnik:"
          - link "jawne.info.pl" [ref=e84] [cursor=pointer]:
            - /url: http://jawne.info.pl
          - text: "| v1.5.77.deps |"
          - link "GitHub" [ref=e85] [cursor=pointer]:
            - /url: https://github.com/watchdogpolska/feder
          - text: –
          - link "efd8a3b" [ref=e87] [cursor=pointer]:
            - /url: https://github.com/watchdogpolska/feder/compare/efd8a3b2...master
          - text: "|"
          - link "API" [ref=e88] [cursor=pointer]:
            - /url: /api/
        - generic [ref=e90]: Ta strona wykorzystuje cookies.
  - list [ref=e92]:
    - listitem [ref=e93]:
      - link "Ukryj »" [ref=e94] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e95]:
      - link "Toggle Theme" [ref=e96] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e99]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e100]
      - link "Historia /alerty/1/~status" [ref=e101] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e102]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e103]
      - link "Wersje Django 5.2.17" [ref=e104] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e105]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e106]
      - 'link "Czas CPU: 76.50ms (79.29ms)" [ref=e107] [cursor=pointer]':
        - /url: "#"
    - listitem [ref=e108]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e109]
      - link "Ustawienia" [ref=e110] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e111]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e112]
      - link "Nagłówki" [ref=e113] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e114]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e115]
      - link "Zapytania AlertStatusView" [ref=e116] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e117]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e118]
      - link "SQL 9 queries in 3.25ms" [ref=e119] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e120]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e121]
      - link "Pliki statyczne 3 użyte plików" [ref=e122] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e123]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e124]
      - link "Templatki alerts/alert_switch.html" [ref=e125] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e126]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e127]
      - link "Alerty" [ref=e128] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e129]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e130]
      - link "Cache 2 wywołania w 0.15ms" [ref=e131] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e132]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e133]
      - link "Sygnały 88 odbiorców 15 sygnałów" [ref=e134] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e135]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e136]
      - link "Gmina" [ref=e137] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e138]:
      - checkbox "Enable for next and successive requests" [ref=e139]
      - generic [ref=e140]: Przechwycone przekierowania
    - listitem [ref=e141]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e142]
      - link "Profilowanie" [ref=e143] [cursor=pointer]:
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