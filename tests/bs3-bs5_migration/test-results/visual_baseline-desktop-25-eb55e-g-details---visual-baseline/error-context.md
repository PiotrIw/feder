# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: visual_baseline.spec.ts >> desktop (2560px) >> parcels-outgoing-details - visual baseline
- Location: tests/bs3-bs5_migration/visual_baseline.spec.ts:10:11

# Error details

```
Error: expect(page).toHaveScreenshot(expected) failed

  349379 pixels (ratio 0.10 of all image pixels) are different.

  Snapshot: parcels-outgoing-details-desktop.png

Call log:
  - Expect "toHaveScreenshot(parcels-outgoing-details-desktop.png)" with timeout 5000ms
    - verifying given screenshot expectation
  - taking page screenshot
    - disabled all CSS animations
  - waiting for fonts to load...
  - fonts loaded
  - 349379 pixels (ratio 0.10 of all image pixels) are different.
  - waiting 100ms before taking screenshot
  - taking page screenshot
    - disabled all CSS animations
  - waiting for fonts to load...
  - fonts loaded
  - captured a stable screenshot
  - 349379 pixels (ratio 0.10 of all image pixels) are different.

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
          - link "Żywienie w szpitalach" [ref=e63] [cursor=pointer]:
            - /url: /monitoringi/zywienie-w-szpitalach
        - listitem [ref=e64]:
          - text: /
          - 'link "Żywienie w szpitalach #576" [ref=e65] [cursor=pointer]':
            - /url: /sprawy/zywienie-w-szpitalach-576
        - listitem [ref=e66]: / Centrum Medyczne Ujastek w Krakowie (żywienie w szpitalach) - wniosek epuap30.10.2019
      - generic [ref=e68]:
        - link "Edytuj" [ref=e69] [cursor=pointer]:
          - /url: /przesylki/outgoing-1/~update
        - link "Usuń" [ref=e70] [cursor=pointer]:
          - /url: /przesylki/outgoing-1/~delete
      - heading [level=2] [ref=e72]:
        - link "Centrum Medyczne Ujastek w Krakowie (żywienie w szpitalach) - wniosek epuap30.10.2019" [ref=e74] [cursor=pointer]:
          - /url: /przesylki/outgoing-1
      - table [ref=e76]:
        - rowgroup [ref=e77]:
          - row [ref=e78]:
            - cell "Adresat" [ref=e79]
            - cell [ref=e81]:
              - link "CENTRUM MEDYCZNE UJASTEK SPÓŁKA Z OGRANICZONĄ ODPOWIEDZIALNOŚCIĄ" [ref=e82] [cursor=pointer]:
                - /url: /instytucje/centrum-medyczne-ujastek-spolka-z-ograniczona-odpo
          - row [ref=e83]:
            - cell "Data wysłania" [ref=e84]
            - cell "30 października 2019" [ref=e86]
          - row [ref=e87]:
            - cell "Data utworzenia" [ref=e88]
            - cell "30 października 2019 14:53" [ref=e90]
          - row [ref=e91]:
            - cell "Data modyfikacji" [ref=e92]
            - cell "30 października 2019 14:53" [ref=e94]
          - row [ref=e95]:
            - cell "Treść" [ref=e96]
            - cell [ref=e98]:
              - link "Pobierz" [ref=e99] [cursor=pointer]:
                - /url: /przesylki/outgoing-1/~download
      - generic [ref=e100]:
        - generic [ref=e101]:
          - text: Sieć Obywatelska - Watchdog Polska ul. Szpitalna 5/5 00-031 Warszawa
          - paragraph [ref=e102]:
            - link "Klauzula RODO" [ref=e103] [cursor=pointer]:
              - /url: https://fedrowanie.siecobywatelska.pl/media_internal/tinycontent/uploads/11_KLAUZULA.-.dane.z.udip.na.stronach.www.i.w.mediach.spol.pdf
        - generic [ref=e104]:
          - text: "silnik:"
          - link "jawne.info.pl" [ref=e105] [cursor=pointer]:
            - /url: http://jawne.info.pl
          - text: "| v1.5.77.deps |"
          - link "GitHub" [ref=e106] [cursor=pointer]:
            - /url: https://github.com/watchdogpolska/feder
          - text: –
          - link "efd8a3b" [ref=e108] [cursor=pointer]:
            - /url: https://github.com/watchdogpolska/feder/compare/efd8a3b2...master
          - text: "|"
          - link "API" [ref=e109] [cursor=pointer]:
            - /url: /api/
        - generic [ref=e111]: Ta strona wykorzystuje cookies.
  - list [ref=e113]:
    - listitem [ref=e114]:
      - link "Ukryj »" [ref=e115] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e116]:
      - link "Toggle Theme" [ref=e117] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e120]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e121]
      - link "Historia /przesylki/outgoing-1" [ref=e122] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e123]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e124]
      - link "Wersje Django 5.2.17" [ref=e125] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e126]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e127]
      - 'link "Czas CPU: 91.28ms (88.47ms)" [ref=e128] [cursor=pointer]':
        - /url: "#"
    - listitem [ref=e129]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e130]
      - link "Ustawienia" [ref=e131] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e132]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e133]
      - link "Nagłówki" [ref=e134] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e135]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e136]
      - link "Zapytania OutgoingParcelPostDetailView" [ref=e137] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e138]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e139]
      - link "SQL 7 queries in 2.97ms" [ref=e140] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e141]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e142]
      - link "Pliki statyczne 3 użyte plików" [ref=e143] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e144]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e145]
      - link "Templatki parcels/outgoingparcelpost_detail.html" [ref=e146] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e147]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e148]
      - link "Alerty" [ref=e149] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e150]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e151]
      - link "Cache 2 wywołania w 0.13ms" [ref=e152] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e153]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e154]
      - link "Sygnały 88 odbiorców 15 sygnałów" [ref=e155] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e156]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e157]
      - link "Gmina" [ref=e158] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e159]:
      - checkbox "Enable for next and successive requests" [ref=e160]
      - generic [ref=e161]: Przechwycone przekierowania
    - listitem [ref=e162]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e163]
      - link "Profilowanie" [ref=e164] [cursor=pointer]:
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