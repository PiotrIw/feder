# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: visual_baseline.spec.ts >> mobile (375px) >> parcels-outgoing-details - visual baseline
- Location: tests/bs3-bs5_migration/visual_baseline.spec.ts:10:11

# Error details

```
Error: expect(page).toHaveScreenshot(expected) failed

  Expected an image 375px by 1081px, received 375px by 1065px. 170037 pixels (ratio 0.42 of all image pixels) are different.

  Snapshot: parcels-outgoing-details-mobile.png

Call log:
  - Expect "toHaveScreenshot(parcels-outgoing-details-mobile.png)" with timeout 5000ms
    - verifying given screenshot expectation
  - taking page screenshot
    - disabled all CSS animations
  - waiting for fonts to load...
  - fonts loaded
  - Expected an image 375px by 1081px, received 375px by 1065px. 170037 pixels (ratio 0.42 of all image pixels) are different.
  - waiting 100ms before taking screenshot
  - taking page screenshot
    - disabled all CSS animations
  - waiting for fonts to load...
  - fonts loaded
  - captured a stable screenshot
  - Expected an image 375px by 1081px, received 375px by 1065px. 170037 pixels (ratio 0.42 of all image pixels) are different.

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
          - link "Żywienie w szpitalach" [ref=e71] [cursor=pointer]:
            - /url: /monitoringi/zywienie-w-szpitalach
        - listitem [ref=e72]:
          - text: /
          - 'link "Żywienie w szpitalach #576" [ref=e73] [cursor=pointer]':
            - /url: /sprawy/zywienie-w-szpitalach-576
        - listitem [ref=e74]: / Centrum Medyczne Ujastek w Krakowie (żywienie w szpitalach) - wniosek epuap30.10.2019
      - generic [ref=e76]:
        - link "Edytuj" [ref=e77] [cursor=pointer]:
          - /url: /przesylki/outgoing-1/~update
        - link "Usuń" [ref=e78] [cursor=pointer]:
          - /url: /przesylki/outgoing-1/~delete
      - heading [level=2] [ref=e80]:
        - link "Centrum Medyczne Ujastek w Krakowie (żywienie w szpitalach) - wniosek epuap30.10.2019" [ref=e82] [cursor=pointer]:
          - /url: /przesylki/outgoing-1
      - table [ref=e84]:
        - rowgroup [ref=e85]:
          - row [ref=e86]:
            - cell "Adresat" [ref=e87]
            - cell [ref=e89]:
              - link "CENTRUM MEDYCZNE UJASTEK SPÓŁKA Z OGRANICZONĄ ODPOWIEDZIALNOŚCIĄ" [ref=e90] [cursor=pointer]:
                - /url: /instytucje/centrum-medyczne-ujastek-spolka-z-ograniczona-odpo
          - row [ref=e91]:
            - cell "Data wysłania" [ref=e92]
            - cell "30 października 2019" [ref=e94]
          - row [ref=e95]:
            - cell "Data utworzenia" [ref=e96]
            - cell "30 października 2019 14:53" [ref=e98]
          - row [ref=e99]:
            - cell "Data modyfikacji" [ref=e100]
            - cell "30 października 2019 14:53" [ref=e102]
          - row [ref=e103]:
            - cell "Treść" [ref=e104]
            - cell [ref=e106]:
              - link "Pobierz" [ref=e107] [cursor=pointer]:
                - /url: /przesylki/outgoing-1/~download
      - generic [ref=e108]:
        - generic [ref=e109]:
          - text: Sieć Obywatelska - Watchdog Polska ul. Szpitalna 5/5 00-031 Warszawa
          - paragraph [ref=e110]:
            - link "Klauzula RODO" [ref=e111] [cursor=pointer]:
              - /url: https://fedrowanie.siecobywatelska.pl/media_internal/tinycontent/uploads/11_KLAUZULA.-.dane.z.udip.na.stronach.www.i.w.mediach.spol.pdf
        - generic [ref=e112]:
          - text: "silnik:"
          - link "jawne.info.pl" [ref=e113] [cursor=pointer]:
            - /url: http://jawne.info.pl
          - text: "| v1.5.77.deps |"
          - link "GitHub" [ref=e114] [cursor=pointer]:
            - /url: https://github.com/watchdogpolska/feder
          - text: –
          - link "efd8a3b" [ref=e116] [cursor=pointer]:
            - /url: https://github.com/watchdogpolska/feder/compare/efd8a3b2...master
          - text: "|"
          - link "API" [ref=e117] [cursor=pointer]:
            - /url: /api/
        - generic [ref=e119]: Ta strona wykorzystuje cookies.
  - list [ref=e121]:
    - listitem [ref=e122]:
      - link "Ukryj »" [ref=e123] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e124]:
      - link "Toggle Theme" [ref=e125] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e128]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e129]
      - link "Historia /przesylki/outgoing-1" [ref=e130] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e131]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e132]
      - link "Wersje Django 5.2.17" [ref=e133] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e134]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e135]
      - 'link "Czas CPU: 98.27ms (93.53ms)" [ref=e136] [cursor=pointer]':
        - /url: "#"
    - listitem [ref=e137]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e138]
      - link "Ustawienia" [ref=e139] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e140]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e141]
      - link "Nagłówki" [ref=e142] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e143]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e144]
      - link "Zapytania OutgoingParcelPostDetailView" [ref=e145] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e146]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e147]
      - link "SQL 7 queries in 2.89ms" [ref=e148] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e149]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e150]
      - link "Pliki statyczne 3 użyte plików" [ref=e151] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e152]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e153]
      - link "Templatki parcels/outgoingparcelpost_detail.html" [ref=e154] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e155]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e156]
      - link "Alerty" [ref=e157] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e158]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e159]
      - link "Cache 2 wywołania w 0.13ms" [ref=e160] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e161]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e162]
      - link "Sygnały 88 odbiorców 15 sygnałów" [ref=e163] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e164]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e165]
      - link "Gmina" [ref=e166] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e167]:
      - checkbox "Enable for next and successive requests" [ref=e168]
      - generic [ref=e169]: Przechwycone przekierowania
    - listitem [ref=e170]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e171]
      - link "Profilowanie" [ref=e172] [cursor=pointer]:
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