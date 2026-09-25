# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: visual_baseline.spec.ts >> mobile (375px) >> parcels-outgoing-update - visual baseline
- Location: tests/bs3-bs5_migration/visual_baseline.spec.ts:10:11

# Error details

```
Error: expect(page).toHaveScreenshot(expected) failed

  Expected an image 488px by 1195px, received 375px by 1266px. 180914 pixels (ratio 0.30 of all image pixels) are different.

  Snapshot: parcels-outgoing-update-mobile.png

Call log:
  - Expect "toHaveScreenshot(parcels-outgoing-update-mobile.png)" with timeout 5000ms
    - verifying given screenshot expectation
  - taking page screenshot
    - disabled all CSS animations
  - waiting for fonts to load...
  - fonts loaded
  - Expected an image 488px by 1195px, received 375px by 1266px. 180914 pixels (ratio 0.30 of all image pixels) are different.
  - waiting 100ms before taking screenshot
  - taking page screenshot
    - disabled all CSS animations
  - waiting for fonts to load...
  - fonts loaded
  - captured a stable screenshot
  - Expected an image 488px by 1195px, received 375px by 1266px. 180914 pixels (ratio 0.30 of all image pixels) are different.

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
        - listitem [ref=e74]:
          - text: /
          - link "Centrum Medyczne Ujastek w Krakowie (żywienie w szpitalach) - wniosek epuap30.10.2019" [ref=e75] [cursor=pointer]:
            - /url: /przesylki/outgoing-1
        - listitem [ref=e76]: / Zaktualizujs wychodzącą przesyłkę pocztową
      - generic [ref=e78]:
        - link "Edytuj" [ref=e79] [cursor=pointer]:
          - /url: /przesylki/outgoing-1/~update
        - link "Usuń" [ref=e80] [cursor=pointer]:
          - /url: /przesylki/outgoing-1/~delete
      - heading [level=2] [ref=e82]:
        - link "Centrum Medyczne Ujastek w Krakowie (żywienie w szpitalach) - wniosek epuap30.10.2019" [ref=e84] [cursor=pointer]:
          - /url: /przesylki/outgoing-1
      - generic [ref=e86]:
        - generic [ref=e87]:
          - generic [ref=e88]: Tytuł*
          - textbox "Tytuł*" [ref=e89]: Centrum Medyczne Ujastek w Krakowie (żywienie w szpitalach) - wniosek epuap30.10.2019
        - generic [ref=e90]:
          - generic [ref=e91]: Treść*
          - generic [ref=e92]:
            - generic [ref=e93]: Teraz
            - link "Centrum_Medyczne_Ujastek_w_Krakowie_żywienie_w_szpitalach__-_wniosek_epuap30.10.2019.pdf" [ref=e96] [cursor=pointer]:
              - /url: /media/Centrum_Medyczne_Ujastek_w_Krakowie_%C5%BCywienie_w_szpitalach__-_wniosek_epuap30.10.2019.pdf
          - button "Treść*" [ref=e98] [cursor=pointer]
        - generic [ref=e99]:
          - generic [ref=e100]: Adresat*
          - combobox [aria-hidden] [ref=e101]
          - combobox [ref=e104] [cursor=pointer]:
            - textbox "CENTRUM MEDYCZNE UJASTEK SPÓŁKA Z OGRANICZONĄ ODPOWIEDZIALNOŚCIĄ" [ref=e105]
        - generic [ref=e106]:
          - generic [ref=e107]: Data wysłania*
          - textbox "Data wysłania*" [ref=e108]: 30.10.2019
        - button "Aktualizuj" [ref=e111] [cursor=pointer]
      - generic [ref=e112]:
        - generic [ref=e113]:
          - text: Sieć Obywatelska - Watchdog Polska ul. Szpitalna 5/5 00-031 Warszawa
          - paragraph [ref=e114]:
            - link "Klauzula RODO" [ref=e115] [cursor=pointer]:
              - /url: https://fedrowanie.siecobywatelska.pl/media_internal/tinycontent/uploads/11_KLAUZULA.-.dane.z.udip.na.stronach.www.i.w.mediach.spol.pdf
        - generic [ref=e116]:
          - text: "silnik:"
          - link "jawne.info.pl" [ref=e117] [cursor=pointer]:
            - /url: http://jawne.info.pl
          - text: "| v1.5.77.deps |"
          - link "GitHub" [ref=e118] [cursor=pointer]:
            - /url: https://github.com/watchdogpolska/feder
          - text: –
          - link "efd8a3b" [ref=e120] [cursor=pointer]:
            - /url: https://github.com/watchdogpolska/feder/compare/efd8a3b2...master
          - text: "|"
          - link "API" [ref=e121] [cursor=pointer]:
            - /url: /api/
        - generic [ref=e123]: Ta strona wykorzystuje cookies.
  - list [ref=e125]:
    - listitem [ref=e126]:
      - link "Ukryj »" [ref=e127] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e128]:
      - link "Toggle Theme" [ref=e129] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e132]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e133]
      - link "Historia /przesylki/outgoing-1/~update" [ref=e134] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e135]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e136]
      - link "Wersje Django 5.2.17" [ref=e137] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e138]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e139]
      - 'link "Czas CPU: 136.41ms (133.92ms)" [ref=e140] [cursor=pointer]':
        - /url: "#"
    - listitem [ref=e141]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e142]
      - link "Ustawienia" [ref=e143] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e144]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e145]
      - link "Nagłówki" [ref=e146] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e147]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e148]
      - link "Zapytania OutgoingParcelPostUpdateView" [ref=e149] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e150]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e151]
      - link "SQL 11 queries in 3.59ms" [ref=e152] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e153]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e154]
      - link "Pliki statyczne 10 użytych plików" [ref=e155] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e156]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e157]
      - link "Templatki parcels/outgoingparcelpost_form.html" [ref=e158] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e159]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e160]
      - link "Alerty" [ref=e161] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e162]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e163]
      - link "Cache 2 wywołania w 0.13ms" [ref=e164] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e165]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e166]
      - link "Sygnały 88 odbiorców 15 sygnałów" [ref=e167] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e168]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e169]
      - link "Gmina" [ref=e170] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e171]:
      - checkbox "Enable for next and successive requests" [ref=e172]
      - generic [ref=e173]: Przechwycone przekierowania
    - listitem [ref=e174]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e175]
      - link "Profilowanie" [ref=e176] [cursor=pointer]:
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