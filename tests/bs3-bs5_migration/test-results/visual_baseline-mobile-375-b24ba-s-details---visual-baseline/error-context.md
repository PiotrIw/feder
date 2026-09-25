# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: visual_baseline.spec.ts >> mobile (375px) >> casetags-details - visual baseline
- Location: tests/bs3-bs5_migration/visual_baseline.spec.ts:10:11

# Error details

```
Error: expect(page).toHaveScreenshot(expected) failed

  Expected an image 386px by 812px, received 383px by 812px. 166326 pixels (ratio 0.54 of all image pixels) are different.

  Snapshot: casetags-details-mobile.png

Call log:
  - Expect "toHaveScreenshot(casetags-details-mobile.png)" with timeout 5000ms
    - verifying given screenshot expectation
  - taking page screenshot
    - disabled all CSS animations
  - waiting for fonts to load...
  - fonts loaded
  - Expected an image 386px by 812px, received 383px by 812px. 166326 pixels (ratio 0.54 of all image pixels) are different.
  - waiting 100ms before taking screenshot
  - taking page screenshot
    - disabled all CSS animations
  - waiting for fonts to load...
  - fonts loaded
  - captured a stable screenshot
  - Expected an image 386px by 812px, received 383px by 812px. 166326 pixels (ratio 0.54 of all image pixels) are different.

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
          - link "Jawność w spółkach komunalnych" [ref=e71] [cursor=pointer]:
            - /url: /monitoringi/jawnosc-w-spolkach-komunalnych
        - listitem [ref=e72]:
          - text: /
          - link "Wykaz tagów" [ref=e73] [cursor=pointer]:
            - /url: /sprawy/tagi/monitoring-44
        - listitem [ref=e74]:
          - text: /
          - link "procedury z wolnej ręki i dotacji" [ref=e75] [cursor=pointer]:
            - /url: /sprawy/tagi/monitoring-44/2
      - generic [ref=e77]:
        - link "Edytuj" [ref=e78] [cursor=pointer]:
          - /url: /sprawy/tagi/monitoring-44/2/~update
        - link "Usuń" [ref=e79] [cursor=pointer]:
          - /url: /sprawy/tagi/monitoring-44/2/~delete
      - heading "procedury z wolnej ręki i dotacji" [level=1] [ref=e81]
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
      - link "Historia /sprawy/tagi/monitoring-44/2" [ref=e107] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e108]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e109]
      - link "Wersje Django 5.2.17" [ref=e110] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e111]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e112]
      - 'link "Czas CPU: 74.33ms (76.89ms)" [ref=e113] [cursor=pointer]':
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
      - link "Zapytania TagDetailView" [ref=e122] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e123]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e124]
      - link "SQL 7 queries in 2.61ms" [ref=e125] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e126]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e127]
      - link "Pliki statyczne 3 użyte plików" [ref=e128] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e129]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e130]
      - link "Templatki cases_tags/tag_detail.html" [ref=e131] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e132]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e133]
      - link "Alerty" [ref=e134] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e135]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e136]
      - link "Cache 2 wywołania w 0.13ms" [ref=e137] [cursor=pointer]:
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