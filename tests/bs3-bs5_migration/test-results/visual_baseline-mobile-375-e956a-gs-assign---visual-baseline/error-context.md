# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: visual_baseline.spec.ts >> mobile (375px) >> monitorings-assign - visual baseline
- Location: tests/bs3-bs5_migration/visual_baseline.spec.ts:10:11

# Error details

```
Error: expect(page).toHaveScreenshot(expected) failed

  Expected an image 386px by 1986px, received 383px by 2072px. 447265 pixels (ratio 0.56 of all image pixels) are different.

  Snapshot: monitorings-assign-mobile.png

Call log:
  - Expect "toHaveScreenshot(monitorings-assign-mobile.png)" with timeout 5000ms
    - verifying given screenshot expectation
  - taking page screenshot
    - disabled all CSS animations
  - waiting for fonts to load...
  - fonts loaded
  - Expected an image 386px by 1986px, received 383px by 2072px. 447265 pixels (ratio 0.56 of all image pixels) are different.
  - waiting 100ms before taking screenshot
  - taking page screenshot
    - disabled all CSS animations
  - waiting for fonts to load...
  - fonts loaded
  - captured a stable screenshot
  - Expected an image 386px by 1986px, received 383px by 2072px. 447265 pixels (ratio 0.56 of all image pixels) are different.

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
        - listitem [ref=e72]: / Przypisz instytucje
      - heading "Przypisz instytucje" [level=1] [ref=e74]
      - generic [ref=e75]:
        - generic [ref=e78]:
          - generic [ref=e79]:
            - generic [ref=e80]: Nazwa zawiera
            - textbox "Nazwa zawiera" [ref=e81]
          - generic [ref=e82]:
            - generic [ref=e83]: Kod REGON
            - textbox "Kod REGON" [ref=e84]
          - generic [ref=e85]:
            - generic [ref=e86]: Tagi
            - listbox [aria-hidden] [ref=e87]
            - combobox [ref=e90]:
              - list [ref=e91]:
                - listitem [ref=e92]:
                  - searchbox [ref=e93]
          - generic [ref=e94]:
            - generic [ref=e95]: Instytucja archiwalna
            - combobox "Instytucja archiwalna" [ref=e96]:
              - option "Nieznany" [selected]
              - option "Tak"
              - option "Nie"
          - generic [ref=e97]:
            - generic [ref=e98]: Jednostka podziału terytorialnego active
            - combobox "Jednostka podziału terytorialnego active" [ref=e99]:
              - option "Nieznany" [selected]
              - option "Tak"
              - option "Nie"
          - generic [ref=e100]:
            - generic [ref=e101]: Metoda filtrowania tagów
            - combobox "Metoda filtrowania tagów" [ref=e102]:
              - option "AND"
              - option "OR" [selected]
          - generic [ref=e103]:
            - generic [ref=e104]: Województwa
            - combobox [aria-hidden] [ref=e105]
            - combobox [ref=e108] [cursor=pointer]:
              - textbox
          - generic [ref=e109]:
            - generic [ref=e110]: Powiat
            - combobox [aria-hidden] [ref=e111]
            - combobox [ref=e114] [cursor=pointer]:
              - textbox
          - generic [ref=e115]:
            - generic [ref=e116]: Gmina
            - combobox [aria-hidden] [ref=e117]
            - combobox [ref=e120] [cursor=pointer]:
              - textbox
          - button "Filtruj" [ref=e121] [cursor=pointer]
        - generic [ref=e124]:
          - button "Zastosuj filtr, aby przypisać i wysłać listy do instytucji." [disabled]
          - table [ref=e125]:
            - rowgroup [ref=e126]:
              - row [ref=e127]:
                - columnheader [ref=e128]
                - 'columnheader "Nazwa, wybrane: 0" [ref=e129]'
                - columnheader "Region" [ref=e130]
          - generic [ref=e131]:
            - table:
              - rowgroup
      - generic [ref=e132]:
        - generic [ref=e133]:
          - text: Sieć Obywatelska - Watchdog Polska ul. Szpitalna 5/5 00-031 Warszawa
          - paragraph [ref=e134]:
            - link "Klauzula RODO" [ref=e135] [cursor=pointer]:
              - /url: https://fedrowanie.siecobywatelska.pl/media_internal/tinycontent/uploads/11_KLAUZULA.-.dane.z.udip.na.stronach.www.i.w.mediach.spol.pdf
        - generic [ref=e136]:
          - text: "silnik:"
          - link "jawne.info.pl" [ref=e137] [cursor=pointer]:
            - /url: http://jawne.info.pl
          - text: "| v1.5.77.deps |"
          - link "GitHub" [ref=e138] [cursor=pointer]:
            - /url: https://github.com/watchdogpolska/feder
          - text: –
          - link "efd8a3b" [ref=e140] [cursor=pointer]:
            - /url: https://github.com/watchdogpolska/feder/compare/efd8a3b2...master
          - text: "|"
          - link "API" [ref=e141] [cursor=pointer]:
            - /url: /api/
        - generic [ref=e143]: Ta strona wykorzystuje cookies.
  - list [ref=e145]:
    - listitem [ref=e146]:
      - link "Ukryj »" [ref=e147] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e148]:
      - link "Toggle Theme" [ref=e149] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e152]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e153]
      - link "Historia /monitoringi/monitoring-sadow-apelacyjnych/~przypisz" [ref=e154] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e155]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e156]
      - link "Wersje Django 5.2.17" [ref=e157] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e158]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e159]
      - 'link "Czas CPU: 133.11ms (133.13ms)" [ref=e160] [cursor=pointer]':
        - /url: "#"
    - listitem [ref=e161]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e162]
      - link "Ustawienia" [ref=e163] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e164]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e165]
      - link "Nagłówki" [ref=e166] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e167]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e168]
      - link "Zapytania MonitoringAssignView" [ref=e169] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e170]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e171]
      - link "SQL 7 queries in 3.89ms" [ref=e172] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e173]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e174]
      - link "Pliki statyczne 10 użytych plików" [ref=e175] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e176]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e177]
      - link "Templatki monitorings/institution_assign.html" [ref=e178] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e179]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e180]
      - link "Alerty" [ref=e181] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e182]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e183]
      - link "Cache 2 wywołania w 0.17ms" [ref=e184] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e185]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e186]
      - link "Sygnały 88 odbiorców 15 sygnałów" [ref=e187] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e188]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e189]
      - link "Gmina" [ref=e190] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e191]:
      - checkbox "Enable for next and successive requests" [ref=e192]
      - generic [ref=e193]: Przechwycone przekierowania
    - listitem [ref=e194]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e195]
      - link "Profilowanie" [ref=e196] [cursor=pointer]:
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