# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: visual_baseline.spec.ts >> desktop (2560px) >> monitorings-assign - visual baseline
- Location: tests/bs3-bs5_migration/visual_baseline.spec.ts:10:11

# Error details

```
Error: expect(page).toHaveScreenshot(expected) failed

  611731 pixels (ratio 0.17 of all image pixels) are different.

  Snapshot: monitorings-assign-desktop.png

Call log:
  - Expect "toHaveScreenshot(monitorings-assign-desktop.png)" with timeout 5000ms
    - verifying given screenshot expectation
  - taking page screenshot
    - disabled all CSS animations
  - waiting for fonts to load...
  - fonts loaded
  - 611731 pixels (ratio 0.17 of all image pixels) are different.
  - waiting 100ms before taking screenshot
  - taking page screenshot
    - disabled all CSS animations
  - waiting for fonts to load...
  - fonts loaded
  - captured a stable screenshot
  - 611731 pixels (ratio 0.17 of all image pixels) are different.

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
        - listitem [ref=e64]: / Przypisz instytucje
      - heading "Przypisz instytucje" [level=1] [ref=e66]
      - generic [ref=e67]:
        - generic [ref=e70]:
          - generic [ref=e71]:
            - generic [ref=e72]: Nazwa zawiera
            - textbox "Nazwa zawiera" [ref=e73]
          - generic [ref=e74]:
            - generic [ref=e75]: Kod REGON
            - textbox "Kod REGON" [ref=e76]
          - generic [ref=e77]:
            - generic [ref=e78]: Tagi
            - listbox [aria-hidden] [ref=e79]
            - combobox [ref=e82]:
              - list [ref=e83]:
                - listitem [ref=e84]:
                  - searchbox [ref=e85]
          - generic [ref=e86]:
            - generic [ref=e87]: Instytucja archiwalna
            - combobox "Instytucja archiwalna" [ref=e88]:
              - option "Nieznany" [selected]
              - option "Tak"
              - option "Nie"
          - generic [ref=e89]:
            - generic [ref=e90]: Jednostka podziału terytorialnego active
            - combobox "Jednostka podziału terytorialnego active" [ref=e91]:
              - option "Nieznany" [selected]
              - option "Tak"
              - option "Nie"
          - generic [ref=e92]:
            - generic [ref=e93]: Metoda filtrowania tagów
            - combobox "Metoda filtrowania tagów" [ref=e94]:
              - option "AND"
              - option "OR" [selected]
          - generic [ref=e95]:
            - generic [ref=e96]: Województwa
            - combobox [aria-hidden] [ref=e97]
            - combobox [ref=e100] [cursor=pointer]:
              - textbox
          - generic [ref=e101]:
            - generic [ref=e102]: Powiat
            - combobox [aria-hidden] [ref=e103]
            - combobox [ref=e106] [cursor=pointer]:
              - textbox
          - generic [ref=e107]:
            - generic [ref=e108]: Gmina
            - combobox [aria-hidden] [ref=e109]
            - combobox [ref=e112] [cursor=pointer]:
              - textbox
          - button "Filtruj" [ref=e113] [cursor=pointer]
        - generic [ref=e116]:
          - button "Zastosuj filtr, aby przypisać i wysłać listy do instytucji." [disabled]
          - table [ref=e117]:
            - rowgroup [ref=e118]:
              - row [ref=e119]:
                - columnheader [ref=e120]
                - 'columnheader "Nazwa, wybrane: 0" [ref=e121]'
                - columnheader "Region" [ref=e122]
          - generic [ref=e123]:
            - table:
              - rowgroup
      - generic [ref=e124]:
        - generic [ref=e125]:
          - text: Sieć Obywatelska - Watchdog Polska ul. Szpitalna 5/5 00-031 Warszawa
          - paragraph [ref=e126]:
            - link "Klauzula RODO" [ref=e127] [cursor=pointer]:
              - /url: https://fedrowanie.siecobywatelska.pl/media_internal/tinycontent/uploads/11_KLAUZULA.-.dane.z.udip.na.stronach.www.i.w.mediach.spol.pdf
        - generic [ref=e128]:
          - text: "silnik:"
          - link "jawne.info.pl" [ref=e129] [cursor=pointer]:
            - /url: http://jawne.info.pl
          - text: "| v1.5.77.deps |"
          - link "GitHub" [ref=e130] [cursor=pointer]:
            - /url: https://github.com/watchdogpolska/feder
          - text: –
          - link "efd8a3b" [ref=e132] [cursor=pointer]:
            - /url: https://github.com/watchdogpolska/feder/compare/efd8a3b2...master
          - text: "|"
          - link "API" [ref=e133] [cursor=pointer]:
            - /url: /api/
        - generic [ref=e135]: Ta strona wykorzystuje cookies.
  - list [ref=e137]:
    - listitem [ref=e138]:
      - link "Ukryj »" [ref=e139] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e140]:
      - link "Toggle Theme" [ref=e141] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e144]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e145]
      - link "Historia /monitoringi/monitoring-sadow-apelacyjnych/~przypisz" [ref=e146] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e147]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e148]
      - link "Wersje Django 5.2.17" [ref=e149] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e150]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e151]
      - 'link "Czas CPU: 128.54ms (132.12ms)" [ref=e152] [cursor=pointer]':
        - /url: "#"
    - listitem [ref=e153]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e154]
      - link "Ustawienia" [ref=e155] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e156]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e157]
      - link "Nagłówki" [ref=e158] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e159]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e160]
      - link "Zapytania MonitoringAssignView" [ref=e161] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e162]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e163]
      - link "SQL 7 queries in 3.77ms" [ref=e164] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e165]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e166]
      - link "Pliki statyczne 10 użytych plików" [ref=e167] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e168]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e169]
      - link "Templatki monitorings/institution_assign.html" [ref=e170] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e171]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e172]
      - link "Alerty" [ref=e173] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e174]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e175]
      - link "Cache 2 wywołania w 0.14ms" [ref=e176] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e177]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e178]
      - link "Sygnały 88 odbiorców 15 sygnałów" [ref=e179] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e180]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e181]
      - link "Gmina" [ref=e182] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e183]:
      - checkbox "Enable for next and successive requests" [ref=e184]
      - generic [ref=e185]: Przechwycone przekierowania
    - listitem [ref=e186]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e187]
      - link "Profilowanie" [ref=e188] [cursor=pointer]:
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