# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: visual_baseline.spec.ts >> desktop (2560px) >> cases-list - visual baseline
- Location: tests/bs3-bs5_migration/visual_baseline.spec.ts:10:11

# Error details

```
Error: expect(page).toHaveScreenshot(expected) failed

  517764 pixels (ratio 0.15 of all image pixels) are different.

  Snapshot: cases-list-desktop.png

Call log:
  - Expect "toHaveScreenshot(cases-list-desktop.png)" with timeout 5000ms
    - verifying given screenshot expectation
  - taking page screenshot
    - disabled all CSS animations
  - waiting for fonts to load...
  - fonts loaded
  - 517764 pixels (ratio 0.15 of all image pixels) are different.
  - waiting 100ms before taking screenshot
  - taking page screenshot
    - disabled all CSS animations
  - waiting for fonts to load...
  - fonts loaded
  - captured a stable screenshot
  - 517764 pixels (ratio 0.15 of all image pixels) are different.

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
      - heading "Wykaz spraw" [level=2] [ref=e62]
      - generic [ref=e63]:
        - generic [ref=e66]:
          - generic [ref=e67]:
            - generic [ref=e68]: Nazwa
            - textbox "Nazwa" [ref=e69]
          - generic [ref=e70]:
            - generic [ref=e71]: Monitoring
            - combobox [aria-hidden] [ref=e72]
            - combobox [ref=e75] [cursor=pointer]:
              - textbox
          - generic [ref=e76]:
            - generic [ref=e77]: Instytucja
            - combobox [aria-hidden] [ref=e78]
            - combobox [ref=e81] [cursor=pointer]:
              - textbox
          - generic [ref=e82]:
            - generic [ref=e83]: Data utworzenia
            - combobox "Data utworzenia" [ref=e84]:
              - option "---------"
              - option "Dzisiaj"
              - option "Wczoraj"
              - option "Ostatnie 7 dni"
              - option "Ten miesiąc"
              - option "Ten rok" [selected]
              - option "2025"
              - option "2024"
              - option "2023"
              - option "2022"
              - option "2021"
              - option "2020"
              - option "2019"
              - option "2018"
              - option "2017"
              - option "2016"
          - generic [ref=e85]:
            - generic [ref=e86]: Otrzymano potwierdzenie
            - combobox "Otrzymano potwierdzenie" [ref=e87]:
              - option "Nieznany" [selected]
              - option "Tak"
              - option "Nie"
          - generic [ref=e88]:
            - generic [ref=e89]: Otrzymano odpowiedź
            - combobox "Otrzymano odpowiedź" [ref=e90]:
              - option "Nieznany" [selected]
              - option "Tak"
              - option "Nie"
          - generic [ref=e91]:
            - generic [ref=e92]: Województwa
            - combobox [aria-hidden] [ref=e93]
            - combobox [ref=e96] [cursor=pointer]:
              - textbox
          - generic [ref=e97]:
            - generic [ref=e98]: Powiat
            - combobox [aria-hidden] [ref=e99]
            - combobox [ref=e102] [cursor=pointer]:
              - textbox
          - generic [ref=e103]:
            - generic [ref=e104]: Gmina
            - combobox [aria-hidden] [ref=e105]
            - combobox [ref=e108] [cursor=pointer]:
              - textbox
          - button "Filtruj" [ref=e109] [cursor=pointer]
        - generic [ref=e111]:
          - paragraph [ref=e113]: Brak wierszy.
          - list [ref=e114]:
            - listitem [ref=e115]:
              - generic [aria-hidden]: ←
            - listitem [ref=e116]:
              - generic "Current Page" [ref=e117]: "1"
            - listitem [ref=e118]:
              - generic [aria-hidden]: →
      - generic [ref=e119]:
        - generic [ref=e120]:
          - text: Sieć Obywatelska - Watchdog Polska ul. Szpitalna 5/5 00-031 Warszawa
          - paragraph [ref=e121]:
            - link "Klauzula RODO" [ref=e122] [cursor=pointer]:
              - /url: https://fedrowanie.siecobywatelska.pl/media_internal/tinycontent/uploads/11_KLAUZULA.-.dane.z.udip.na.stronach.www.i.w.mediach.spol.pdf
        - generic [ref=e123]:
          - text: "silnik:"
          - link "jawne.info.pl" [ref=e124] [cursor=pointer]:
            - /url: http://jawne.info.pl
          - text: "| v1.5.77.deps |"
          - link "GitHub" [ref=e125] [cursor=pointer]:
            - /url: https://github.com/watchdogpolska/feder
          - text: –
          - link "efd8a3b" [ref=e127] [cursor=pointer]:
            - /url: https://github.com/watchdogpolska/feder/compare/efd8a3b2...master
          - text: "|"
          - link "API" [ref=e128] [cursor=pointer]:
            - /url: /api/
        - generic [ref=e130]: Ta strona wykorzystuje cookies.
  - list [ref=e132]:
    - listitem [ref=e133]:
      - link "Ukryj »" [ref=e134] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e135]:
      - link "Toggle Theme" [ref=e136] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e139]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e140]
      - link "Historia /sprawy/" [ref=e141] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e142]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e143]
      - link "Wersje Django 5.2.17" [ref=e144] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e145]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e146]
      - 'link "Czas CPU: 157.83ms (153.79ms)" [ref=e147] [cursor=pointer]':
        - /url: "#"
    - listitem [ref=e148]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e149]
      - link "Ustawienia" [ref=e150] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e151]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e152]
      - link "Nagłówki" [ref=e153] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e154]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e155]
      - link "Zapytania CaseListView" [ref=e156] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e157]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e158]
      - link "SQL 5 queries in 2.11ms" [ref=e159] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e160]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e161]
      - link "Pliki statyczne 10 użytych plików" [ref=e162] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e163]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e164]
      - link "Templatki cases/case_filter.html" [ref=e165] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e166]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e167]
      - link "Alerty" [ref=e168] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e169]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e170]
      - link "Cache 2 wywołania w 0.15ms" [ref=e171] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e172]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e173]
      - link "Sygnały 88 odbiorców 15 sygnałów" [ref=e174] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e175]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e176]
      - link "Gmina" [ref=e177] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e178]:
      - checkbox "Enable for next and successive requests" [ref=e179]
      - generic [ref=e180]: Przechwycone przekierowania
    - listitem [ref=e181]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e182]
      - link "Profilowanie" [ref=e183] [cursor=pointer]:
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