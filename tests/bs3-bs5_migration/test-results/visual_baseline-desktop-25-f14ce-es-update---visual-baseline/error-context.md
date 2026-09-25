# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: visual_baseline.spec.ts >> desktop (2560px) >> cases-update - visual baseline
- Location: tests/bs3-bs5_migration/visual_baseline.spec.ts:10:11

# Error details

```
Error: expect(page).toHaveScreenshot(expected) failed

  354115 pixels (ratio 0.10 of all image pixels) are different.

  Snapshot: cases-update-desktop.png

Call log:
  - Expect "toHaveScreenshot(cases-update-desktop.png)" with timeout 5000ms
    - verifying given screenshot expectation
  - taking page screenshot
    - disabled all CSS animations
  - waiting for fonts to load...
  - fonts loaded
  - 354114 pixels (ratio 0.10 of all image pixels) are different.
  - waiting 100ms before taking screenshot
  - taking page screenshot
    - disabled all CSS animations
  - waiting for fonts to load...
  - fonts loaded
  - captured a stable screenshot
  - 354115 pixels (ratio 0.10 of all image pixels) are different.

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
        - listitem [ref=e64]: / Zaktualizuj sprawę
      - generic [ref=e66]:
        - link "Edytuj" [ref=e67] [cursor=pointer]:
          - /url: /sprawy/monitoring-sadow-apelacyjnych-1/~edytuj
        - link "Usuń" [ref=e68] [cursor=pointer]:
          - /url: /sprawy/monitoring-sadow-apelacyjnych-1/~usun
        - link "Zobacz dzienniki" [ref=e69] [cursor=pointer]:
          - /url: /listy/logi/spraw-2684
        - button "Dodaj przesyłkę pocztową" [ref=e71] [cursor=pointer]
        - link "Dodaj list" [ref=e73] [cursor=pointer]:
          - /url: /listy/~utworz-2684
      - 'heading "Monitoring sądów apelacyjnych #1" [level=1] [ref=e76]'
      - generic [ref=e80]:
        - generic [ref=e81]:
          - generic [ref=e82]: Nazwa*
          - textbox "Nazwa*" [ref=e83]: "Monitoring sądów apelacyjnych #1"
        - generic [ref=e84]:
          - generic [ref=e85]: Instytucja*
          - combobox [aria-hidden] [ref=e86]
          - combobox [ref=e89] [cursor=pointer]:
            - textbox "Sąd Apelacyjny w Białymstoku" [ref=e90]
        - generic [ref=e92]:
          - checkbox "Poddany kwarantannie" [ref=e93]
          - generic [ref=e94]: Poddany kwarantannie
        - generic [ref=e96]:
          - checkbox "Otrzymano potwierdzenie" [ref=e97]
          - generic [ref=e98]: Otrzymano potwierdzenie
        - generic [ref=e100]:
          - checkbox "Otrzymano odpowiedź" [checked] [ref=e101]
          - generic [ref=e102]: Otrzymano odpowiedź
        - group "Tagi" [ref=e104]:
          - generic [ref=e107]:
            - checkbox "test" [ref=e108]
            - generic [ref=e109]: test
        - button "Aktualizuj" [ref=e112] [cursor=pointer]
      - generic [ref=e113]:
        - generic [ref=e114]:
          - text: Sieć Obywatelska - Watchdog Polska ul. Szpitalna 5/5 00-031 Warszawa
          - paragraph [ref=e115]:
            - link "Klauzula RODO" [ref=e116] [cursor=pointer]:
              - /url: https://fedrowanie.siecobywatelska.pl/media_internal/tinycontent/uploads/11_KLAUZULA.-.dane.z.udip.na.stronach.www.i.w.mediach.spol.pdf
        - generic [ref=e117]:
          - text: "silnik:"
          - link "jawne.info.pl" [ref=e118] [cursor=pointer]:
            - /url: http://jawne.info.pl
          - text: "| v1.5.77.deps |"
          - link "GitHub" [ref=e119] [cursor=pointer]:
            - /url: https://github.com/watchdogpolska/feder
          - text: –
          - link "efd8a3b" [ref=e121] [cursor=pointer]:
            - /url: https://github.com/watchdogpolska/feder/compare/efd8a3b2...master
          - text: "|"
          - link "API" [ref=e122] [cursor=pointer]:
            - /url: /api/
        - generic [ref=e124]: Ta strona wykorzystuje cookies.
  - list [ref=e126]:
    - listitem [ref=e127]:
      - link "Ukryj »" [ref=e128] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e129]:
      - link "Toggle Theme" [ref=e130] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e133]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e134]
      - link "Historia /sprawy/monitoring-sadow-apelacyjnych-1/~edytuj" [ref=e135] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e136]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e137]
      - link "Wersje Django 5.2.17" [ref=e138] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e139]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e140]
      - 'link "Czas CPU: 153.60ms (158.14ms)" [ref=e141] [cursor=pointer]':
        - /url: "#"
    - listitem [ref=e142]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e143]
      - link "Ustawienia" [ref=e144] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e145]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e146]
      - link "Nagłówki" [ref=e147] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e148]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e149]
      - link "Zapytania CaseUpdateView" [ref=e150] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e151]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e152]
      - link "SQL 13 queries in 5.31ms" [ref=e153] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e154]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e155]
      - link "Pliki statyczne 10 użytych plików" [ref=e156] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e157]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e158]
      - link "Templatki cases/case_form.html" [ref=e159] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e160]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e161]
      - link "Alerty" [ref=e162] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e163]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e164]
      - link "Cache 2 wywołania w 0.14ms" [ref=e165] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e166]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e167]
      - link "Sygnały 88 odbiorców 15 sygnałów" [ref=e168] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e169]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e170]
      - link "Gmina" [ref=e171] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e172]:
      - checkbox "Enable for next and successive requests" [ref=e173]
      - generic [ref=e174]: Przechwycone przekierowania
    - listitem [ref=e175]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e176]
      - link "Profilowanie" [ref=e177] [cursor=pointer]:
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