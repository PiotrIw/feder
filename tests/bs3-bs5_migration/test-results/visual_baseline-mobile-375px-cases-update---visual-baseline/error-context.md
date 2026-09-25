# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: visual_baseline.spec.ts >> mobile (375px) >> cases-update - visual baseline
- Location: tests/bs3-bs5_migration/visual_baseline.spec.ts:10:11

# Error details

```
Error: expect(page).toHaveScreenshot(expected) failed

  Expected an image 386px by 1076px, received 383px by 1049px. 181061 pixels (ratio 0.44 of all image pixels) are different.

  Snapshot: cases-update-mobile.png

Call log:
  - Expect "toHaveScreenshot(cases-update-mobile.png)" with timeout 5000ms
    - verifying given screenshot expectation
  - taking page screenshot
    - disabled all CSS animations
  - waiting for fonts to load...
  - fonts loaded
  - Expected an image 386px by 1076px, received 383px by 1049px. 181061 pixels (ratio 0.44 of all image pixels) are different.
  - waiting 100ms before taking screenshot
  - taking page screenshot
    - disabled all CSS animations
  - waiting for fonts to load...
  - fonts loaded
  - captured a stable screenshot
  - Expected an image 386px by 1076px, received 383px by 1049px. 181061 pixels (ratio 0.44 of all image pixels) are different.

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
        - listitem [ref=e72]: / Zaktualizuj sprawę
      - generic [ref=e74]:
        - link "Edytuj" [ref=e75] [cursor=pointer]:
          - /url: /sprawy/monitoring-sadow-apelacyjnych-1/~edytuj
        - link "Usuń" [ref=e76] [cursor=pointer]:
          - /url: /sprawy/monitoring-sadow-apelacyjnych-1/~usun
        - link "Zobacz dzienniki" [ref=e77] [cursor=pointer]:
          - /url: /listy/logi/spraw-2684
        - button "Dodaj przesyłkę pocztową" [ref=e79] [cursor=pointer]
        - link "Dodaj list" [ref=e81] [cursor=pointer]:
          - /url: /listy/~utworz-2684
      - 'heading "Monitoring sądów apelacyjnych #1" [level=1] [ref=e84]'
      - generic [ref=e88]:
        - generic [ref=e89]:
          - generic [ref=e90]: Nazwa*
          - textbox "Nazwa*" [ref=e91]: "Monitoring sądów apelacyjnych #1"
        - generic [ref=e92]:
          - generic [ref=e93]: Instytucja*
          - combobox [aria-hidden] [ref=e94]
          - combobox [ref=e97] [cursor=pointer]:
            - textbox "Sąd Apelacyjny w Białymstoku" [ref=e98]
        - generic [ref=e100]:
          - checkbox "Poddany kwarantannie" [ref=e101]
          - generic [ref=e102]: Poddany kwarantannie
        - generic [ref=e104]:
          - checkbox "Otrzymano potwierdzenie" [ref=e105]
          - generic [ref=e106]: Otrzymano potwierdzenie
        - generic [ref=e108]:
          - checkbox "Otrzymano odpowiedź" [checked] [ref=e109]
          - generic [ref=e110]: Otrzymano odpowiedź
        - group "Tagi" [ref=e112]:
          - generic [ref=e115]:
            - checkbox "test" [ref=e116]
            - generic [ref=e117]: test
        - button "Aktualizuj" [ref=e120] [cursor=pointer]
      - generic [ref=e121]:
        - generic [ref=e122]:
          - text: Sieć Obywatelska - Watchdog Polska ul. Szpitalna 5/5 00-031 Warszawa
          - paragraph [ref=e123]:
            - link "Klauzula RODO" [ref=e124] [cursor=pointer]:
              - /url: https://fedrowanie.siecobywatelska.pl/media_internal/tinycontent/uploads/11_KLAUZULA.-.dane.z.udip.na.stronach.www.i.w.mediach.spol.pdf
        - generic [ref=e125]:
          - text: "silnik:"
          - link "jawne.info.pl" [ref=e126] [cursor=pointer]:
            - /url: http://jawne.info.pl
          - text: "| v1.5.77.deps |"
          - link "GitHub" [ref=e127] [cursor=pointer]:
            - /url: https://github.com/watchdogpolska/feder
          - text: –
          - link "efd8a3b" [ref=e129] [cursor=pointer]:
            - /url: https://github.com/watchdogpolska/feder/compare/efd8a3b2...master
          - text: "|"
          - link "API" [ref=e130] [cursor=pointer]:
            - /url: /api/
        - generic [ref=e132]: Ta strona wykorzystuje cookies.
  - list [ref=e134]:
    - listitem [ref=e135]:
      - link "Ukryj »" [ref=e136] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e137]:
      - link "Toggle Theme" [ref=e138] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e141]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e142]
      - link "Historia /sprawy/monitoring-sadow-apelacyjnych-1/~edytuj" [ref=e143] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e144]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e145]
      - link "Wersje Django 5.2.17" [ref=e146] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e147]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e148]
      - 'link "Czas CPU: 171.12ms (162.93ms)" [ref=e149] [cursor=pointer]':
        - /url: "#"
    - listitem [ref=e150]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e151]
      - link "Ustawienia" [ref=e152] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e153]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e154]
      - link "Nagłówki" [ref=e155] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e156]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e157]
      - link "Zapytania CaseUpdateView" [ref=e158] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e159]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e160]
      - link "SQL 13 queries in 5.43ms" [ref=e161] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e162]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e163]
      - link "Pliki statyczne 10 użytych plików" [ref=e164] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e165]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e166]
      - link "Templatki cases/case_form.html" [ref=e167] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e168]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e169]
      - link "Alerty" [ref=e170] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e171]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e172]
      - link "Cache 2 wywołania w 0.13ms" [ref=e173] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e174]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e175]
      - link "Sygnały 88 odbiorców 15 sygnałów" [ref=e176] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e177]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e178]
      - link "Gmina" [ref=e179] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e180]:
      - checkbox "Enable for next and successive requests" [ref=e181]
      - generic [ref=e182]: Przechwycone przekierowania
    - listitem [ref=e183]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e184]
      - link "Profilowanie" [ref=e185] [cursor=pointer]:
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