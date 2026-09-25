# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: visual_baseline.spec.ts >> mobile (375px) >> cases-create - visual baseline
- Location: tests/bs3-bs5_migration/visual_baseline.spec.ts:10:11

# Error details

```
Error: expect(page).toHaveScreenshot(expected) failed

  Expected an image 386px by 828px, received 383px by 889px. 175234 pixels (ratio 0.52 of all image pixels) are different.

  Snapshot: cases-create-mobile.png

Call log:
  - Expect "toHaveScreenshot(cases-create-mobile.png)" with timeout 5000ms
    - verifying given screenshot expectation
  - taking page screenshot
    - disabled all CSS animations
  - waiting for fonts to load...
  - fonts loaded
  - Expected an image 386px by 828px, received 383px by 889px. 175234 pixels (ratio 0.52 of all image pixels) are different.
  - waiting 100ms before taking screenshot
  - taking page screenshot
    - disabled all CSS animations
  - waiting for fonts to load...
  - fonts loaded
  - captured a stable screenshot
  - Expected an image 386px by 828px, received 383px by 889px. 175234 pixels (ratio 0.52 of all image pixels) are different.

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
        - listitem [ref=e72]: / Utwórz sprawę
      - generic [ref=e75]:
        - generic [ref=e76]:
          - generic [ref=e77]: Nazwa*
          - textbox "Nazwa*" [ref=e78]
        - generic [ref=e79]:
          - generic [ref=e80]: Instytucja*
          - combobox [aria-hidden] [ref=e81]
          - combobox [ref=e84] [cursor=pointer]:
            - textbox
        - generic [ref=e86]:
          - checkbox "Poddany kwarantannie" [ref=e87]
          - generic [ref=e88]: Poddany kwarantannie
        - generic [ref=e90]:
          - checkbox "Otrzymano potwierdzenie" [ref=e91]
          - generic [ref=e92]: Otrzymano potwierdzenie
        - generic [ref=e94]:
          - checkbox "Otrzymano odpowiedź" [ref=e95]
          - generic [ref=e96]: Otrzymano odpowiedź
        - group "Tagi" [ref=e98]:
          - generic [ref=e101]:
            - checkbox "test" [ref=e102]
            - generic [ref=e103]: test
        - button "Zapisz" [ref=e106] [cursor=pointer]
      - generic [ref=e107]:
        - generic [ref=e108]:
          - text: Sieć Obywatelska - Watchdog Polska ul. Szpitalna 5/5 00-031 Warszawa
          - paragraph [ref=e109]:
            - link "Klauzula RODO" [ref=e110] [cursor=pointer]:
              - /url: https://fedrowanie.siecobywatelska.pl/media_internal/tinycontent/uploads/11_KLAUZULA.-.dane.z.udip.na.stronach.www.i.w.mediach.spol.pdf
        - generic [ref=e111]:
          - text: "silnik:"
          - link "jawne.info.pl" [ref=e112] [cursor=pointer]:
            - /url: http://jawne.info.pl
          - text: "| v1.5.77.deps |"
          - link "GitHub" [ref=e113] [cursor=pointer]:
            - /url: https://github.com/watchdogpolska/feder
          - text: –
          - link "efd8a3b" [ref=e115] [cursor=pointer]:
            - /url: https://github.com/watchdogpolska/feder/compare/efd8a3b2...master
          - text: "|"
          - link "API" [ref=e116] [cursor=pointer]:
            - /url: /api/
        - generic [ref=e118]: Ta strona wykorzystuje cookies.
  - list [ref=e120]:
    - listitem [ref=e121]:
      - link "Ukryj »" [ref=e122] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e123]:
      - link "Toggle Theme" [ref=e124] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e127]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e128]
      - link "Historia /sprawy/~utworz-5" [ref=e129] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e130]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e131]
      - link "Wersje Django 5.2.17" [ref=e132] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e133]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e134]
      - 'link "Czas CPU: 116.02ms (118.10ms)" [ref=e135] [cursor=pointer]':
        - /url: "#"
    - listitem [ref=e136]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e137]
      - link "Ustawienia" [ref=e138] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e139]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e140]
      - link "Nagłówki" [ref=e141] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e142]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e143]
      - link "Zapytania CaseCreateView" [ref=e144] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e145]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e146]
      - link "SQL 6 queries in 2.07ms" [ref=e147] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e148]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e149]
      - link "Pliki statyczne 10 użytych plików" [ref=e150] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e151]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e152]
      - link "Templatki cases/case_form.html" [ref=e153] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e154]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e155]
      - link "Alerty" [ref=e156] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e157]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e158]
      - link "Cache 2 wywołania w 0.13ms" [ref=e159] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e160]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e161]
      - link "Sygnały 88 odbiorców 15 sygnałów" [ref=e162] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e163]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e164]
      - link "Gmina" [ref=e165] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e166]:
      - checkbox "Enable for next and successive requests" [ref=e167]
      - generic [ref=e168]: Przechwycone przekierowania
    - listitem [ref=e169]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e170]
      - link "Profilowanie" [ref=e171] [cursor=pointer]:
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