# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: visual_baseline.spec.ts >> desktop (2560px) >> casetags-create - visual baseline
- Location: tests/bs3-bs5_migration/visual_baseline.spec.ts:10:11

# Error details

```
Error: expect(page).toHaveScreenshot(expected) failed

  327621 pixels (ratio 0.09 of all image pixels) are different.

  Snapshot: casetags-create-desktop.png

Call log:
  - Expect "toHaveScreenshot(casetags-create-desktop.png)" with timeout 5000ms
    - verifying given screenshot expectation
  - taking page screenshot
    - disabled all CSS animations
  - waiting for fonts to load...
  - fonts loaded
  - 327621 pixels (ratio 0.09 of all image pixels) are different.
  - waiting 100ms before taking screenshot
  - taking page screenshot
    - disabled all CSS animations
  - waiting for fonts to load...
  - fonts loaded
  - captured a stable screenshot
  - 327621 pixels (ratio 0.09 of all image pixels) are different.

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
          - link "Jawność w spółkach komunalnych" [ref=e63] [cursor=pointer]:
            - /url: /monitoringi/jawnosc-w-spolkach-komunalnych
        - listitem [ref=e64]: / Dodaj tag
      - link "Utwórz" [ref=e67] [cursor=pointer]:
        - /url: /sprawy/tagi/monitoring-44/~create
      - generic [ref=e70]:
        - generic [ref=e71]:
          - generic [ref=e72]: Nazwa*
          - textbox "Nazwa*" [ref=e73]
        - button "Zapisz" [ref=e76] [cursor=pointer]
      - generic [ref=e77]:
        - generic [ref=e78]:
          - text: Sieć Obywatelska - Watchdog Polska ul. Szpitalna 5/5 00-031 Warszawa
          - paragraph [ref=e79]:
            - link "Klauzula RODO" [ref=e80] [cursor=pointer]:
              - /url: https://fedrowanie.siecobywatelska.pl/media_internal/tinycontent/uploads/11_KLAUZULA.-.dane.z.udip.na.stronach.www.i.w.mediach.spol.pdf
        - generic [ref=e81]:
          - text: "silnik:"
          - link "jawne.info.pl" [ref=e82] [cursor=pointer]:
            - /url: http://jawne.info.pl
          - text: "| v1.5.77.deps |"
          - link "GitHub" [ref=e83] [cursor=pointer]:
            - /url: https://github.com/watchdogpolska/feder
          - text: –
          - link "efd8a3b" [ref=e85] [cursor=pointer]:
            - /url: https://github.com/watchdogpolska/feder/compare/efd8a3b2...master
          - text: "|"
          - link "API" [ref=e86] [cursor=pointer]:
            - /url: /api/
        - generic [ref=e88]: Ta strona wykorzystuje cookies.
  - list [ref=e90]:
    - listitem [ref=e91]:
      - link "Ukryj »" [ref=e92] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e93]:
      - link "Toggle Theme" [ref=e94] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e97]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e98]
      - link "Historia /sprawy/tagi/monitoring-44/~create" [ref=e99] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e100]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e101]
      - link "Wersje Django 5.2.17" [ref=e102] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e103]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e104]
      - 'link "Czas CPU: 81.10ms (82.73ms)" [ref=e105] [cursor=pointer]':
        - /url: "#"
    - listitem [ref=e106]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e107]
      - link "Ustawienia" [ref=e108] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e109]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e110]
      - link "Nagłówki" [ref=e111] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e112]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e113]
      - link "Zapytania TagCreateView" [ref=e114] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e115]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e116]
      - link "SQL 6 queries in 1.72ms" [ref=e117] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e118]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e119]
      - link "Pliki statyczne 3 użyte plików" [ref=e120] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e121]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e122]
      - link "Templatki cases_tags/tag_form.html" [ref=e123] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e124]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e125]
      - link "Alerty" [ref=e126] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e127]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e128]
      - link "Cache 2 wywołania w 0.15ms" [ref=e129] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e130]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e131]
      - link "Sygnały 88 odbiorców 15 sygnałów" [ref=e132] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e133]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e134]
      - link "Gmina" [ref=e135] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e136]:
      - checkbox "Enable for next and successive requests" [ref=e137]
      - generic [ref=e138]: Przechwycone przekierowania
    - listitem [ref=e139]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e140]
      - link "Profilowanie" [ref=e141] [cursor=pointer]:
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