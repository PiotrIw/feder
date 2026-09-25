# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: visual_baseline.spec.ts >> desktop (2560px) >> casetags-list - visual baseline
- Location: tests/bs3-bs5_migration/visual_baseline.spec.ts:10:11

# Error details

```
Error: expect(page).toHaveScreenshot(expected) failed

  Expected an image 2560px by 1442px, received 2560px by 1440px. 438741 pixels (ratio 0.12 of all image pixels) are different.

  Snapshot: casetags-list-desktop.png

Call log:
  - Expect "toHaveScreenshot(casetags-list-desktop.png)" with timeout 5000ms
    - verifying given screenshot expectation
  - taking page screenshot
    - disabled all CSS animations
  - waiting for fonts to load...
  - fonts loaded
  - Expected an image 2560px by 1442px, received 2560px by 1440px. 438741 pixels (ratio 0.12 of all image pixels) are different.
  - waiting 100ms before taking screenshot
  - taking page screenshot
    - disabled all CSS animations
  - waiting for fonts to load...
  - fonts loaded
  - captured a stable screenshot
  - Expected an image 2560px by 1442px, received 2560px by 1440px. 438741 pixels (ratio 0.12 of all image pixels) are different.

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
        - listitem [ref=e64]:
          - text: /
          - link "Wykaz tagów" [ref=e65] [cursor=pointer]:
            - /url: /sprawy/tagi/monitoring-44
      - link "Utwórz" [ref=e68] [cursor=pointer]:
        - /url: /sprawy/tagi/monitoring-44/~create
      - heading "Wykaz tagów" [level=2] [ref=e70]
      - generic [ref=e71]:
        - generic [ref=e74]:
          - generic [ref=e75]:
            - generic [ref=e76]: Nazwa zawiera
            - textbox "Nazwa zawiera" [ref=e77]
          - button "Filtruj" [ref=e78] [cursor=pointer]
        - generic [ref=e80]:
          - heading [level=3] [ref=e82]:
            - link "test" [ref=e84] [cursor=pointer]:
              - /url: None
          - heading [level=3] [ref=e86]:
            - link "procedury z wolnej ręki i dotacji" [ref=e88] [cursor=pointer]:
              - /url: /sprawy/tagi/monitoring-44/2
          - heading [level=3] [ref=e90]:
            - link "bip_brak_lub_nie_wiadomo" [ref=e92] [cursor=pointer]:
              - /url: /sprawy/tagi/monitoring-44/23
          - heading [level=3] [ref=e94]:
            - link "rejestr_umów_brak_lub_nie_wiadomo" [ref=e96] [cursor=pointer]:
              - /url: /sprawy/tagi/monitoring-44/24
          - heading [level=3] [ref=e98]:
            - link "rejestr_umów_jest_ale_nie_publikowany_lub_nie_wiadomo" [ref=e100] [cursor=pointer]:
              - /url: /sprawy/tagi/monitoring-44/25
          - heading [level=3] [ref=e102]:
            - link "procedury_przyznawania_dotacji_brak_lub_nie_wiadomo" [ref=e104] [cursor=pointer]:
              - /url: /sprawy/tagi/monitoring-44/26
          - list [ref=e105]:
            - listitem [ref=e106]:
              - generic [aria-hidden]: ←
            - listitem [ref=e107]:
              - generic "Current Page" [ref=e108]: "1"
            - listitem [ref=e109]:
              - generic [aria-hidden]: →
      - generic [ref=e110]:
        - generic [ref=e111]:
          - text: Sieć Obywatelska - Watchdog Polska ul. Szpitalna 5/5 00-031 Warszawa
          - paragraph [ref=e112]:
            - link "Klauzula RODO" [ref=e113] [cursor=pointer]:
              - /url: https://fedrowanie.siecobywatelska.pl/media_internal/tinycontent/uploads/11_KLAUZULA.-.dane.z.udip.na.stronach.www.i.w.mediach.spol.pdf
        - generic [ref=e114]:
          - text: "silnik:"
          - link "jawne.info.pl" [ref=e115] [cursor=pointer]:
            - /url: http://jawne.info.pl
          - text: "| v1.5.77.deps |"
          - link "GitHub" [ref=e116] [cursor=pointer]:
            - /url: https://github.com/watchdogpolska/feder
          - text: –
          - link "efd8a3b" [ref=e118] [cursor=pointer]:
            - /url: https://github.com/watchdogpolska/feder/compare/efd8a3b2...master
          - text: "|"
          - link "API" [ref=e119] [cursor=pointer]:
            - /url: /api/
        - generic [ref=e121]: Ta strona wykorzystuje cookies.
  - list [ref=e123]:
    - listitem [ref=e124]:
      - link "Ukryj »" [ref=e125] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e126]:
      - link "Toggle Theme" [ref=e127] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e130]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e131]
      - link "Historia /sprawy/tagi/monitoring-44" [ref=e132] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e133]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e134]
      - link "Wersje Django 5.2.17" [ref=e135] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e136]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e137]
      - 'link "Czas CPU: 268.65ms (287.08ms)" [ref=e138] [cursor=pointer]':
        - /url: "#"
    - listitem [ref=e139]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e140]
      - link "Ustawienia" [ref=e141] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e142]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e143]
      - link "Nagłówki" [ref=e144] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e145]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e146]
      - link "Zapytania TagListView" [ref=e147] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e148]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e149]
      - link "SQL 8 queries in 19.17ms" [ref=e150] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e151]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e152]
      - link "Pliki statyczne 3 użyte plików" [ref=e153] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e154]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e155]
      - link "Templatki cases_tags/tag_filter.html" [ref=e156] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e157]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e158]
      - link "Alerty" [ref=e159] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e160]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e161]
      - link "Cache 2 wywołania w 0.13ms" [ref=e162] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e163]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e164]
      - link "Sygnały 88 odbiorców 15 sygnałów" [ref=e165] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e166]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e167]
      - link "Gmina" [ref=e168] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e169]:
      - checkbox "Enable for next and successive requests" [ref=e170]
      - generic [ref=e171]: Przechwycone przekierowania
    - listitem [ref=e172]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e173]
      - link "Profilowanie" [ref=e174] [cursor=pointer]:
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