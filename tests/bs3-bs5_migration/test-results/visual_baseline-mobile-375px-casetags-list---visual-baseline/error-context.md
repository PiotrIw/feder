# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: visual_baseline.spec.ts >> mobile (375px) >> casetags-list - visual baseline
- Location: tests/bs3-bs5_migration/visual_baseline.spec.ts:10:11

# Error details

```
Error: expect(page).toHaveScreenshot(expected) failed

  Expected an image 686px by 1967px, received 684px by 1699px. 262645 pixels (ratio 0.20 of all image pixels) are different.

  Snapshot: casetags-list-mobile.png

Call log:
  - Expect "toHaveScreenshot(casetags-list-mobile.png)" with timeout 5000ms
    - verifying given screenshot expectation
  - taking page screenshot
    - disabled all CSS animations
  - waiting for fonts to load...
  - fonts loaded
  - Expected an image 686px by 1967px, received 684px by 1699px. 262645 pixels (ratio 0.20 of all image pixels) are different.
  - waiting 100ms before taking screenshot
  - taking page screenshot
    - disabled all CSS animations
  - waiting for fonts to load...
  - fonts loaded
  - captured a stable screenshot
  - Expected an image 686px by 1967px, received 684px by 1699px. 262645 pixels (ratio 0.20 of all image pixels) are different.

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
      - link "Utwórz" [ref=e76] [cursor=pointer]:
        - /url: /sprawy/tagi/monitoring-44/~create
      - heading "Wykaz tagów" [level=2] [ref=e78]
      - generic [ref=e79]:
        - generic [ref=e82]:
          - generic [ref=e83]:
            - generic [ref=e84]: Nazwa zawiera
            - textbox "Nazwa zawiera" [ref=e85]
          - button "Filtruj" [ref=e86] [cursor=pointer]
        - generic [ref=e88]:
          - heading [level=3] [ref=e90]:
            - link "test" [ref=e92] [cursor=pointer]:
              - /url: None
          - heading [level=3] [ref=e94]:
            - link "procedury z wolnej ręki i dotacji" [ref=e96] [cursor=pointer]:
              - /url: /sprawy/tagi/monitoring-44/2
          - heading [level=3] [ref=e98]:
            - link "bip_brak_lub_nie_wiadomo" [ref=e100] [cursor=pointer]:
              - /url: /sprawy/tagi/monitoring-44/23
          - heading [level=3] [ref=e102]:
            - link "rejestr_umów_brak_lub_nie_wiadomo" [ref=e104] [cursor=pointer]:
              - /url: /sprawy/tagi/monitoring-44/24
          - heading [level=3] [ref=e106]:
            - link "rejestr_umów_jest_ale_nie_publikowany_lub_nie_wiadomo" [ref=e108] [cursor=pointer]:
              - /url: /sprawy/tagi/monitoring-44/25
          - heading [level=3] [ref=e110]:
            - link "procedury_przyznawania_dotacji_brak_lub_nie_wiadomo" [ref=e112] [cursor=pointer]:
              - /url: /sprawy/tagi/monitoring-44/26
          - list [ref=e113]:
            - listitem [ref=e114]:
              - generic [aria-hidden]: ←
            - listitem [ref=e115]:
              - generic "Current Page" [ref=e116]: "1"
            - listitem [ref=e117]:
              - generic [aria-hidden]: →
      - generic [ref=e118]:
        - generic [ref=e119]:
          - text: Sieć Obywatelska - Watchdog Polska ul. Szpitalna 5/5 00-031 Warszawa
          - paragraph [ref=e120]:
            - link "Klauzula RODO" [ref=e121] [cursor=pointer]:
              - /url: https://fedrowanie.siecobywatelska.pl/media_internal/tinycontent/uploads/11_KLAUZULA.-.dane.z.udip.na.stronach.www.i.w.mediach.spol.pdf
        - generic [ref=e122]:
          - text: "silnik:"
          - link "jawne.info.pl" [ref=e123] [cursor=pointer]:
            - /url: http://jawne.info.pl
          - text: "| v1.5.77.deps |"
          - link "GitHub" [ref=e124] [cursor=pointer]:
            - /url: https://github.com/watchdogpolska/feder
          - text: –
          - link "efd8a3b" [ref=e126] [cursor=pointer]:
            - /url: https://github.com/watchdogpolska/feder/compare/efd8a3b2...master
          - text: "|"
          - link "API" [ref=e127] [cursor=pointer]:
            - /url: /api/
        - generic [ref=e129]: Ta strona wykorzystuje cookies.
  - list [ref=e131]:
    - listitem [ref=e132]:
      - link "Ukryj »" [ref=e133] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e134]:
      - link "Toggle Theme" [ref=e135] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e138]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e139]
      - link "Historia /sprawy/tagi/monitoring-44" [ref=e140] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e141]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e142]
      - link "Wersje Django 5.2.17" [ref=e143] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e144]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e145]
      - 'link "Czas CPU: 105.86ms (121.46ms)" [ref=e146] [cursor=pointer]':
        - /url: "#"
    - listitem [ref=e147]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e148]
      - link "Ustawienia" [ref=e149] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e150]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e151]
      - link "Nagłówki" [ref=e152] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e153]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e154]
      - link "Zapytania TagListView" [ref=e155] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e156]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e157]
      - link "SQL 8 queries in 15.74ms" [ref=e158] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e159]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e160]
      - link "Pliki statyczne 3 użyte plików" [ref=e161] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e162]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e163]
      - link "Templatki cases_tags/tag_filter.html" [ref=e164] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e165]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e166]
      - link "Alerty" [ref=e167] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e168]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e169]
      - link "Cache 2 wywołania w 0.14ms" [ref=e170] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e171]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e172]
      - link "Sygnały 88 odbiorców 15 sygnałów" [ref=e173] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e174]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e175]
      - link "Gmina" [ref=e176] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e177]:
      - checkbox "Enable for next and successive requests" [ref=e178]
      - generic [ref=e179]: Przechwycone przekierowania
    - listitem [ref=e180]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e181]
      - link "Profilowanie" [ref=e182] [cursor=pointer]:
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