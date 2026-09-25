# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: layout_assertions.spec.ts >> Layout sanity - mobile >> letters-unrecognized-list - no horizontal overflow on mobile
- Location: tests/bs3-bs5_migration/layout_assertions.spec.ts:30:9

# Error details

```
Error: Mobile layout has horizontal scroll

expect(received).toBeLessThanOrEqual(expected)

Expected: <= 20
Received:    128
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
      - heading "Listy nieprzypisane do spraw" [level=2] [ref=e70]
      - generic [ref=e71]:
        - generic [ref=e72]:
          - generic [ref=e74]:
            - generic [ref=e75]:
              - generic [ref=e76]: Tytuł
              - textbox "Tytuł" [ref=e77]
            - generic [ref=e78]:
              - generic [ref=e79]: Data utworzenia
              - combobox "Data utworzenia" [ref=e80]:
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
            - generic [ref=e81]:
              - generic [ref=e82]: Record case Instytucja
              - combobox [aria-hidden] [ref=e83]
              - combobox [ref=e86] [cursor=pointer]:
                - textbox
            - generic [ref=e87]:
              - generic [ref=e88]: Ma plik .eml?
              - combobox "Ma plik .eml?" [ref=e89]:
                - option "Nieznany" [selected]
                - option "Tak"
                - option "Nie"
            - button "Filtruj" [ref=e90] [cursor=pointer]
          - table [ref=e93]:
            - rowgroup [ref=e94]:
              - row [ref=e95]:
                - columnheader [ref=e96]
                - columnheader "Sprawa nierozpoznana" [ref=e97]
                - columnheader "Sprawa rozpoznane" [ref=e98]
            - rowgroup [ref=e99]:
              - row [ref=e100]:
                - columnheader "Okres" [ref=e101]
                - columnheader "Liczba listów" [ref=e102]
                - columnheader "Nie spam" [ref=e103]
                - columnheader "Spam" [ref=e104]
                - columnheader "Spam" [ref=e105]
                - columnheader "Nie spam" [ref=e106]
            - rowgroup [ref=e107]:
              - row [ref=e108]:
                - cell "Dzisiaj" [ref=e109]
                - cell "0" [ref=e110]
                - cell "0" [ref=e111]
                - cell "0" [ref=e112]
                - cell "0" [ref=e113]
                - cell "0" [ref=e114]
              - row [ref=e115]:
                - cell "Wczoraj" [ref=e116]
                - cell "0" [ref=e117]
                - cell "0" [ref=e118]
                - cell "0" [ref=e119]
                - cell "0" [ref=e120]
                - cell "0" [ref=e121]
              - row [ref=e122]:
                - cell "Ostatnie 7 dni" [ref=e123]
                - cell "0" [ref=e124]
                - cell "0" [ref=e125]
                - cell "0" [ref=e126]
                - cell "0" [ref=e127]
                - cell "0" [ref=e128]
              - row [ref=e129]:
                - cell "Ten miesiąc" [ref=e130]
                - cell "0" [ref=e131]
                - cell "0" [ref=e132]
                - cell "0" [ref=e133]
                - cell "0" [ref=e134]
                - cell "0" [ref=e135]
              - row [ref=e136]:
                - cell "Ten rok" [ref=e137]
                - cell "0" [ref=e138]
                - cell "0" [ref=e139]
                - cell "0" [ref=e140]
                - cell "0" [ref=e141]
                - cell "0" [ref=e142]
              - row [ref=e143]:
                - cell "2025" [ref=e144]
                - cell "0" [ref=e145]
                - cell "0" [ref=e146]
                - cell "0" [ref=e147]
                - cell "0" [ref=e148]
                - cell "0" [ref=e149]
              - row [ref=e150]:
                - cell "2024" [ref=e151]
                - cell "46365" [ref=e152]
                - cell "3" [ref=e153]
                - cell "22803" [ref=e154]
                - cell "11035" [ref=e155]
                - cell "12524" [ref=e156]
              - row [ref=e157]:
                - cell "2023" [ref=e158]
                - cell "53486" [ref=e159]
                - cell "2" [ref=e160]
                - cell "25609" [ref=e161]
                - cell "3163" [ref=e162]
                - cell "24712" [ref=e163]
              - row [ref=e164]:
                - cell "2022" [ref=e165]
                - cell "18356" [ref=e166]
                - cell "4" [ref=e167]
                - cell "21" [ref=e168]
                - cell "264" [ref=e169]
                - cell "18067" [ref=e170]
              - row [ref=e171]:
                - cell "2021" [ref=e172]
                - cell "23961" [ref=e173]
                - cell "131" [ref=e174]
                - cell "0" [ref=e175]
                - cell "0" [ref=e176]
                - cell "23830" [ref=e177]
              - row [ref=e178]:
                - cell "2020" [ref=e179]
                - cell "34775" [ref=e180]
                - cell "358" [ref=e181]
                - cell "0" [ref=e182]
                - cell "0" [ref=e183]
                - cell "34417" [ref=e184]
              - row [ref=e185]:
                - cell "2019" [ref=e186]
                - cell "12784" [ref=e187]
                - cell "444" [ref=e188]
                - cell "0" [ref=e189]
                - cell "0" [ref=e190]
                - cell "12340" [ref=e191]
              - row [ref=e192]:
                - cell "2018" [ref=e193]
                - cell "14455" [ref=e194]
                - cell "6" [ref=e195]
                - cell "0" [ref=e196]
                - cell "0" [ref=e197]
                - cell "14449" [ref=e198]
              - row [ref=e199]:
                - cell "2017" [ref=e200]
                - cell "8703" [ref=e201]
                - cell "0" [ref=e202]
                - cell "0" [ref=e203]
                - cell "0" [ref=e204]
                - cell "8703" [ref=e205]
              - row [ref=e206]:
                - cell "2016" [ref=e207]
                - cell "95" [ref=e208]
                - cell "0" [ref=e209]
                - cell "0" [ref=e210]
                - cell "0" [ref=e211]
                - cell "95" [ref=e212]
        - generic [ref=e213]:
          - paragraph [ref=e215]: Brak wierszy.
          - list [ref=e216]:
            - listitem [ref=e217]:
              - generic [aria-hidden]: ←
            - listitem [ref=e218]:
              - generic "Current Page" [ref=e219]: "1"
            - listitem [ref=e220]:
              - generic [aria-hidden]: →
      - generic [ref=e221]:
        - generic [ref=e222]:
          - text: Sieć Obywatelska - Watchdog Polska ul. Szpitalna 5/5 00-031 Warszawa
          - paragraph [ref=e223]:
            - link "Klauzula RODO" [ref=e224] [cursor=pointer]:
              - /url: https://fedrowanie.siecobywatelska.pl/media_internal/tinycontent/uploads/11_KLAUZULA.-.dane.z.udip.na.stronach.www.i.w.mediach.spol.pdf
        - generic [ref=e225]:
          - text: "silnik:"
          - link "jawne.info.pl" [ref=e226] [cursor=pointer]:
            - /url: http://jawne.info.pl
          - text: "| v1.5.77.deps |"
          - link "GitHub" [ref=e227] [cursor=pointer]:
            - /url: https://github.com/watchdogpolska/feder
          - text: –
          - link "efd8a3b" [ref=e229] [cursor=pointer]:
            - /url: https://github.com/watchdogpolska/feder/compare/efd8a3b2...master
          - text: "|"
          - link "API" [ref=e230] [cursor=pointer]:
            - /url: /api/
        - generic [ref=e232]: Ta strona wykorzystuje cookies.
  - list [ref=e234]:
    - listitem [ref=e235]:
      - link "Ukryj »" [ref=e236] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e237]:
      - link "Toggle Theme" [ref=e238] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e241]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e242]
      - link "Historia /listy/przypisz" [ref=e243] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e244]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e245]
      - link "Wersje Django 5.2.17" [ref=e246] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e247]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e248]
      - 'link "Czas CPU: 780.81ms (1981.82ms)" [ref=e249] [cursor=pointer]':
        - /url: "#"
    - listitem [ref=e250]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e251]
      - link "Ustawienia" [ref=e252] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e253]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e254]
      - link "Nagłówki" [ref=e255] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e256]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e257]
      - link "Zapytania UnrecognizedLetterListView" [ref=e258] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e259]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e260]
      - link "SQL 20 queries in 1433.95ms" [ref=e261] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e262]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e263]
      - link "Pliki statyczne 10 użytych plików" [ref=e264] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e265]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e266]
      - link "Templatki letters/letter_unrecognized_list.html" [ref=e267] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e268]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e269]
      - link "Alerty" [ref=e270] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e271]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e272]
      - link "Cache 2 wywołania w 0.15ms" [ref=e273] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e274]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e275]
      - link "Sygnały 88 odbiorców 15 sygnałów" [ref=e276] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e277]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e278]
      - link "Gmina" [ref=e279] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e280]:
      - checkbox "Enable for next and successive requests" [ref=e281]
      - generic [ref=e282]: Przechwycone przekierowania
    - listitem [ref=e283]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e284]
      - link "Profilowanie" [ref=e285] [cursor=pointer]:
        - /url: "#"
```

# Test source

```ts
  1  | import { test, expect } from '@playwright/test';
  2  | import { PAGES } from './pages';
  3  | import { VIEWPORTS } from './viewports';
  4  | 
  5  | // Bootstrap 3's `.row` negative margins (-15px) routinely push scrollWidth ~4px past
  6  | // clientWidth on every page even with no visible scrollbar - that's cosmetic BS3 grid
  7  | // noise, not real overflow. A genuinely overflowing wide table measured ~278px over.
  8  | // This tolerance separates the two instead of flagging every single page.
  9  | const OVERFLOW_TOLERANCE_PX = 20;
  10 | 
  11 | test.describe('Layout sanity - desktop', () => {
  12 |   test.use({ viewport: VIEWPORTS.desktop });
  13 | 
  14 |   for (const page of PAGES) {
  15 |     test(`${page.name} - no horizontal overflow`, async ({ page: pw }) => {
  16 |       await pw.goto(page.path);
  17 |       await pw.waitForLoadState('networkidle');
  18 |       const overflowPx = await pw.evaluate(() =>
  19 |         document.documentElement.scrollWidth - document.documentElement.clientWidth
  20 |       );
  21 |       expect(overflowPx, 'Page has horizontal scroll').toBeLessThanOrEqual(OVERFLOW_TOLERANCE_PX);
  22 |     });
  23 |   }
  24 | });
  25 | 
  26 | test.describe('Layout sanity - mobile', () => {
  27 |   test.use({ viewport: VIEWPORTS.mobile });
  28 | 
  29 |   for (const page of PAGES) {
  30 |     test(`${page.name} - no horizontal overflow on mobile`, async ({ page: pw }) => {
  31 |       await pw.goto(page.path);
  32 |       await pw.waitForLoadState('networkidle');
  33 |       const overflowPx = await pw.evaluate(() =>
  34 |         document.documentElement.scrollWidth - document.documentElement.clientWidth
  35 |       );
> 36 |       expect(overflowPx, 'Mobile layout has horizontal scroll').toBeLessThanOrEqual(OVERFLOW_TOLERANCE_PX);
     |                                                                 ^ Error: Mobile layout has horizontal scroll
  37 |     });
  38 |   }
  39 | });
  40 | 
  41 | // This app's desktop layout (feder/main/templates/base.html) is a permanent left
  42 | // `.sidebar` next to `.content`, not a top navbar - `.navbar` is `display: none` above
  43 | // the mobile breakpoint (it only reappears, with `.navbar-toggle`, on small screens).
  44 | // So "nav above content" doesn't apply on desktop; the real desktop invariant is
  45 | // "sidebar sits to the left of content", checked below instead.
  46 | test.describe('Navigation structure', () => {
  47 |   test.use({ viewport: VIEWPORTS.desktop });
  48 | 
  49 |   test('sidebar is left of main content on desktop', async ({ page: pw }) => {
  50 |     await pw.goto('/');
  51 |     await pw.waitForLoadState('networkidle');
  52 |     const sidebarBox = await pw.locator('.sidebar').first().boundingBox();
  53 |     const contentBox = await pw.locator('.content').first().boundingBox();
  54 |     expect(sidebarBox).toBeTruthy();
  55 |     expect(contentBox).toBeTruthy();
  56 |     expect(sidebarBox!.x + sidebarBox!.width).toBeLessThanOrEqual(contentBox!.x + 5);
  57 |   });
  58 | 
  59 |   test('navbar collapses on mobile', async ({ page: pw }) => {
  60 |     await pw.setViewportSize(VIEWPORTS.mobile);
  61 |     await pw.goto('/');
  62 |     await pw.waitForLoadState('networkidle');
  63 |     const toggle = pw.locator('.navbar-toggle, .navbar-toggler');
  64 |     await expect(toggle).toBeVisible();
  65 |   });
  66 | });
  67 | 
```