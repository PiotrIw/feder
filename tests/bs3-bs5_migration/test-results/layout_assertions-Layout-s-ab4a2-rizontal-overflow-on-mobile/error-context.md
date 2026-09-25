# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: layout_assertions.spec.ts >> Layout sanity - mobile >> home - no horizontal overflow on mobile
- Location: tests/bs3-bs5_migration/layout_assertions.spec.ts:30:9

# Error details

```
Error: Mobile layout has horizontal scroll

expect(received).toBeLessThanOrEqual(expected)

Expected: <= 20
Received:    40
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
      - generic [ref=e70]:
        - generic [ref=e71]:
          - heading "Obywatelsko fedrujmy dane!" [level=1] [ref=e72]
          - heading "Razem przenieśmy debatę publiczną do nowej epoki!" [level=4] [ref=e73]
          - paragraph [ref=e74]:
            - button "Czytaj więcej" [ref=e75] [cursor=pointer]
        - generic [ref=e76]:
          - generic [ref=e78]:
            - heading "Województwa" [level=2] [ref=e79]
            - img [ref=e81]:
              - generic "Podkarpackie" [ref=e83] [cursor=pointer]
              - generic "Małopolskie" [ref=e85] [cursor=pointer]
              - generic "Śląskie" [ref=e87] [cursor=pointer]
              - generic "Opolskie" [ref=e89] [cursor=pointer]
              - generic "Dolnośląskie" [ref=e91] [cursor=pointer]
              - generic "Świętokrzyskie" [ref=e93] [cursor=pointer]
              - generic "Lubelskie" [ref=e95] [cursor=pointer]
              - generic "Łódzkie" [ref=e97] [cursor=pointer]
              - generic "Mazowieckie" [ref=e99] [cursor=pointer]
              - generic "Wielkopolska" [ref=e101] [cursor=pointer]
              - generic "Lubuskie" [ref=e103] [cursor=pointer]
              - generic "Kujawsko-pomorskie" [ref=e105] [cursor=pointer]
              - generic "Podlaskie" [ref=e107] [cursor=pointer]
              - generic "Zachodniopomorskie" [ref=e109] [cursor=pointer]
              - generic "Warmińsko-mazurskie" [ref=e111] [cursor=pointer]
              - generic "Pomorskie" [ref=e113] [cursor=pointer]
          - table [ref=e116]:
            - rowgroup [ref=e117]:
              - row [ref=e118]:
                - columnheader "Województwo" [ref=e119]
                - columnheader "Liczba instytucji" [ref=e120]
                - columnheader "Liczba spraw" [ref=e121]
              - row [ref=e122]:
                - cell "Dolnośląskie" [ref=e123]
                - cell "2626" [ref=e124]
                - cell "5095" [ref=e125]
              - row [ref=e126]:
                - cell "Kujawsko-Pomorskie" [ref=e127]
                - cell "2003" [ref=e128]
                - cell "4153" [ref=e129]
              - row [ref=e130]:
                - cell "Lubelskie" [ref=e131]
                - cell "2484" [ref=e132]
                - cell "5571" [ref=e133]
              - row [ref=e134]:
                - cell "Lubuskie" [ref=e135]
                - cell "1122" [ref=e136]
                - cell "2434" [ref=e137]
              - row [ref=e138]:
                - cell "Łódzkie" [ref=e139]
                - cell "2431" [ref=e140]
                - cell "4813" [ref=e141]
              - row [ref=e142]:
                - cell "Małopolskie" [ref=e143]
                - cell "3517" [ref=e144]
                - cell "4764" [ref=e145]
              - row [ref=e146]:
                - cell "Mazowieckie" [ref=e147]
                - cell "5147" [ref=e148]
                - cell "8796" [ref=e149]
              - row [ref=e150]:
                - cell "Opolskie" [ref=e151]
                - cell "1146" [ref=e152]
                - cell "1894" [ref=e153]
              - row [ref=e154]:
                - cell "Podkarpackie" [ref=e155]
                - cell "2617" [ref=e156]
                - cell "4150" [ref=e157]
              - row [ref=e158]:
                - cell "Podlaskie" [ref=e159]
                - cell "1274" [ref=e160]
                - cell "3005" [ref=e161]
              - row [ref=e162]:
                - cell "Pomorskie" [ref=e163]
                - cell "2099" [ref=e164]
                - cell "3413" [ref=e165]
              - row [ref=e166]:
                - cell "Śląskie" [ref=e167]
                - cell "3933" [ref=e168]
                - cell "4940" [ref=e169]
              - row [ref=e170]:
                - cell "Świętokrzyskie" [ref=e171]
                - cell "1418" [ref=e172]
                - cell "2618" [ref=e173]
              - row [ref=e174]:
                - cell "Warmińsko-Mazurskie" [ref=e175]
                - cell "1700" [ref=e176]
                - cell "3250" [ref=e177]
              - row [ref=e178]:
                - cell "Wielkopolskie" [ref=e179]
                - cell "3544" [ref=e180]
                - cell "5950" [ref=e181]
              - row [ref=e182]:
                - cell "Zachodniopomorskie" [ref=e183]
                - cell "1627" [ref=e184]
                - cell "3226" [ref=e185]
              - row [ref=e186]:
                - cell "Wszystkie" [ref=e187]
                - cell "38689" [ref=e188]
                - cell "68075" [ref=e189]
          - generic [ref=e191]:
            - heading "Ostatnie monitoringi" [level=2] [ref=e192]
            - list [ref=e194]:
              - listitem [ref=e195]:
                - link "Kontrole punktów gastronomicznych" [ref=e196] [cursor=pointer]:
                  - /url: /monitoringi/kontrole-punktow-gastronomicznych
              - listitem [ref=e197]:
                - link "Zespoły w ministerstwach" [ref=e198] [cursor=pointer]:
                  - /url: /monitoringi/zespoly-w-ministerstwach
              - listitem [ref=e199]:
                - link "Ministerstwa - baza danych umów cywilnoprawnych" [ref=e200] [cursor=pointer]:
                  - /url: /monitoringi/ministerstwa-baza-danych-umow-cywilnoprawnych
              - listitem [ref=e201]:
                - link "Nagrody w ministerstwach 2024" [ref=e202] [cursor=pointer]:
                  - /url: /monitoringi/nagrody-w-ministerstwach-2
              - listitem [ref=e203]:
                - link "Lasy Państwowe a SLAPPy" [ref=e204] [cursor=pointer]:
                  - /url: /monitoringi/lasy-panstwowe-a-slappy
              - listitem [ref=e205]:
                - link "Wnioski o informację o środowisku w 2023 - RDOŚ-ie" [ref=e206] [cursor=pointer]:
                  - /url: /monitoringi/wnioski-o-informacje-o-srodowisku-w-2023-rdos-ie-gdos-parki-narodowe
              - listitem [ref=e207]:
                - link "Wnioski o informację o środowisku w 2023 - gminy, starostwa, urzędy marszałkowskie, GDOŚ, parki" [ref=e208] [cursor=pointer]:
                  - /url: /monitoringi/wnioski-o-informacje-o-srodowisku-w-2023-gminy-starostwa-urzedy-marszalkowskie
              - listitem [ref=e209]:
                - link "Wnioski o informację w 2023 - Komendanci Wojewódzcy i Stołeczny Policji" [ref=e210] [cursor=pointer]:
                  - /url: /monitoringi/wnioski-o-informacje-w-2023-komendanci-wojewodzcy-i-stoleczny-policji
              - listitem [ref=e211]:
                - link "Wnioski o informację w 2023 - pytamy wojewodów o straże gminne (miejskie)" [ref=e212] [cursor=pointer]:
                  - /url: /monitoringi/wnioski-o-informacje-w-2023-pytamy-wojewodow-o-straze-gminne-miejskie
              - listitem [ref=e213]:
                - link "Doradcy Marszałków Sejmu i Senatu" [ref=e214] [cursor=pointer]:
                  - /url: /monitoringi/doradcy-marszalkow-sejmu-i-senatu
              - listitem [ref=e215]:
                - link "Karty płatnicze w ministerstwach" [ref=e216] [cursor=pointer]:
                  - /url: /monitoringi/karty-platnicze-w-ministerstwach
              - listitem [ref=e217]:
                - link "Wnioski o informację w 2023 - samorządowe kolegia odwoławcze" [ref=e218] [cursor=pointer]:
                  - /url: /monitoringi/wnioski-o-informacje-w-2023-samorzadowe-kolegia-odwolawcze
              - listitem [ref=e219]:
                - link "Wnioski o informację w 2023 - Akademia Wymiaru Sprawiedliwości" [ref=e220] [cursor=pointer]:
                  - /url: /monitoringi/wnioski-o-informacje-w-2023-akademia-wymiaru-sprawiedliwosci
              - listitem [ref=e221]:
                - link "Wnioski o informację w 2023 - Komendanci Ośrodków Szkolenia Służby Więziennej" [ref=e222] [cursor=pointer]:
                  - /url: /monitoringi/wnioski-o-informacje-w-2023-komendanci-osrodkow-szkolenia-sluzby-wieziennej
              - listitem [ref=e223]:
                - link "Wnioski o informację w 2023 - Dyrektorzy Okręgowi Służby Więziennej" [ref=e224] [cursor=pointer]:
                  - /url: /monitoringi/wnioski-o-informacje-w-2023-dyrektorzy-okregowi-sluzby-wieziennej
              - listitem [ref=e225]:
                - link "Wnioski o informację w 2023 - Dyrektor Generalny Służby Więziennej" [ref=e226] [cursor=pointer]:
                  - /url: /monitoringi/wnioski-o-informacje-w-2023-dyrektor-generalny-sluzby-wieziennej
      - generic [ref=e227]:
        - generic [ref=e228]:
          - text: Sieć Obywatelska - Watchdog Polska ul. Szpitalna 5/5 00-031 Warszawa
          - paragraph [ref=e229]:
            - link "Klauzula RODO" [ref=e230] [cursor=pointer]:
              - /url: https://fedrowanie.siecobywatelska.pl/media_internal/tinycontent/uploads/11_KLAUZULA.-.dane.z.udip.na.stronach.www.i.w.mediach.spol.pdf
        - generic [ref=e231]:
          - text: "silnik:"
          - link "jawne.info.pl" [ref=e232] [cursor=pointer]:
            - /url: http://jawne.info.pl
          - text: "| v1.5.77.deps |"
          - link "GitHub" [ref=e233] [cursor=pointer]:
            - /url: https://github.com/watchdogpolska/feder
          - text: –
          - link "efd8a3b" [ref=e235] [cursor=pointer]:
            - /url: https://github.com/watchdogpolska/feder/compare/efd8a3b2...master
          - text: "|"
          - link "API" [ref=e236] [cursor=pointer]:
            - /url: /api/
        - generic [ref=e238]: Ta strona wykorzystuje cookies.
  - list [ref=e240]:
    - listitem [ref=e241]:
      - link "Ukryj »" [ref=e242] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e243]:
      - link "Toggle Theme" [ref=e244] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e247]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e248]
      - link "Historia /" [ref=e249] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e250]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e251]
      - link "Wersje Django 5.2.17" [ref=e252] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e253]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e254]
      - 'link "Czas CPU: 143.87ms (352.57ms)" [ref=e255] [cursor=pointer]':
        - /url: "#"
    - listitem [ref=e256]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e257]
      - link "Ustawienia" [ref=e258] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e259]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e260]
      - link "Nagłówki" [ref=e261] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e262]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e263]
      - link "Zapytania HomeView" [ref=e264] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e265]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e266]
      - link "SQL 40 queries in 215.15ms" [ref=e267] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e268]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e269]
      - link "Pliki statyczne 3 użyte plików" [ref=e270] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e271]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e272]
      - link "Templatki main/home.html" [ref=e273] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e274]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e275]
      - link "Alerty" [ref=e276] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e277]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e278]
      - link "Cache 2 wywołania w 0.16ms" [ref=e279] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e280]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e281]
      - link "Sygnały 88 odbiorców 15 sygnałów" [ref=e282] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e283]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e284]
      - link "Gmina" [ref=e285] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e286]:
      - checkbox "Enable for next and successive requests" [ref=e287]
      - generic [ref=e288]: Przechwycone przekierowania
    - listitem [ref=e289]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e290]
      - link "Profilowanie" [ref=e291] [cursor=pointer]:
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