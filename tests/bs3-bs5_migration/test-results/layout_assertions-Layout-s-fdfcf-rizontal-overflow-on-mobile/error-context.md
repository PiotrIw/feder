# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: layout_assertions.spec.ts >> Layout sanity - mobile >> monitorings-answers-categories - no horizontal overflow on mobile
- Location: tests/bs3-bs5_migration/layout_assertions.spec.ts:30:9

# Error details

```
Error: Mobile layout has horizontal scroll

expect(received).toBeLessThanOrEqual(expected)

Expected: <= 20
Received:    126
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
        - listitem [ref=e70]: Monitoring sądów apelacyjnych
      - generic [ref=e72]:
        - link "Edytuj" [ref=e73] [cursor=pointer]:
          - /url: /monitoringi/monitoring-sadow-apelacyjnych/~edytuj
        - link "Aktualizuj wyniki" [ref=e74] [cursor=pointer]:
          - /url: /monitoringi/monitoring-sadow-apelacyjnych/~results-update
        - link "Przypisz" [ref=e75] [cursor=pointer]:
          - /url: /monitoringi/monitoring-sadow-apelacyjnych/~przypisz
        - link "Usuń" [ref=e76] [cursor=pointer]:
          - /url: /monitoringi/monitoring-sadow-apelacyjnych/~usun
        - link "Utwórz sprawę" [ref=e77] [cursor=pointer]:
          - /url: /sprawy/~utworz-5
        - link "Wiadomość masowa" [ref=e78] [cursor=pointer]:
          - /url: /monitoringi/monitoring-sadow-apelacyjnych/~wiadomosc-masowa
        - link "Uprawnienia" [ref=e79] [cursor=pointer]:
          - /url: /monitoringi/monitoring-sadow-apelacyjnych/~uprawnienia
        - link "Lista alertów" [ref=e80] [cursor=pointer]:
          - /url: /alerty/monitoring-5
        - link "Zobacz dzienniki" [ref=e81] [cursor=pointer]:
          - /url: /listy/logi/monitoring-5
        - link "Zobacz tagi" [ref=e83] [cursor=pointer]:
          - /url: /sprawy/tagi/monitoring-5
        - link "Zobacz raport" [ref=e85] [cursor=pointer]:
          - /url: /monitoringi/monitoring-sadow-apelacyjnych/raport
        - link "Zobacz tabelę spraw" [ref=e87] [cursor=pointer]:
          - /url: /monitoringi/monitoring-sadow-apelacyjnych/monitoring_cases_table
      - heading [level=2] [ref=e90]:
        - text: Monitoring sądów apelacyjnych
        - generic [ref=e92]:
          - text: przez
          - link "adobrawy" [ref=e93] [cursor=pointer]:
            - /url: /uzytkownik/adobrawy/
          - time [ref=e94]: 11 sierpnia 2017 02:47
      - generic [ref=e95]:
        - table [ref=e98]:
          - rowgroup [ref=e99]:
            - row [ref=e100]:
              - columnheader "Województwo" [ref=e101]
              - columnheader "Liczba spraw" [ref=e102]
              - columnheader "Liczba spraw z potw. odbioru" [ref=e103]
              - columnheader "Liczba spraw z odpowiedzią" [ref=e104]
            - row [ref=e105]:
              - cell "Dolnośląskie" [ref=e106]
              - cell "1" [ref=e107]
              - cell "0" [ref=e108]
              - cell "1" [ref=e109]
            - row [ref=e110]:
              - cell "Kujawsko-Pomorskie" [ref=e111]
              - cell "0" [ref=e112]
              - cell "0" [ref=e113]
              - cell "0" [ref=e114]
            - row [ref=e115]:
              - cell "Lubelskie" [ref=e116]
              - cell "1" [ref=e117]
              - cell "0" [ref=e118]
              - cell "1" [ref=e119]
            - row [ref=e120]:
              - cell "Lubuskie" [ref=e121]
              - cell "0" [ref=e122]
              - cell "0" [ref=e123]
              - cell "0" [ref=e124]
            - row [ref=e125]:
              - cell "Łódzkie" [ref=e126]
              - cell "1" [ref=e127]
              - cell "0" [ref=e128]
              - cell "1" [ref=e129]
            - row [ref=e130]:
              - cell "Małopolskie" [ref=e131]
              - cell "1" [ref=e132]
              - cell "0" [ref=e133]
              - cell "1" [ref=e134]
            - row [ref=e135]:
              - cell "Mazowieckie" [ref=e136]
              - cell "1" [ref=e137]
              - cell "0" [ref=e138]
              - cell "1" [ref=e139]
            - row [ref=e140]:
              - cell "Opolskie" [ref=e141]
              - cell "0" [ref=e142]
              - cell "0" [ref=e143]
              - cell "0" [ref=e144]
            - row [ref=e145]:
              - cell "Podkarpackie" [ref=e146]
              - cell "1" [ref=e147]
              - cell "0" [ref=e148]
              - cell "1" [ref=e149]
            - row [ref=e150]:
              - cell "Podlaskie" [ref=e151]
              - cell "1" [ref=e152]
              - cell "0" [ref=e153]
              - cell "1" [ref=e154]
            - row [ref=e155]:
              - cell "Pomorskie" [ref=e156]
              - cell "1" [ref=e157]
              - cell "0" [ref=e158]
              - cell "1" [ref=e159]
            - row [ref=e160]:
              - cell "Śląskie" [ref=e161]
              - cell "1" [ref=e162]
              - cell "0" [ref=e163]
              - cell "1" [ref=e164]
            - row [ref=e165]:
              - cell "Świętokrzyskie" [ref=e166]
              - cell "0" [ref=e167]
              - cell "0" [ref=e168]
              - cell "0" [ref=e169]
            - row [ref=e170]:
              - cell "Warmińsko-Mazurskie" [ref=e171]
              - cell "0" [ref=e172]
              - cell "0" [ref=e173]
              - cell "0" [ref=e174]
            - row [ref=e175]:
              - cell "Wielkopolskie" [ref=e176]
              - cell "1" [ref=e177]
              - cell "0" [ref=e178]
              - cell "1" [ref=e179]
            - row [ref=e180]:
              - cell "Zachodniopomorskie" [ref=e181]
              - cell "1" [ref=e182]
              - cell "0" [ref=e183]
              - cell "1" [ref=e184]
            - row [ref=e185]:
              - cell "Wszystkie" [ref=e186]
              - cell "11" [ref=e187]
              - cell "0" [ref=e188]
              - cell "11" [ref=e189]
        - generic [ref=e190]:
          - list [ref=e191]:
            - listitem [ref=e192]:
              - link "Instytucje i sprawy" [ref=e193] [cursor=pointer]:
                - /url: /monitoringi/monitoring-sadow-apelacyjnych
            - listitem [ref=e194]:
              - link "Listy" [ref=e195] [cursor=pointer]:
                - /url: /monitoringi/monitoring-sadow-apelacyjnych/listy
            - listitem [ref=e196]:
              - link "Projekty" [ref=e197] [cursor=pointer]:
                - /url: /monitoringi/monitoring-sadow-apelacyjnych/projekty
            - listitem [ref=e198]:
              - link "Szablon" [ref=e199] [cursor=pointer]:
                - /url: /monitoringi/monitoring-sadow-apelacyjnych/template
            - listitem [ref=e200]:
              - link "Wyniki" [ref=e201] [cursor=pointer]:
                - /url: /monitoringi/monitoring-sadow-apelacyjnych/results
          - heading "Kategorie odpowiedzi" [level=3] [ref=e202]
          - generic [ref=e203]:
            - paragraph [ref=e204]: "Uwaga: Proszę sporządzić przejrzystą listę co najmniej 2 kategorii odpowiedzi, używając wypunktowań literowych z \")\". Możesz dodać dodatkową instrukcję dla LLM po liście i linii separatora zawierającej ```, jak w poniższym przykładzie: __________________________________________________________ a) Tak b) Nie c) Nie jestem pewien ``` W przypadku jakichkolwiek wątpliwości użyj \"c) Nie jestem pewien” jako odpowiedzi."
            - button "Wygeneruj zadania kategoryzacji odpowiedzi w sprawach" [ref=e207] [cursor=pointer]
      - generic [ref=e208]:
        - generic [ref=e209]:
          - text: Sieć Obywatelska - Watchdog Polska ul. Szpitalna 5/5 00-031 Warszawa
          - paragraph [ref=e210]:
            - link "Klauzula RODO" [ref=e211] [cursor=pointer]:
              - /url: https://fedrowanie.siecobywatelska.pl/media_internal/tinycontent/uploads/11_KLAUZULA.-.dane.z.udip.na.stronach.www.i.w.mediach.spol.pdf
        - generic [ref=e212]:
          - text: "silnik:"
          - link "jawne.info.pl" [ref=e213] [cursor=pointer]:
            - /url: http://jawne.info.pl
          - text: "| v1.5.77.deps |"
          - link "GitHub" [ref=e214] [cursor=pointer]:
            - /url: https://github.com/watchdogpolska/feder
          - text: –
          - link "efd8a3b" [ref=e216] [cursor=pointer]:
            - /url: https://github.com/watchdogpolska/feder/compare/efd8a3b2...master
          - text: "|"
          - link "API" [ref=e217] [cursor=pointer]:
            - /url: /api/
        - generic [ref=e219]: Ta strona wykorzystuje cookies.
  - list [ref=e221]:
    - listitem [ref=e222]:
      - link "Ukryj »" [ref=e223] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e224]:
      - link "Toggle Theme" [ref=e225] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e228]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e229]
      - link "Historia /monitoringi/monitoring-sadow-apelacyjnych/answers-categories" [ref=e230] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e231]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e232]
      - link "Wersje Django 5.2.17" [ref=e233] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e234]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e235]
      - 'link "Czas CPU: 217.82ms (247.49ms)" [ref=e236] [cursor=pointer]':
        - /url: "#"
    - listitem [ref=e237]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e238]
      - link "Ustawienia" [ref=e239] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e240]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e241]
      - link "Nagłówki" [ref=e242] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e243]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e244]
      - link "Zapytania MonitoringAnswersCategoriesView" [ref=e245] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e246]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e247]
      - link "SQL 62 queries in 30.03ms" [ref=e248] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e249]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e250]
      - link "Pliki statyczne 3 użyte plików" [ref=e251] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e252]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e253]
      - link "Templatki monitorings/monitoring_answers_categories.html" [ref=e254] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e255]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e256]
      - link "Alerty" [ref=e257] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e258]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e259]
      - link "Cache 2 wywołania w 0.13ms" [ref=e260] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e261]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e262]
      - link "Sygnały 88 odbiorców 15 sygnałów" [ref=e263] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e264]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e265]
      - link "Gmina" [ref=e266] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e267]:
      - checkbox "Enable for next and successive requests" [ref=e268]
      - generic [ref=e269]: Przechwycone przekierowania
    - listitem [ref=e270]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e271]
      - link "Profilowanie" [ref=e272] [cursor=pointer]:
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