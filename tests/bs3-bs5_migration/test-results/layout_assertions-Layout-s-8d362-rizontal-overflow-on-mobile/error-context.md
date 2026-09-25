# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: layout_assertions.spec.ts >> Layout sanity - mobile >> monitorings-perm - no horizontal overflow on mobile
- Location: tests/bs3-bs5_migration/layout_assertions.spec.ts:30:9

# Error details

```
Error: Mobile layout has horizontal scroll

expect(received).toBeLessThanOrEqual(expected)

Expected: <= 20
Received:    56
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
      - generic [ref=e73]:
        - link "Edytuj" [ref=e74] [cursor=pointer]:
          - /url: /monitoringi/monitoring-sadow-apelacyjnych/~edytuj
        - link "Aktualizuj wyniki" [ref=e75] [cursor=pointer]:
          - /url: /monitoringi/monitoring-sadow-apelacyjnych/~results-update
        - link "Przypisz" [ref=e76] [cursor=pointer]:
          - /url: /monitoringi/monitoring-sadow-apelacyjnych/~przypisz
        - link "Usuń" [ref=e77] [cursor=pointer]:
          - /url: /monitoringi/monitoring-sadow-apelacyjnych/~usun
        - link "Utwórz sprawę" [ref=e78] [cursor=pointer]:
          - /url: /sprawy/~utworz-5
        - link "Wiadomość masowa" [ref=e79] [cursor=pointer]:
          - /url: /monitoringi/monitoring-sadow-apelacyjnych/~wiadomosc-masowa
        - link "Uprawnienia" [ref=e80] [cursor=pointer]:
          - /url: /monitoringi/monitoring-sadow-apelacyjnych/~uprawnienia
        - link "Lista alertów" [ref=e81] [cursor=pointer]:
          - /url: /alerty/monitoring-5
        - link "Zobacz dzienniki" [ref=e82] [cursor=pointer]:
          - /url: /listy/logi/monitoring-5
        - link "Zobacz tagi" [ref=e84] [cursor=pointer]:
          - /url: /sprawy/tagi/monitoring-5
        - link "Zobacz raport" [ref=e86] [cursor=pointer]:
          - /url: /monitoringi/monitoring-sadow-apelacyjnych/raport
        - link "Zobacz tabelę spraw" [ref=e88] [cursor=pointer]:
          - /url: /monitoringi/monitoring-sadow-apelacyjnych/monitoring_cases_table
      - heading [level=2] [ref=e91]:
        - link "Monitoring sądów apelacyjnych" [ref=e93] [cursor=pointer]:
          - /url: /monitoringi/monitoring-sadow-apelacyjnych
      - table [ref=e95]:
        - rowgroup [ref=e96]:
          - row [ref=e97]:
            - cell [ref=e98]:
              - link "Dodaj użytkownika" [ref=e99] [cursor=pointer]:
                - /url: /monitoringi/monitoring-sadow-apelacyjnych/~uprawnienia/~dodaj
            - rowheader [ref=e100]:
              - link "Szymon_Osowski" [ref=e101] [cursor=pointer]:
                - /url: /monitoringi/monitoring-sadow-apelacyjnych/~uprawnienia-9
            - columnheader [ref=e102]:
              - link "adobrawy" [ref=e103] [cursor=pointer]:
                - /url: /monitoringi/monitoring-sadow-apelacyjnych/~uprawnienia-2
          - row [ref=e104]:
            - rowheader "Może tworzyć sprawę" [ref=e105]
            - cell [ref=e106]
            - cell [ref=e108]
          - row [ref=e110]:
            - rowheader "Dodaj szkic odpowiedzi" [ref=e111]
            - cell [ref=e112]
            - cell [ref=e114]
          - row [ref=e116]:
            - rowheader "Może dodawać list" [ref=e117]
            - cell [ref=e118]
            - cell [ref=e120]
          - row [ref=e122]:
            - rowheader "Can add questionary" [ref=e123]
            - cell [ref=e124]
            - cell [ref=e126]
          - row [ref=e128]:
            - rowheader "Can add task" [ref=e129]
            - cell [ref=e130]
            - cell [ref=e132]
          - row [ref=e134]:
            - rowheader "Może zmieniać alert" [ref=e135]
            - cell [ref=e136]
            - cell [ref=e138]
          - row [ref=e140]:
            - rowheader "Może zmieniać sprawę" [ref=e141]
            - cell [ref=e142]
            - cell [ref=e144]
          - row [ref=e146]:
            - rowheader "Może zmieniać monitoring" [ref=e147]
            - cell [ref=e148]
            - cell [ref=e150]
          - row [ref=e152]:
            - rowheader "Can change questionary" [ref=e153]
            - cell [ref=e154]
            - cell [ref=e156]
          - row [ref=e158]:
            - rowheader "Can change task" [ref=e159]
            - cell [ref=e160]
            - cell [ref=e162]
          - row [ref=e164]:
            - rowheader "Może usuwać alert" [ref=e165]
            - cell [ref=e166]
            - cell [ref=e168]
          - row [ref=e170]:
            - rowheader "Może usuwać sprawę" [ref=e171]
            - cell [ref=e172]
            - cell [ref=e174]
          - row [ref=e176]:
            - rowheader "Może usunąć monitoring" [ref=e177]
            - cell [ref=e178]
            - cell [ref=e180]
          - row [ref=e182]:
            - rowheader "Can delete questionary" [ref=e183]
            - cell [ref=e184]
            - cell [ref=e186]
          - row [ref=e188]:
            - rowheader "Can delete task" [ref=e189]
            - cell [ref=e190]
            - cell [ref=e192]
          - row [ref=e194]:
            - rowheader "Może zarządzać uprawnieniami" [ref=e195]
            - cell [ref=e196]
            - cell [ref=e198]
          - row [ref=e200]:
            - rowheader "Może odpowiadać" [ref=e201]
            - cell [ref=e202]
            - cell [ref=e204]
          - row [ref=e206]:
            - rowheader "Can select answer" [ref=e207]
            - cell [ref=e208]
            - cell [ref=e210]
          - row [ref=e212]:
            - rowheader "Może dodawać alert" [ref=e213]
            - cell [ref=e214]
            - cell [ref=e216]
      - generic [ref=e218]:
        - generic [ref=e219]:
          - text: Sieć Obywatelska - Watchdog Polska ul. Szpitalna 5/5 00-031 Warszawa
          - paragraph [ref=e220]:
            - link "Klauzula RODO" [ref=e221] [cursor=pointer]:
              - /url: https://fedrowanie.siecobywatelska.pl/media_internal/tinycontent/uploads/11_KLAUZULA.-.dane.z.udip.na.stronach.www.i.w.mediach.spol.pdf
        - generic [ref=e222]:
          - text: "silnik:"
          - link "jawne.info.pl" [ref=e223] [cursor=pointer]:
            - /url: http://jawne.info.pl
          - text: "| v1.5.77.deps |"
          - link "GitHub" [ref=e224] [cursor=pointer]:
            - /url: https://github.com/watchdogpolska/feder
          - text: –
          - link "efd8a3b" [ref=e226] [cursor=pointer]:
            - /url: https://github.com/watchdogpolska/feder/compare/efd8a3b2...master
          - text: "|"
          - link "API" [ref=e227] [cursor=pointer]:
            - /url: /api/
        - generic [ref=e229]: Ta strona wykorzystuje cookies.
  - list [ref=e231]:
    - listitem [ref=e232]:
      - link "Ukryj »" [ref=e233] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e234]:
      - link "Toggle Theme" [ref=e235] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e238]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e239]
      - link "Historia /monitoringi/monitoring-sadow-apelacyjnych/~uprawnienia" [ref=e240] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e241]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e242]
      - link "Wersje Django 5.2.17" [ref=e243] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e244]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e245]
      - 'link "Czas CPU: 127.95ms (119.54ms)" [ref=e246] [cursor=pointer]':
        - /url: "#"
    - listitem [ref=e247]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e248]
      - link "Ustawienia" [ref=e249] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e250]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e251]
      - link "Nagłówki" [ref=e252] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e253]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e254]
      - link "Zapytania MonitoringPermissionView" [ref=e255] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e256]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e257]
      - link "SQL 8 queries in 5.94ms" [ref=e258] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e259]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e260]
      - link "Pliki statyczne 3 użyte plików" [ref=e261] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e262]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e263]
      - link "Templatki monitorings/monitoring_permissions.html" [ref=e264] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e265]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e266]
      - link "Alerty" [ref=e267] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e268]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e269]
      - link "Cache 2 wywołania w 0.12ms" [ref=e270] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e271]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e272]
      - link "Sygnały 88 odbiorców 15 sygnałów" [ref=e273] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e274]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e275]
      - link "Gmina" [ref=e276] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e277]:
      - checkbox "Enable for next and successive requests" [ref=e278]
      - generic [ref=e279]: Przechwycone przekierowania
    - listitem [ref=e280]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e281]
      - link "Profilowanie" [ref=e282] [cursor=pointer]:
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