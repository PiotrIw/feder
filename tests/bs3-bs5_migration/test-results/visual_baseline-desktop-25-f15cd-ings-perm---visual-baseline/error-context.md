# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: visual_baseline.spec.ts >> desktop (2560px) >> monitorings-perm - visual baseline
- Location: tests/bs3-bs5_migration/visual_baseline.spec.ts:10:11

# Error details

```
Error: expect(page).toHaveScreenshot(expected) failed

  369873 pixels (ratio 0.11 of all image pixels) are different.

  Snapshot: monitorings-perm-desktop.png

Call log:
  - Expect "toHaveScreenshot(monitorings-perm-desktop.png)" with timeout 5000ms
    - verifying given screenshot expectation
  - taking page screenshot
    - disabled all CSS animations
  - waiting for fonts to load...
  - fonts loaded
  - 369873 pixels (ratio 0.11 of all image pixels) are different.
  - waiting 100ms before taking screenshot
  - taking page screenshot
    - disabled all CSS animations
  - waiting for fonts to load...
  - fonts loaded
  - captured a stable screenshot
  - 369873 pixels (ratio 0.11 of all image pixels) are different.

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
      - generic [ref=e65]:
        - link "Edytuj" [ref=e66] [cursor=pointer]:
          - /url: /monitoringi/monitoring-sadow-apelacyjnych/~edytuj
        - link "Aktualizuj wyniki" [ref=e67] [cursor=pointer]:
          - /url: /monitoringi/monitoring-sadow-apelacyjnych/~results-update
        - link "Przypisz" [ref=e68] [cursor=pointer]:
          - /url: /monitoringi/monitoring-sadow-apelacyjnych/~przypisz
        - link "Usuń" [ref=e69] [cursor=pointer]:
          - /url: /monitoringi/monitoring-sadow-apelacyjnych/~usun
        - link "Utwórz sprawę" [ref=e70] [cursor=pointer]:
          - /url: /sprawy/~utworz-5
        - link "Wiadomość masowa" [ref=e71] [cursor=pointer]:
          - /url: /monitoringi/monitoring-sadow-apelacyjnych/~wiadomosc-masowa
        - link "Uprawnienia" [ref=e72] [cursor=pointer]:
          - /url: /monitoringi/monitoring-sadow-apelacyjnych/~uprawnienia
        - link "Lista alertów" [ref=e73] [cursor=pointer]:
          - /url: /alerty/monitoring-5
        - link "Zobacz dzienniki" [ref=e74] [cursor=pointer]:
          - /url: /listy/logi/monitoring-5
        - link "Zobacz tagi" [ref=e76] [cursor=pointer]:
          - /url: /sprawy/tagi/monitoring-5
        - link "Zobacz raport" [ref=e78] [cursor=pointer]:
          - /url: /monitoringi/monitoring-sadow-apelacyjnych/raport
        - link "Zobacz tabelę spraw" [ref=e80] [cursor=pointer]:
          - /url: /monitoringi/monitoring-sadow-apelacyjnych/monitoring_cases_table
      - heading [level=2] [ref=e83]:
        - link "Monitoring sądów apelacyjnych" [ref=e85] [cursor=pointer]:
          - /url: /monitoringi/monitoring-sadow-apelacyjnych
      - table [ref=e87]:
        - rowgroup [ref=e88]:
          - row [ref=e89]:
            - cell [ref=e90]:
              - link "Dodaj użytkownika" [ref=e91] [cursor=pointer]:
                - /url: /monitoringi/monitoring-sadow-apelacyjnych/~uprawnienia/~dodaj
            - rowheader [ref=e92]:
              - link "Szymon_Osowski" [ref=e93] [cursor=pointer]:
                - /url: /monitoringi/monitoring-sadow-apelacyjnych/~uprawnienia-9
            - columnheader [ref=e94]:
              - link "adobrawy" [ref=e95] [cursor=pointer]:
                - /url: /monitoringi/monitoring-sadow-apelacyjnych/~uprawnienia-2
          - row [ref=e96]:
            - rowheader "Może tworzyć sprawę" [ref=e97]
            - cell [ref=e98]
            - cell [ref=e100]
          - row [ref=e102]:
            - rowheader "Dodaj szkic odpowiedzi" [ref=e103]
            - cell [ref=e104]
            - cell [ref=e106]
          - row [ref=e108]:
            - rowheader "Może dodawać list" [ref=e109]
            - cell [ref=e110]
            - cell [ref=e112]
          - row [ref=e114]:
            - rowheader "Can add questionary" [ref=e115]
            - cell [ref=e116]
            - cell [ref=e118]
          - row [ref=e120]:
            - rowheader "Can add task" [ref=e121]
            - cell [ref=e122]
            - cell [ref=e124]
          - row [ref=e126]:
            - rowheader "Może zmieniać alert" [ref=e127]
            - cell [ref=e128]
            - cell [ref=e130]
          - row [ref=e132]:
            - rowheader "Może zmieniać sprawę" [ref=e133]
            - cell [ref=e134]
            - cell [ref=e136]
          - row [ref=e138]:
            - rowheader "Może zmieniać monitoring" [ref=e139]
            - cell [ref=e140]
            - cell [ref=e142]
          - row [ref=e144]:
            - rowheader "Can change questionary" [ref=e145]
            - cell [ref=e146]
            - cell [ref=e148]
          - row [ref=e150]:
            - rowheader "Can change task" [ref=e151]
            - cell [ref=e152]
            - cell [ref=e154]
          - row [ref=e156]:
            - rowheader "Może usuwać alert" [ref=e157]
            - cell [ref=e158]
            - cell [ref=e160]
          - row [ref=e162]:
            - rowheader "Może usuwać sprawę" [ref=e163]
            - cell [ref=e164]
            - cell [ref=e166]
          - row [ref=e168]:
            - rowheader "Może usunąć monitoring" [ref=e169]
            - cell [ref=e170]
            - cell [ref=e172]
          - row [ref=e174]:
            - rowheader "Can delete questionary" [ref=e175]
            - cell [ref=e176]
            - cell [ref=e178]
          - row [ref=e180]:
            - rowheader "Can delete task" [ref=e181]
            - cell [ref=e182]
            - cell [ref=e184]
          - row [ref=e186]:
            - rowheader "Może zarządzać uprawnieniami" [ref=e187]
            - cell [ref=e188]
            - cell [ref=e190]
          - row [ref=e192]:
            - rowheader "Może odpowiadać" [ref=e193]
            - cell [ref=e194]
            - cell [ref=e196]
          - row [ref=e198]:
            - rowheader "Can select answer" [ref=e199]
            - cell [ref=e200]
            - cell [ref=e202]
          - row [ref=e204]:
            - rowheader "Może dodawać alert" [ref=e205]
            - cell [ref=e206]
            - cell [ref=e208]
      - generic [ref=e210]:
        - generic [ref=e211]:
          - text: Sieć Obywatelska - Watchdog Polska ul. Szpitalna 5/5 00-031 Warszawa
          - paragraph [ref=e212]:
            - link "Klauzula RODO" [ref=e213] [cursor=pointer]:
              - /url: https://fedrowanie.siecobywatelska.pl/media_internal/tinycontent/uploads/11_KLAUZULA.-.dane.z.udip.na.stronach.www.i.w.mediach.spol.pdf
        - generic [ref=e214]:
          - text: "silnik:"
          - link "jawne.info.pl" [ref=e215] [cursor=pointer]:
            - /url: http://jawne.info.pl
          - text: "| v1.5.77.deps |"
          - link "GitHub" [ref=e216] [cursor=pointer]:
            - /url: https://github.com/watchdogpolska/feder
          - text: –
          - link "efd8a3b" [ref=e218] [cursor=pointer]:
            - /url: https://github.com/watchdogpolska/feder/compare/efd8a3b2...master
          - text: "|"
          - link "API" [ref=e219] [cursor=pointer]:
            - /url: /api/
        - generic [ref=e221]: Ta strona wykorzystuje cookies.
  - list [ref=e223]:
    - listitem [ref=e224]:
      - link "Ukryj »" [ref=e225] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e226]:
      - link "Toggle Theme" [ref=e227] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e230]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e231]
      - link "Historia /monitoringi/monitoring-sadow-apelacyjnych/~uprawnienia" [ref=e232] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e233]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e234]
      - link "Wersje Django 5.2.17" [ref=e235] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e236]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e237]
      - 'link "Czas CPU: 87.77ms (91.05ms)" [ref=e238] [cursor=pointer]':
        - /url: "#"
    - listitem [ref=e239]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e240]
      - link "Ustawienia" [ref=e241] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e242]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e243]
      - link "Nagłówki" [ref=e244] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e245]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e246]
      - link "Zapytania MonitoringPermissionView" [ref=e247] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e248]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e249]
      - link "SQL 8 queries in 3.69ms" [ref=e250] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e251]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e252]
      - link "Pliki statyczne 3 użyte plików" [ref=e253] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e254]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e255]
      - link "Templatki monitorings/monitoring_permissions.html" [ref=e256] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e257]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e258]
      - link "Alerty" [ref=e259] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e260]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e261]
      - link "Cache 2 wywołania w 0.14ms" [ref=e262] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e263]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e264]
      - link "Sygnały 88 odbiorców 15 sygnałów" [ref=e265] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e266]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e267]
      - link "Gmina" [ref=e268] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e269]:
      - checkbox "Enable for next and successive requests" [ref=e270]
      - generic [ref=e271]: Przechwycone przekierowania
    - listitem [ref=e272]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e273]
      - link "Profilowanie" [ref=e274] [cursor=pointer]:
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