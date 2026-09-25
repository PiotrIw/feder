# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: visual_baseline.spec.ts >> desktop (2560px) >> monitorings-drafts - visual baseline
- Location: tests/bs3-bs5_migration/visual_baseline.spec.ts:10:11

# Error details

```
Error: expect(page).toHaveScreenshot(expected) failed

  358599 pixels (ratio 0.10 of all image pixels) are different.

  Snapshot: monitorings-drafts-desktop.png

Call log:
  - Expect "toHaveScreenshot(monitorings-drafts-desktop.png)" with timeout 5000ms
    - verifying given screenshot expectation
  - taking page screenshot
    - disabled all CSS animations
  - waiting for fonts to load...
  - fonts loaded
  - 358599 pixels (ratio 0.10 of all image pixels) are different.
  - waiting 100ms before taking screenshot
  - taking page screenshot
    - disabled all CSS animations
  - waiting for fonts to load...
  - fonts loaded
  - captured a stable screenshot
  - 358599 pixels (ratio 0.10 of all image pixels) are different.

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
        - listitem [ref=e62]: Monitoring sądów apelacyjnych
      - generic [ref=e64]:
        - link "Edytuj" [ref=e65] [cursor=pointer]:
          - /url: /monitoringi/monitoring-sadow-apelacyjnych/~edytuj
        - link "Aktualizuj wyniki" [ref=e66] [cursor=pointer]:
          - /url: /monitoringi/monitoring-sadow-apelacyjnych/~results-update
        - link "Przypisz" [ref=e67] [cursor=pointer]:
          - /url: /monitoringi/monitoring-sadow-apelacyjnych/~przypisz
        - link "Usuń" [ref=e68] [cursor=pointer]:
          - /url: /monitoringi/monitoring-sadow-apelacyjnych/~usun
        - link "Utwórz sprawę" [ref=e69] [cursor=pointer]:
          - /url: /sprawy/~utworz-5
        - link "Wiadomość masowa" [ref=e70] [cursor=pointer]:
          - /url: /monitoringi/monitoring-sadow-apelacyjnych/~wiadomosc-masowa
        - link "Uprawnienia" [ref=e71] [cursor=pointer]:
          - /url: /monitoringi/monitoring-sadow-apelacyjnych/~uprawnienia
        - link "Lista alertów" [ref=e72] [cursor=pointer]:
          - /url: /alerty/monitoring-5
        - link "Zobacz dzienniki" [ref=e73] [cursor=pointer]:
          - /url: /listy/logi/monitoring-5
        - link "Zobacz tagi" [ref=e75] [cursor=pointer]:
          - /url: /sprawy/tagi/monitoring-5
        - link "Zobacz raport" [ref=e77] [cursor=pointer]:
          - /url: /monitoringi/monitoring-sadow-apelacyjnych/raport
        - link "Zobacz tabelę spraw" [ref=e79] [cursor=pointer]:
          - /url: /monitoringi/monitoring-sadow-apelacyjnych/monitoring_cases_table
      - heading [level=2] [ref=e82]:
        - text: Monitoring sądów apelacyjnych
        - generic [ref=e84]:
          - text: przez
          - link "adobrawy" [ref=e85] [cursor=pointer]:
            - /url: /uzytkownik/adobrawy/
          - time [ref=e86]: 11 sierpnia 2017 02:47
      - generic [ref=e87]:
        - table [ref=e90]:
          - rowgroup [ref=e91]:
            - row [ref=e92]:
              - columnheader "Województwo" [ref=e93]
              - columnheader "Liczba spraw" [ref=e94]
              - columnheader "Liczba spraw z potw. odbioru" [ref=e95]
              - columnheader "Liczba spraw z odpowiedzią" [ref=e96]
            - row [ref=e97]:
              - cell "Dolnośląskie" [ref=e98]
              - cell "1" [ref=e99]
              - cell "0" [ref=e100]
              - cell "1" [ref=e101]
            - row [ref=e102]:
              - cell "Kujawsko-Pomorskie" [ref=e103]
              - cell "0" [ref=e104]
              - cell "0" [ref=e105]
              - cell "0" [ref=e106]
            - row [ref=e107]:
              - cell "Lubelskie" [ref=e108]
              - cell "1" [ref=e109]
              - cell "0" [ref=e110]
              - cell "1" [ref=e111]
            - row [ref=e112]:
              - cell "Lubuskie" [ref=e113]
              - cell "0" [ref=e114]
              - cell "0" [ref=e115]
              - cell "0" [ref=e116]
            - row [ref=e117]:
              - cell "Łódzkie" [ref=e118]
              - cell "1" [ref=e119]
              - cell "0" [ref=e120]
              - cell "1" [ref=e121]
            - row [ref=e122]:
              - cell "Małopolskie" [ref=e123]
              - cell "1" [ref=e124]
              - cell "0" [ref=e125]
              - cell "1" [ref=e126]
            - row [ref=e127]:
              - cell "Mazowieckie" [ref=e128]
              - cell "1" [ref=e129]
              - cell "0" [ref=e130]
              - cell "1" [ref=e131]
            - row [ref=e132]:
              - cell "Opolskie" [ref=e133]
              - cell "0" [ref=e134]
              - cell "0" [ref=e135]
              - cell "0" [ref=e136]
            - row [ref=e137]:
              - cell "Podkarpackie" [ref=e138]
              - cell "1" [ref=e139]
              - cell "0" [ref=e140]
              - cell "1" [ref=e141]
            - row [ref=e142]:
              - cell "Podlaskie" [ref=e143]
              - cell "1" [ref=e144]
              - cell "0" [ref=e145]
              - cell "1" [ref=e146]
            - row [ref=e147]:
              - cell "Pomorskie" [ref=e148]
              - cell "1" [ref=e149]
              - cell "0" [ref=e150]
              - cell "1" [ref=e151]
            - row [ref=e152]:
              - cell "Śląskie" [ref=e153]
              - cell "1" [ref=e154]
              - cell "0" [ref=e155]
              - cell "1" [ref=e156]
            - row [ref=e157]:
              - cell "Świętokrzyskie" [ref=e158]
              - cell "0" [ref=e159]
              - cell "0" [ref=e160]
              - cell "0" [ref=e161]
            - row [ref=e162]:
              - cell "Warmińsko-Mazurskie" [ref=e163]
              - cell "0" [ref=e164]
              - cell "0" [ref=e165]
              - cell "0" [ref=e166]
            - row [ref=e167]:
              - cell "Wielkopolskie" [ref=e168]
              - cell "1" [ref=e169]
              - cell "0" [ref=e170]
              - cell "1" [ref=e171]
            - row [ref=e172]:
              - cell "Zachodniopomorskie" [ref=e173]
              - cell "1" [ref=e174]
              - cell "0" [ref=e175]
              - cell "1" [ref=e176]
            - row [ref=e177]:
              - cell "Wszystkie" [ref=e178]
              - cell "11" [ref=e179]
              - cell "0" [ref=e180]
              - cell "11" [ref=e181]
        - generic [ref=e182]:
          - list [ref=e183]:
            - listitem [ref=e184]:
              - link "Instytucje i sprawy" [ref=e185] [cursor=pointer]:
                - /url: /monitoringi/monitoring-sadow-apelacyjnych
            - listitem [ref=e186]:
              - link "Listy" [ref=e187] [cursor=pointer]:
                - /url: /monitoringi/monitoring-sadow-apelacyjnych/listy
            - listitem [ref=e188]:
              - generic [ref=e189]: Projekty
            - listitem [ref=e190]:
              - link "Szablon" [ref=e191] [cursor=pointer]:
                - /url: /monitoringi/monitoring-sadow-apelacyjnych/template
            - listitem [ref=e192]:
              - link "Wyniki" [ref=e193] [cursor=pointer]:
                - /url: /monitoringi/monitoring-sadow-apelacyjnych/results
          - heading "Projekty" [level=3] [ref=e194]
          - paragraph [ref=e196]: Brak wierszy.
      - generic [ref=e197]:
        - generic [ref=e198]:
          - text: Sieć Obywatelska - Watchdog Polska ul. Szpitalna 5/5 00-031 Warszawa
          - paragraph [ref=e199]:
            - link "Klauzula RODO" [ref=e200] [cursor=pointer]:
              - /url: https://fedrowanie.siecobywatelska.pl/media_internal/tinycontent/uploads/11_KLAUZULA.-.dane.z.udip.na.stronach.www.i.w.mediach.spol.pdf
        - generic [ref=e201]:
          - text: "silnik:"
          - link "jawne.info.pl" [ref=e202] [cursor=pointer]:
            - /url: http://jawne.info.pl
          - text: "| v1.5.77.deps |"
          - link "GitHub" [ref=e203] [cursor=pointer]:
            - /url: https://github.com/watchdogpolska/feder
          - text: –
          - link "efd8a3b" [ref=e205] [cursor=pointer]:
            - /url: https://github.com/watchdogpolska/feder/compare/efd8a3b2...master
          - text: "|"
          - link "API" [ref=e206] [cursor=pointer]:
            - /url: /api/
        - generic [ref=e208]: Ta strona wykorzystuje cookies.
  - list [ref=e210]:
    - listitem [ref=e211]:
      - link "Ukryj »" [ref=e212] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e213]:
      - link "Toggle Theme" [ref=e214] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e217]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e218]
      - link "Historia /monitoringi/monitoring-sadow-apelacyjnych/projekty" [ref=e219] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e220]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e221]
      - link "Wersje Django 5.2.17" [ref=e222] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e223]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e224]
      - 'link "Czas CPU: 714.30ms (4530.69ms)" [ref=e225] [cursor=pointer]':
        - /url: "#"
    - listitem [ref=e226]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e227]
      - link "Ustawienia" [ref=e228] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e229]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e230]
      - link "Nagłówki" [ref=e231] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e232]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e233]
      - link "Zapytania DraftListMonitoringView" [ref=e234] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e235]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e236]
      - link "SQL 60 queries in 4164.93ms" [ref=e237] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e238]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e239]
      - link "Pliki statyczne 3 użyte plików" [ref=e240] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e241]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e242]
      - link "Templatki monitorings/monitoring_draft_list.html" [ref=e243] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e244]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e245]
      - link "Alerty" [ref=e246] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e247]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e248]
      - link "Cache 2 wywołania w 0.12ms" [ref=e249] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e250]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e251]
      - link "Sygnały 88 odbiorców 15 sygnałów" [ref=e252] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e253]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e254]
      - link "Gmina" [ref=e255] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e256]:
      - checkbox "Enable for next and successive requests" [ref=e257]
      - generic [ref=e258]: Przechwycone przekierowania
    - listitem [ref=e259]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e260]
      - link "Profilowanie" [ref=e261] [cursor=pointer]:
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