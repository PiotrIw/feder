# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: visual_baseline.spec.ts >> desktop (2560px) >> home - visual baseline
- Location: tests/bs3-bs5_migration/visual_baseline.spec.ts:10:11

# Error details

```
Error: expect(page).toHaveScreenshot(expected) failed

  482724 pixels (ratio 0.14 of all image pixels) are different.

  Snapshot: home-desktop.png

Call log:
  - Expect "toHaveScreenshot(home-desktop.png)" with timeout 5000ms
    - verifying given screenshot expectation
  - taking page screenshot
    - disabled all CSS animations
  - waiting for fonts to load...
  - fonts loaded
  - 482724 pixels (ratio 0.14 of all image pixels) are different.
  - waiting 100ms before taking screenshot
  - taking page screenshot
    - disabled all CSS animations
  - waiting for fonts to load...
  - fonts loaded
  - captured a stable screenshot
  - 482724 pixels (ratio 0.14 of all image pixels) are different.

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
      - generic [ref=e62]:
        - generic [ref=e63]:
          - heading "Obywatelsko fedrujmy dane!" [level=1] [ref=e64]
          - heading "Razem przenieśmy debatę publiczną do nowej epoki!" [level=4] [ref=e65]
          - paragraph [ref=e66]:
            - button "Czytaj więcej" [ref=e67] [cursor=pointer]
        - generic [ref=e68]:
          - generic [ref=e70]:
            - heading "Województwa" [level=2] [ref=e71]
            - img [ref=e73]:
              - generic "Podkarpackie" [ref=e75] [cursor=pointer]
              - generic "Małopolskie" [ref=e77] [cursor=pointer]
              - generic "Śląskie" [ref=e79] [cursor=pointer]
              - generic "Opolskie" [ref=e81] [cursor=pointer]
              - generic "Dolnośląskie" [ref=e83] [cursor=pointer]
              - generic "Świętokrzyskie" [ref=e85] [cursor=pointer]
              - generic "Lubelskie" [ref=e87] [cursor=pointer]
              - generic "Łódzkie" [ref=e89] [cursor=pointer]
              - generic "Mazowieckie" [ref=e91] [cursor=pointer]
              - generic "Wielkopolska" [ref=e93] [cursor=pointer]
              - generic "Lubuskie" [ref=e95] [cursor=pointer]
              - generic "Kujawsko-pomorskie" [ref=e97] [cursor=pointer]
              - generic "Podlaskie" [ref=e99] [cursor=pointer]
              - generic "Zachodniopomorskie" [ref=e101] [cursor=pointer]
              - generic "Warmińsko-mazurskie" [ref=e103] [cursor=pointer]
              - generic "Pomorskie" [ref=e105] [cursor=pointer]
          - table [ref=e108]:
            - rowgroup [ref=e109]:
              - row [ref=e110]:
                - columnheader "Województwo" [ref=e111]
                - columnheader "Liczba instytucji" [ref=e112]
                - columnheader "Liczba spraw" [ref=e113]
              - row [ref=e114]:
                - cell "Dolnośląskie" [ref=e115]
                - cell "2626" [ref=e116]
                - cell "5095" [ref=e117]
              - row [ref=e118]:
                - cell "Kujawsko-Pomorskie" [ref=e119]
                - cell "2003" [ref=e120]
                - cell "4153" [ref=e121]
              - row [ref=e122]:
                - cell "Lubelskie" [ref=e123]
                - cell "2484" [ref=e124]
                - cell "5571" [ref=e125]
              - row [ref=e126]:
                - cell "Lubuskie" [ref=e127]
                - cell "1122" [ref=e128]
                - cell "2434" [ref=e129]
              - row [ref=e130]:
                - cell "Łódzkie" [ref=e131]
                - cell "2431" [ref=e132]
                - cell "4813" [ref=e133]
              - row [ref=e134]:
                - cell "Małopolskie" [ref=e135]
                - cell "3517" [ref=e136]
                - cell "4764" [ref=e137]
              - row [ref=e138]:
                - cell "Mazowieckie" [ref=e139]
                - cell "5147" [ref=e140]
                - cell "8796" [ref=e141]
              - row [ref=e142]:
                - cell "Opolskie" [ref=e143]
                - cell "1146" [ref=e144]
                - cell "1894" [ref=e145]
              - row [ref=e146]:
                - cell "Podkarpackie" [ref=e147]
                - cell "2617" [ref=e148]
                - cell "4150" [ref=e149]
              - row [ref=e150]:
                - cell "Podlaskie" [ref=e151]
                - cell "1274" [ref=e152]
                - cell "3005" [ref=e153]
              - row [ref=e154]:
                - cell "Pomorskie" [ref=e155]
                - cell "2099" [ref=e156]
                - cell "3413" [ref=e157]
              - row [ref=e158]:
                - cell "Śląskie" [ref=e159]
                - cell "3933" [ref=e160]
                - cell "4940" [ref=e161]
              - row [ref=e162]:
                - cell "Świętokrzyskie" [ref=e163]
                - cell "1418" [ref=e164]
                - cell "2618" [ref=e165]
              - row [ref=e166]:
                - cell "Warmińsko-Mazurskie" [ref=e167]
                - cell "1700" [ref=e168]
                - cell "3250" [ref=e169]
              - row [ref=e170]:
                - cell "Wielkopolskie" [ref=e171]
                - cell "3544" [ref=e172]
                - cell "5950" [ref=e173]
              - row [ref=e174]:
                - cell "Zachodniopomorskie" [ref=e175]
                - cell "1627" [ref=e176]
                - cell "3226" [ref=e177]
              - row [ref=e178]:
                - cell "Wszystkie" [ref=e179]
                - cell "38689" [ref=e180]
                - cell "68075" [ref=e181]
          - generic [ref=e183]:
            - heading "Ostatnie monitoringi" [level=2] [ref=e184]
            - list [ref=e186]:
              - listitem [ref=e187]:
                - link "Kontrole punktów gastronomicznych" [ref=e188] [cursor=pointer]:
                  - /url: /monitoringi/kontrole-punktow-gastronomicznych
              - listitem [ref=e189]:
                - link "Zespoły w ministerstwach" [ref=e190] [cursor=pointer]:
                  - /url: /monitoringi/zespoly-w-ministerstwach
              - listitem [ref=e191]:
                - link "Ministerstwa - baza danych umów cywilnoprawnych" [ref=e192] [cursor=pointer]:
                  - /url: /monitoringi/ministerstwa-baza-danych-umow-cywilnoprawnych
              - listitem [ref=e193]:
                - link "Nagrody w ministerstwach 2024" [ref=e194] [cursor=pointer]:
                  - /url: /monitoringi/nagrody-w-ministerstwach-2
              - listitem [ref=e195]:
                - link "Lasy Państwowe a SLAPPy" [ref=e196] [cursor=pointer]:
                  - /url: /monitoringi/lasy-panstwowe-a-slappy
              - listitem [ref=e197]:
                - link "Wnioski o informację o środowisku w 2023 - RDOŚ-ie" [ref=e198] [cursor=pointer]:
                  - /url: /monitoringi/wnioski-o-informacje-o-srodowisku-w-2023-rdos-ie-gdos-parki-narodowe
              - listitem [ref=e199]:
                - link "Wnioski o informację o środowisku w 2023 - gminy, starostwa, urzędy marszałkowskie, GDOŚ, parki" [ref=e200] [cursor=pointer]:
                  - /url: /monitoringi/wnioski-o-informacje-o-srodowisku-w-2023-gminy-starostwa-urzedy-marszalkowskie
              - listitem [ref=e201]:
                - link "Wnioski o informację w 2023 - Komendanci Wojewódzcy i Stołeczny Policji" [ref=e202] [cursor=pointer]:
                  - /url: /monitoringi/wnioski-o-informacje-w-2023-komendanci-wojewodzcy-i-stoleczny-policji
              - listitem [ref=e203]:
                - link "Wnioski o informację w 2023 - pytamy wojewodów o straże gminne (miejskie)" [ref=e204] [cursor=pointer]:
                  - /url: /monitoringi/wnioski-o-informacje-w-2023-pytamy-wojewodow-o-straze-gminne-miejskie
              - listitem [ref=e205]:
                - link "Doradcy Marszałków Sejmu i Senatu" [ref=e206] [cursor=pointer]:
                  - /url: /monitoringi/doradcy-marszalkow-sejmu-i-senatu
              - listitem [ref=e207]:
                - link "Karty płatnicze w ministerstwach" [ref=e208] [cursor=pointer]:
                  - /url: /monitoringi/karty-platnicze-w-ministerstwach
              - listitem [ref=e209]:
                - link "Wnioski o informację w 2023 - samorządowe kolegia odwoławcze" [ref=e210] [cursor=pointer]:
                  - /url: /monitoringi/wnioski-o-informacje-w-2023-samorzadowe-kolegia-odwolawcze
              - listitem [ref=e211]:
                - link "Wnioski o informację w 2023 - Akademia Wymiaru Sprawiedliwości" [ref=e212] [cursor=pointer]:
                  - /url: /monitoringi/wnioski-o-informacje-w-2023-akademia-wymiaru-sprawiedliwosci
              - listitem [ref=e213]:
                - link "Wnioski o informację w 2023 - Komendanci Ośrodków Szkolenia Służby Więziennej" [ref=e214] [cursor=pointer]:
                  - /url: /monitoringi/wnioski-o-informacje-w-2023-komendanci-osrodkow-szkolenia-sluzby-wieziennej
              - listitem [ref=e215]:
                - link "Wnioski o informację w 2023 - Dyrektorzy Okręgowi Służby Więziennej" [ref=e216] [cursor=pointer]:
                  - /url: /monitoringi/wnioski-o-informacje-w-2023-dyrektorzy-okregowi-sluzby-wieziennej
              - listitem [ref=e217]:
                - link "Wnioski o informację w 2023 - Dyrektor Generalny Służby Więziennej" [ref=e218] [cursor=pointer]:
                  - /url: /monitoringi/wnioski-o-informacje-w-2023-dyrektor-generalny-sluzby-wieziennej
      - generic [ref=e219]:
        - generic [ref=e220]:
          - text: Sieć Obywatelska - Watchdog Polska ul. Szpitalna 5/5 00-031 Warszawa
          - paragraph [ref=e221]:
            - link "Klauzula RODO" [ref=e222] [cursor=pointer]:
              - /url: https://fedrowanie.siecobywatelska.pl/media_internal/tinycontent/uploads/11_KLAUZULA.-.dane.z.udip.na.stronach.www.i.w.mediach.spol.pdf
        - generic [ref=e223]:
          - text: "silnik:"
          - link "jawne.info.pl" [ref=e224] [cursor=pointer]:
            - /url: http://jawne.info.pl
          - text: "| v1.5.77.deps |"
          - link "GitHub" [ref=e225] [cursor=pointer]:
            - /url: https://github.com/watchdogpolska/feder
          - text: –
          - link "efd8a3b" [ref=e227] [cursor=pointer]:
            - /url: https://github.com/watchdogpolska/feder/compare/efd8a3b2...master
          - text: "|"
          - link "API" [ref=e228] [cursor=pointer]:
            - /url: /api/
        - generic [ref=e230]: Ta strona wykorzystuje cookies.
  - list [ref=e232]:
    - listitem [ref=e233]:
      - link "Ukryj »" [ref=e234] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e235]:
      - link "Toggle Theme" [ref=e236] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e239]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e240]
      - link "Historia /" [ref=e241] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e242]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e243]
      - link "Wersje Django 5.2.17" [ref=e244] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e245]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e246]
      - 'link "Czas CPU: 150.74ms (353.16ms)" [ref=e247] [cursor=pointer]':
        - /url: "#"
    - listitem [ref=e248]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e249]
      - link "Ustawienia" [ref=e250] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e251]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e252]
      - link "Nagłówki" [ref=e253] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e254]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e255]
      - link "Zapytania HomeView" [ref=e256] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e257]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e258]
      - link "SQL 40 queries in 213.37ms" [ref=e259] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e260]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e261]
      - link "Pliki statyczne 3 użyte plików" [ref=e262] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e263]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e264]
      - link "Templatki main/home.html" [ref=e265] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e266]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e267]
      - link "Alerty" [ref=e268] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e269]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e270]
      - link "Cache 2 wywołania w 0.12ms" [ref=e271] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e272]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e273]
      - link "Sygnały 88 odbiorców 15 sygnałów" [ref=e274] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e275]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e276]
      - link "Gmina" [ref=e277] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e278]:
      - checkbox "Enable for next and successive requests" [ref=e279]
      - generic [ref=e280]: Przechwycone przekierowania
    - listitem [ref=e281]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e282]
      - link "Profilowanie" [ref=e283] [cursor=pointer]:
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