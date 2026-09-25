# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: visual_baseline.spec.ts >> desktop (2560px) >> letters-unrecognized-list - visual baseline
- Location: tests/bs3-bs5_migration/visual_baseline.spec.ts:10:11

# Error details

```
Error: expect(page).toHaveScreenshot(expected) failed

  Expected an image 2560px by 1497px, received 2560px by 1518px. 416248 pixels (ratio 0.11 of all image pixels) are different.

  Snapshot: letters-unrecognized-list-desktop.png

Call log:
  - Expect "toHaveScreenshot(letters-unrecognized-list-desktop.png)" with timeout 5000ms
    - verifying given screenshot expectation
  - taking page screenshot
    - disabled all CSS animations
  - waiting for fonts to load...
  - fonts loaded
  - Expected an image 2560px by 1497px, received 2560px by 1518px. 416248 pixels (ratio 0.11 of all image pixels) are different.
  - waiting 100ms before taking screenshot
  - taking page screenshot
    - disabled all CSS animations
  - waiting for fonts to load...
  - fonts loaded
  - captured a stable screenshot
  - Expected an image 2560px by 1497px, received 2560px by 1518px. 416248 pixels (ratio 0.11 of all image pixels) are different.

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
      - heading "Listy nieprzypisane do spraw" [level=2] [ref=e62]
      - generic [ref=e63]:
        - generic [ref=e64]:
          - generic [ref=e66]:
            - generic [ref=e67]:
              - generic [ref=e68]: Tytuł
              - textbox "Tytuł" [ref=e69]
            - generic [ref=e70]:
              - generic [ref=e71]: Data utworzenia
              - combobox "Data utworzenia" [ref=e72]:
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
            - generic [ref=e73]:
              - generic [ref=e74]: Record case Instytucja
              - combobox [aria-hidden] [ref=e75]
              - combobox [ref=e78] [cursor=pointer]:
                - textbox
            - generic [ref=e79]:
              - generic [ref=e80]: Ma plik .eml?
              - combobox "Ma plik .eml?" [ref=e81]:
                - option "Nieznany" [selected]
                - option "Tak"
                - option "Nie"
            - button "Filtruj" [ref=e82] [cursor=pointer]
          - table [ref=e85]:
            - rowgroup [ref=e86]:
              - row [ref=e87]:
                - columnheader [ref=e88]
                - columnheader "Sprawa nierozpoznana" [ref=e89]
                - columnheader "Sprawa rozpoznane" [ref=e90]
            - rowgroup [ref=e91]:
              - row [ref=e92]:
                - columnheader "Okres" [ref=e93]
                - columnheader "Liczba listów" [ref=e94]
                - columnheader "Nie spam" [ref=e95]
                - columnheader "Spam" [ref=e96]
                - columnheader "Spam" [ref=e97]
                - columnheader "Nie spam" [ref=e98]
            - rowgroup [ref=e99]:
              - row [ref=e100]:
                - cell "Dzisiaj" [ref=e101]
                - cell "0" [ref=e102]
                - cell "0" [ref=e103]
                - cell "0" [ref=e104]
                - cell "0" [ref=e105]
                - cell "0" [ref=e106]
              - row [ref=e107]:
                - cell "Wczoraj" [ref=e108]
                - cell "0" [ref=e109]
                - cell "0" [ref=e110]
                - cell "0" [ref=e111]
                - cell "0" [ref=e112]
                - cell "0" [ref=e113]
              - row [ref=e114]:
                - cell "Ostatnie 7 dni" [ref=e115]
                - cell "0" [ref=e116]
                - cell "0" [ref=e117]
                - cell "0" [ref=e118]
                - cell "0" [ref=e119]
                - cell "0" [ref=e120]
              - row [ref=e121]:
                - cell "Ten miesiąc" [ref=e122]
                - cell "0" [ref=e123]
                - cell "0" [ref=e124]
                - cell "0" [ref=e125]
                - cell "0" [ref=e126]
                - cell "0" [ref=e127]
              - row [ref=e128]:
                - cell "Ten rok" [ref=e129]
                - cell "0" [ref=e130]
                - cell "0" [ref=e131]
                - cell "0" [ref=e132]
                - cell "0" [ref=e133]
                - cell "0" [ref=e134]
              - row [ref=e135]:
                - cell "2025" [ref=e136]
                - cell "0" [ref=e137]
                - cell "0" [ref=e138]
                - cell "0" [ref=e139]
                - cell "0" [ref=e140]
                - cell "0" [ref=e141]
              - row [ref=e142]:
                - cell "2024" [ref=e143]
                - cell "46365" [ref=e144]
                - cell "3" [ref=e145]
                - cell "22803" [ref=e146]
                - cell "11035" [ref=e147]
                - cell "12524" [ref=e148]
              - row [ref=e149]:
                - cell "2023" [ref=e150]
                - cell "53486" [ref=e151]
                - cell "2" [ref=e152]
                - cell "25609" [ref=e153]
                - cell "3163" [ref=e154]
                - cell "24712" [ref=e155]
              - row [ref=e156]:
                - cell "2022" [ref=e157]
                - cell "18356" [ref=e158]
                - cell "4" [ref=e159]
                - cell "21" [ref=e160]
                - cell "264" [ref=e161]
                - cell "18067" [ref=e162]
              - row [ref=e163]:
                - cell "2021" [ref=e164]
                - cell "23961" [ref=e165]
                - cell "131" [ref=e166]
                - cell "0" [ref=e167]
                - cell "0" [ref=e168]
                - cell "23830" [ref=e169]
              - row [ref=e170]:
                - cell "2020" [ref=e171]
                - cell "34775" [ref=e172]
                - cell "358" [ref=e173]
                - cell "0" [ref=e174]
                - cell "0" [ref=e175]
                - cell "34417" [ref=e176]
              - row [ref=e177]:
                - cell "2019" [ref=e178]
                - cell "12784" [ref=e179]
                - cell "444" [ref=e180]
                - cell "0" [ref=e181]
                - cell "0" [ref=e182]
                - cell "12340" [ref=e183]
              - row [ref=e184]:
                - cell "2018" [ref=e185]
                - cell "14455" [ref=e186]
                - cell "6" [ref=e187]
                - cell "0" [ref=e188]
                - cell "0" [ref=e189]
                - cell "14449" [ref=e190]
              - row [ref=e191]:
                - cell "2017" [ref=e192]
                - cell "8703" [ref=e193]
                - cell "0" [ref=e194]
                - cell "0" [ref=e195]
                - cell "0" [ref=e196]
                - cell "8703" [ref=e197]
              - row [ref=e198]:
                - cell "2016" [ref=e199]
                - cell "95" [ref=e200]
                - cell "0" [ref=e201]
                - cell "0" [ref=e202]
                - cell "0" [ref=e203]
                - cell "95" [ref=e204]
        - generic [ref=e205]:
          - paragraph [ref=e207]: Brak wierszy.
          - list [ref=e208]:
            - listitem [ref=e209]:
              - generic [aria-hidden]: ←
            - listitem [ref=e210]:
              - generic "Current Page" [ref=e211]: "1"
            - listitem [ref=e212]:
              - generic [aria-hidden]: →
      - generic [ref=e213]:
        - generic [ref=e214]:
          - text: Sieć Obywatelska - Watchdog Polska ul. Szpitalna 5/5 00-031 Warszawa
          - paragraph [ref=e215]:
            - link "Klauzula RODO" [ref=e216] [cursor=pointer]:
              - /url: https://fedrowanie.siecobywatelska.pl/media_internal/tinycontent/uploads/11_KLAUZULA.-.dane.z.udip.na.stronach.www.i.w.mediach.spol.pdf
        - generic [ref=e217]:
          - text: "silnik:"
          - link "jawne.info.pl" [ref=e218] [cursor=pointer]:
            - /url: http://jawne.info.pl
          - text: "| v1.5.77.deps |"
          - link "GitHub" [ref=e219] [cursor=pointer]:
            - /url: https://github.com/watchdogpolska/feder
          - text: –
          - link "efd8a3b" [ref=e221] [cursor=pointer]:
            - /url: https://github.com/watchdogpolska/feder/compare/efd8a3b2...master
          - text: "|"
          - link "API" [ref=e222] [cursor=pointer]:
            - /url: /api/
        - generic [ref=e224]: Ta strona wykorzystuje cookies.
  - list [ref=e226]:
    - listitem [ref=e227]:
      - link "Ukryj »" [ref=e228] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e229]:
      - link "Toggle Theme" [ref=e230] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e233]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e234]
      - link "Historia /listy/przypisz" [ref=e235] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e236]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e237]
      - link "Wersje Django 5.2.17" [ref=e238] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e239]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e240]
      - 'link "Czas CPU: 721.05ms (1742.57ms)" [ref=e241] [cursor=pointer]':
        - /url: "#"
    - listitem [ref=e242]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e243]
      - link "Ustawienia" [ref=e244] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e245]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e246]
      - link "Nagłówki" [ref=e247] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e248]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e249]
      - link "Zapytania UnrecognizedLetterListView" [ref=e250] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e251]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e252]
      - link "SQL 20 queries in 1369.95ms" [ref=e253] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e254]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e255]
      - link "Pliki statyczne 10 użytych plików" [ref=e256] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e257]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e258]
      - link "Templatki letters/letter_unrecognized_list.html" [ref=e259] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e260]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e261]
      - link "Alerty" [ref=e262] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e263]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e264]
      - link "Cache 2 wywołania w 0.12ms" [ref=e265] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e266]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e267]
      - link "Sygnały 88 odbiorców 15 sygnałów" [ref=e268] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e269]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e270]
      - link "Gmina" [ref=e271] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e272]:
      - checkbox "Enable for next and successive requests" [ref=e273]
      - generic [ref=e274]: Przechwycone przekierowania
    - listitem [ref=e275]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e276]
      - link "Profilowanie" [ref=e277] [cursor=pointer]:
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