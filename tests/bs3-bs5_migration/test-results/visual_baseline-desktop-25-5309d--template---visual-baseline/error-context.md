# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: visual_baseline.spec.ts >> desktop (2560px) >> monitorings-template - visual baseline
- Location: tests/bs3-bs5_migration/visual_baseline.spec.ts:10:11

# Error details

```
Error: expect(page).toHaveScreenshot(expected) failed

  446975 pixels (ratio 0.13 of all image pixels) are different.

  Snapshot: monitorings-template-desktop.png

Call log:
  - Expect "toHaveScreenshot(monitorings-template-desktop.png)" with timeout 5000ms
    - verifying given screenshot expectation
  - taking page screenshot
    - disabled all CSS animations
  - waiting for fonts to load...
  - fonts loaded
  - 446975 pixels (ratio 0.13 of all image pixels) are different.
  - waiting 100ms before taking screenshot
  - taking page screenshot
    - disabled all CSS animations
  - waiting for fonts to load...
  - fonts loaded
  - captured a stable screenshot
  - 446975 pixels (ratio 0.13 of all image pixels) are different.

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
              - link "Projekty" [ref=e189] [cursor=pointer]:
                - /url: /monitoringi/monitoring-sadow-apelacyjnych/projekty
            - listitem [ref=e190]:
              - generic [ref=e191]: Szablon
            - listitem [ref=e192]:
              - link "Wyniki" [ref=e193] [cursor=pointer]:
                - /url: /monitoringi/monitoring-sadow-apelacyjnych/results
          - heading "Szablon" [level=3] [ref=e194]
          - generic [ref=e195]:
            - heading "Temat e-mail" [level=5] [ref=e196]
            - generic [ref=e197]: Wniosek o udostępnienie informacji publicznej
            - heading "Szablon" [level=5] [ref=e198]
            - paragraph [ref=e200]: "Stowarzyszenie Sieć Obywatelska Watchdog Polska wnosi o udostępnienie poprzez przesłanie następujących informacji: - adresu strony internetowej, na której znajdują się orzeczenia dyscyplinarne, wydane wobec sędziów przez tutejszy Sąd, - rejestru umów, zawartych w imieniu Sądu od 1 stycznia 2017 r. do 31 lipca 2017 r., zawierającego informacje w zakresie co najmniej dat zawartych umów, przedmiotach umów, stronach umów, kwotach umów, - adresu strony internetowej, na której znajduje się dokumentacja przebiegu i efektów kontroli, przeprowadzonych w sądzie, oraz wystąpienia, stanowiska, wnioski i opinie podmiotów ją przeprowadzających, - kalendarz spotkań prezesa (prezes) sądu, które odbył (odbyła) w lipcu 2017 r., - skany orzeczeń, zapadłych w wyniku wniesienia środka odwoławczego od orzeczenia sądu niższej instancji, zapadłego z oskarżenia o czyn zabroniony art. 231 Kodeksu karnego - wydanych przez Sąd w 2017 r., - skany orzeczeń, zapadłych w wyniku wniesienia środka odwoławczego od orzeczenia sądu niższej instancji, zapadłego z oskarżenia o czyn zabroniony art. 212 Kodeksu karnego - wydanych przez Sąd w 2017 r. Stowarzyszenie wnosi o udostępnienie wskazanych informacji w formie elektronicznej, na adres e-mail {{EMAIL}}. Katarzyna Batko-Tołuć, Bartosz Wilk - członkowie zarządu, zgodnie z zasadami reprezentacji"
            - heading "Podpis w e-mail" [level=5] [ref=e201]
            - paragraph [ref=e203]: "---"
          - generic [ref=e205]:
            - heading "Znormalizowany szablon odpowiedzi" [level=5] [ref=e207]
            - generic [ref=e208]: "Utworzony:"
          - generic [ref=e210]:
            - heading "Prompt normalizacji odpowiedzi listów" [level=5] [ref=e211]
            - generic [ref=e212]: LLM nie został włączony, więc żądanie normalizacji LLM nie zostanie wysłane.
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
      - link "Historia /monitoringi/monitoring-sadow-apelacyjnych/template" [ref=e235] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e236]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e237]
      - link "Wersje Django 5.2.17" [ref=e238] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e239]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e240]
      - 'link "Czas CPU: 209.37ms (228.00ms)" [ref=e241] [cursor=pointer]':
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
      - link "Zapytania MonitoringTemplateView" [ref=e250] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e251]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e252]
      - link "SQL 60 queries in 26.69ms" [ref=e253] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e254]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e255]
      - link "Pliki statyczne 3 użyte plików" [ref=e256] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e257]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e258]
      - link "Templatki monitorings/monitoring_template.html" [ref=e259] [cursor=pointer]:
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