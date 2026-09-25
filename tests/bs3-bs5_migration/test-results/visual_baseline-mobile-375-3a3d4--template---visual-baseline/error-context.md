# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: visual_baseline.spec.ts >> mobile (375px) >> monitorings-template - visual baseline
- Location: tests/bs3-bs5_migration/visual_baseline.spec.ts:10:11

# Error details

```
Error: expect(page).toHaveScreenshot(expected) failed

  Expected an image 434px by 3344px, received 480px by 3420px. 251021 pixels (ratio 0.16 of all image pixels) are different.

  Snapshot: monitorings-template-mobile.png

Call log:
  - Expect "toHaveScreenshot(monitorings-template-mobile.png)" with timeout 5000ms
    - verifying given screenshot expectation
  - taking page screenshot
    - disabled all CSS animations
  - waiting for fonts to load...
  - fonts loaded
  - Expected an image 434px by 3344px, received 480px by 3420px. 251021 pixels (ratio 0.16 of all image pixels) are different.
  - waiting 100ms before taking screenshot
  - taking page screenshot
    - disabled all CSS animations
  - waiting for fonts to load...
  - fonts loaded
  - captured a stable screenshot
  - Expected an image 434px by 3344px, received 480px by 3420px. 251021 pixels (ratio 0.16 of all image pixels) are different.

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
              - generic [ref=e199]: Szablon
            - listitem [ref=e200]:
              - link "Wyniki" [ref=e201] [cursor=pointer]:
                - /url: /monitoringi/monitoring-sadow-apelacyjnych/results
          - heading "Szablon" [level=3] [ref=e202]
          - generic [ref=e203]:
            - heading "Temat e-mail" [level=5] [ref=e204]
            - generic [ref=e205]: Wniosek o udostępnienie informacji publicznej
            - heading "Szablon" [level=5] [ref=e206]
            - paragraph [ref=e208]: "Stowarzyszenie Sieć Obywatelska Watchdog Polska wnosi o udostępnienie poprzez przesłanie następujących informacji: - adresu strony internetowej, na której znajdują się orzeczenia dyscyplinarne, wydane wobec sędziów przez tutejszy Sąd, - rejestru umów, zawartych w imieniu Sądu od 1 stycznia 2017 r. do 31 lipca 2017 r., zawierającego informacje w zakresie co najmniej dat zawartych umów, przedmiotach umów, stronach umów, kwotach umów, - adresu strony internetowej, na której znajduje się dokumentacja przebiegu i efektów kontroli, przeprowadzonych w sądzie, oraz wystąpienia, stanowiska, wnioski i opinie podmiotów ją przeprowadzających, - kalendarz spotkań prezesa (prezes) sądu, które odbył (odbyła) w lipcu 2017 r., - skany orzeczeń, zapadłych w wyniku wniesienia środka odwoławczego od orzeczenia sądu niższej instancji, zapadłego z oskarżenia o czyn zabroniony art. 231 Kodeksu karnego - wydanych przez Sąd w 2017 r., - skany orzeczeń, zapadłych w wyniku wniesienia środka odwoławczego od orzeczenia sądu niższej instancji, zapadłego z oskarżenia o czyn zabroniony art. 212 Kodeksu karnego - wydanych przez Sąd w 2017 r. Stowarzyszenie wnosi o udostępnienie wskazanych informacji w formie elektronicznej, na adres e-mail {{EMAIL}}. Katarzyna Batko-Tołuć, Bartosz Wilk - członkowie zarządu, zgodnie z zasadami reprezentacji"
            - heading "Podpis w e-mail" [level=5] [ref=e209]
            - paragraph [ref=e211]: "---"
          - generic [ref=e213]:
            - heading "Znormalizowany szablon odpowiedzi" [level=5] [ref=e215]
            - generic [ref=e216]: "Utworzony:"
          - generic [ref=e218]:
            - heading "Prompt normalizacji odpowiedzi listów" [level=5] [ref=e219]
            - generic [ref=e220]: LLM nie został włączony, więc żądanie normalizacji LLM nie zostanie wysłane.
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
      - link "Historia /monitoringi/monitoring-sadow-apelacyjnych/template" [ref=e243] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e244]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e245]
      - link "Wersje Django 5.2.17" [ref=e246] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e247]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e248]
      - 'link "Czas CPU: 211.99ms (230.62ms)" [ref=e249] [cursor=pointer]':
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
      - link "Zapytania MonitoringTemplateView" [ref=e258] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e259]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e260]
      - link "SQL 60 queries in 26.13ms" [ref=e261] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e262]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e263]
      - link "Pliki statyczne 3 użyte plików" [ref=e264] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e265]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e266]
      - link "Templatki monitorings/monitoring_template.html" [ref=e267] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e268]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e269]
      - link "Alerty" [ref=e270] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e271]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e272]
      - link "Cache 2 wywołania w 0.12ms" [ref=e273] [cursor=pointer]:
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