# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: visual_baseline.spec.ts >> desktop (2560px) >> monitorings-update - visual baseline
- Location: tests/bs3-bs5_migration/visual_baseline.spec.ts:10:11

# Error details

```
Error: expect(page).toHaveScreenshot(expected) failed

  Expected an image 2560px by 1730px, received 2560px by 1726px. 425859 pixels (ratio 0.10 of all image pixels) are different.

  Snapshot: monitorings-update-desktop.png

Call log:
  - Expect "toHaveScreenshot(monitorings-update-desktop.png)" with timeout 5000ms
    - verifying given screenshot expectation
  - taking page screenshot
    - disabled all CSS animations
  - waiting for fonts to load...
  - fonts loaded
  - Expected an image 2560px by 1730px, received 2560px by 1726px. 425859 pixels (ratio 0.10 of all image pixels) are different.
  - waiting 100ms before taking screenshot
  - taking page screenshot
    - disabled all CSS animations
  - waiting for fonts to load...
  - fonts loaded
  - captured a stable screenshot
  - Expected an image 2560px by 1730px, received 2560px by 1726px. 425859 pixels (ratio 0.10 of all image pixels) are different.

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
        - listitem [ref=e64]: / Zaktualizuj monitoring
      - generic [ref=e66]:
        - link "Edytuj" [ref=e67] [cursor=pointer]:
          - /url: /monitoringi/monitoring-sadow-apelacyjnych/~edytuj
        - link "Aktualizuj wyniki" [ref=e68] [cursor=pointer]:
          - /url: /monitoringi/monitoring-sadow-apelacyjnych/~results-update
        - link "Przypisz" [ref=e69] [cursor=pointer]:
          - /url: /monitoringi/monitoring-sadow-apelacyjnych/~przypisz
        - link "Usuń" [ref=e70] [cursor=pointer]:
          - /url: /monitoringi/monitoring-sadow-apelacyjnych/~usun
        - link "Utwórz sprawę" [ref=e71] [cursor=pointer]:
          - /url: /sprawy/~utworz-5
        - link "Wiadomość masowa" [ref=e72] [cursor=pointer]:
          - /url: /monitoringi/monitoring-sadow-apelacyjnych/~wiadomosc-masowa
        - link "Uprawnienia" [ref=e73] [cursor=pointer]:
          - /url: /monitoringi/monitoring-sadow-apelacyjnych/~uprawnienia
        - link "Lista alertów" [ref=e74] [cursor=pointer]:
          - /url: /alerty/monitoring-5
        - link "Zobacz dzienniki" [ref=e75] [cursor=pointer]:
          - /url: /listy/logi/monitoring-5
        - link "Zobacz tagi" [ref=e77] [cursor=pointer]:
          - /url: /sprawy/tagi/monitoring-5
        - link "Zobacz raport" [ref=e79] [cursor=pointer]:
          - /url: /monitoringi/monitoring-sadow-apelacyjnych/raport
        - link "Zobacz tabelę spraw" [ref=e81] [cursor=pointer]:
          - /url: /monitoringi/monitoring-sadow-apelacyjnych/monitoring_cases_table
      - heading [level=2] [ref=e84]:
        - link "Monitoring sądów apelacyjnych" [ref=e86] [cursor=pointer]:
          - /url: /monitoringi/monitoring-sadow-apelacyjnych
      - generic [ref=e88]:
        - generic [ref=e89]:
          - group "Monitoring" [ref=e91]:
            - generic [ref=e93]:
              - generic [ref=e94]: Nazwa*
              - textbox "Nazwa*" [ref=e95]: Monitoring sądów apelacyjnych
            - generic [ref=e96]:
              - generic [ref=e97]: Opis
              - textbox "Opis" [ref=e98]
            - generic [ref=e100]:
              - checkbox "Powiadamiaj o alertach" [checked] [ref=e101]
              - generic [ref=e102]: Powiadamiaj o alertach
              - generic [ref=e103]: Powiadom o nowych alertach osoby, które mogą je widzieć
            - generic [ref=e105]:
              - checkbox "Czy publicznie widoczny?" [checked] [ref=e106]
              - generic [ref=e107]: Czy publicznie widoczny?
            - generic [ref=e109]:
              - checkbox "Czy ukrywać nowe sprawy przy przypisywaniu?" [ref=e110]
              - generic [ref=e111]: Czy ukrywać nowe sprawy przy przypisywaniu?
            - generic [ref=e113]:
              - checkbox "Korzystaj z LLM" [ref=e114]
              - generic [ref=e115]: Korzystaj z LLM
              - generic [ref=e116]: Przed włączeniem upewnij się, że treść wniosku nie będzie już zmieniana. Zawsze możesz wrócić do edycji i włączyć później.
          - group "Szablon" [ref=e118]:
            - generic [ref=e120]:
              - generic [ref=e121]: Temat*
              - textbox "Temat*" [ref=e122]: Wniosek o udostępnienie informacji publicznej
            - generic [ref=e123]:
              - generic [ref=e124]: Szablon*
              - application [ref=e125]:
                - generic [ref=e126]:
                  - generic [ref=e127]:
                    - menubar [ref=e129]:
                      - menuitem "Plik" [ref=e130] [cursor=pointer]
                      - menuitem "Edytuj" [ref=e132] [cursor=pointer]
                      - menuitem "Widok" [ref=e134] [cursor=pointer]
                      - menuitem "Wstaw" [ref=e136] [cursor=pointer]
                      - menuitem "Format" [ref=e138] [cursor=pointer]
                      - menuitem "Narzędzia" [ref=e140] [cursor=pointer]
                      - menuitem "Tabela" [ref=e142] [cursor=pointer]
                      - menuitem "Pomoc" [ref=e144] [cursor=pointer]
                    - group [ref=e146]:
                      - group [ref=e147]:
                        - toolbar [ref=e148]:
                          - button "Cofnij" [disabled] [ref=e149]
                          - button "Powtórz" [disabled] [ref=e153]
                        - toolbar [ref=e157]:
                          - button "Wysokość Linii" [ref=e158] [cursor=pointer]
                        - toolbar [ref=e165]:
                          - button "Pogrubienie" [ref=e166] [cursor=pointer]
                          - button "Kursywa" [ref=e170] [cursor=pointer]
                          - button "Kolor tła Czarny" [ref=e174]
                        - toolbar [ref=e182]:
                          - button "Wyrównaj do lewej" [ref=e183] [cursor=pointer]
                          - button "Wyrównaj do środka" [ref=e187] [cursor=pointer]
                          - button "Wyrównaj do prawej" [ref=e191] [cursor=pointer]
                          - button "Wyjustuj" [ref=e195] [cursor=pointer]
                        - toolbar [ref=e199]:
                          - button "Lista wypunktowana" [ref=e200] [cursor=pointer]
                          - button "Lista numerowana" [ref=e204] [cursor=pointer]
                          - button "Zmniejsz wcięcie" [disabled] [ref=e208]
                          - button "Zwiększ wcięcie" [ref=e212] [cursor=pointer]
                        - toolbar [ref=e216]:
                          - button "Znak Specjalny" [ref=e217] [cursor=pointer]
                        - toolbar [ref=e221]:
                          - button "Wyczyść formatowanie" [ref=e222] [cursor=pointer]
                        - toolbar [ref=e226]:
                          - button "Pomoc" [ref=e227] [cursor=pointer]
                  - iframe [ref=e236]:
                    - generic "Obszar tekstu sformatowanego. Naciśnij ALT-0, aby uzyskać pomoc." [ref=f1e1]:
                      - paragraph [ref=f1e2]: "Stowarzyszenie Sieć Obywatelska Watchdog Polska wnosi o udostępnienie poprzez przesłanie następujących informacji: - adresu strony internetowej, na której znajdują się orzeczenia dyscyplinarne, wydane wobec sędziów przez tutejszy Sąd, - rejestru umów, zawartych w imieniu Sądu od 1 stycznia 2017 r. do 31 lipca 2017 r., zawierającego informacje w zakresie co najmniej dat zawartych umów, przedmiotach umów, stronach umów, kwotach umów, - adresu strony internetowej, na której znajduje się dokumentacja przebiegu i efektów kontroli, przeprowadzonych w sądzie, oraz wystąpienia, stanowiska, wnioski i opinie podmiotów ją przeprowadzających, - kalendarz spotkań prezesa (prezes) sądu, które odbył (odbyła) w lipcu 2017 r., - skany orzeczeń, zapadłych w wyniku wniesienia środka odwoławczego od orzeczenia sądu niższej instancji, zapadłego z oskarżenia o czyn zabroniony art. 231 Kodeksu karnego - wydanych przez Sąd w 2017 r., - skany orzeczeń, zapadłych w wyniku wniesienia środka odwoławczego od orzeczenia sądu niższej instancji, zapadłego z oskarżenia o czyn zabroniony art. 212 Kodeksu karnego - wydanych przez Sąd w 2017 r. Stowarzyszenie wnosi o udostępnienie wskazanych informacji w formie elektronicznej, na adres e-mail {{EMAIL}}. Katarzyna Batko-Tołuć, Bartosz Wilk - członkowie zarządu, zgodnie z zasadami reprezentacji"
                - generic [ref=e237]:
                  - generic [ref=e238]:
                    - navigation [ref=e239]:
                      - button "p" [ref=e240]
                    - generic [ref=e241]: Naciśnij Alt+0, aby uzyskać pomoc
                    - generic [ref=e242]:
                      - button "175 sł." [ref=e243] [cursor=pointer]
                      - link "Build with TinyMCE" [ref=e245]:
                        - /url: https://www.tiny.cloud/powered-by-tiny?utm_campaign=poweredby&utm_source=tiny&utm_medium=referral&utm_content=v7
                        - text: Build with
                  - generic "Naciśnij klawisze strzałek w górę i w dół, aby zmienić rozmiar edytora." [ref=e252]
              - generic [ref=e256]: "Użyj: {{EMAIL}} aby umieścić adres odpowiedzi, {{ADRESAT}} aby umieścić nazwę adressata."
            - generic [ref=e257]:
              - generic [ref=e258]: Podpis w e-mail*
              - application [ref=e259]:
                - generic [ref=e260]:
                  - generic [ref=e261]:
                    - menubar [ref=e263]:
                      - menuitem "Plik" [ref=e264] [cursor=pointer]
                      - menuitem "Edytuj" [ref=e266] [cursor=pointer]
                      - menuitem "Widok" [ref=e268] [cursor=pointer]
                      - menuitem "Wstaw" [ref=e270] [cursor=pointer]
                      - menuitem "Format" [ref=e272] [cursor=pointer]
                      - menuitem "Narzędzia" [ref=e274] [cursor=pointer]
                      - menuitem "Tabela" [ref=e276] [cursor=pointer]
                      - menuitem "Pomoc" [ref=e278] [cursor=pointer]
                    - group [ref=e280]:
                      - group [ref=e281]:
                        - toolbar [ref=e282]:
                          - button "Cofnij" [disabled] [ref=e283]
                          - button "Powtórz" [disabled] [ref=e287]
                        - toolbar [ref=e291]:
                          - button "Wysokość Linii" [ref=e292] [cursor=pointer]
                        - toolbar [ref=e299]:
                          - button "Pogrubienie" [ref=e300] [cursor=pointer]
                          - button "Kursywa" [ref=e304] [cursor=pointer]
                          - button "Kolor tła Czarny" [ref=e308]
                        - toolbar [ref=e316]:
                          - button "Wyrównaj do lewej" [ref=e317] [cursor=pointer]
                          - button "Wyrównaj do środka" [ref=e321] [cursor=pointer]
                          - button "Wyrównaj do prawej" [ref=e325] [cursor=pointer]
                          - button "Wyjustuj" [ref=e329] [cursor=pointer]
                        - toolbar [ref=e333]:
                          - button "Lista wypunktowana" [ref=e334] [cursor=pointer]
                          - button "Lista numerowana" [ref=e338] [cursor=pointer]
                          - button "Zmniejsz wcięcie" [disabled] [ref=e342]
                          - button "Zwiększ wcięcie" [ref=e346] [cursor=pointer]
                        - toolbar [ref=e350]:
                          - button "Znak Specjalny" [ref=e351] [cursor=pointer]
                        - toolbar [ref=e355]:
                          - button "Wyczyść formatowanie" [ref=e356] [cursor=pointer]
                        - toolbar [ref=e360]:
                          - button "Pomoc" [ref=e361] [cursor=pointer]
                  - iframe [ref=e370]:
                    - generic "Obszar tekstu sformatowanego. Naciśnij ALT-0, aby uzyskać pomoc." [ref=f2e1]:
                      - paragraph [ref=f2e2]: "---"
                - generic [ref=e371]:
                  - generic [ref=e372]:
                    - navigation [ref=e373]:
                      - button "p" [ref=e374]
                    - generic [ref=e375]: Naciśnij Alt+0, aby uzyskać pomoc
                    - generic [ref=e376]:
                      - button "0 sł." [ref=e377] [cursor=pointer]
                      - link "Build with TinyMCE" [ref=e379]:
                        - /url: https://www.tiny.cloud/powered-by-tiny?utm_campaign=poweredby&utm_source=tiny&utm_medium=referral&utm_content=v7
                        - text: Build with
                  - generic "Naciśnij klawisze strzałek w górę i w dół, aby zmienić rozmiar edytora." [ref=e386]
              - generic [ref=e390]: Podpis w stopce e-maili, w tym w odpowiedziach na e-maile
            - generic [ref=e391]:
              - generic [ref=e392]: Domain*
              - combobox "Domain*" [ref=e393]:
                - option "---------"
                - option "fedrowanie.siecobywatelska.pl" [selected]
                - option "pokot.pl"
                - option "monitoring.bartoszwilk.pl"
                - option "info.lasyiobywatele.pl"
                - option "nijakowski.pl"
                - option "fedr.uratujzwierze.pl"
                - option "pytania.ofop.eu"
                - option "info.szkolajestnasza.pl"
              - generic [ref=e394]: Domena użyta do wysłania wiadomości
        - button "Aktualizuj" [ref=e397] [cursor=pointer]
      - generic [ref=e398]:
        - generic [ref=e399]:
          - text: Sieć Obywatelska - Watchdog Polska ul. Szpitalna 5/5 00-031 Warszawa
          - paragraph [ref=e400]:
            - link "Klauzula RODO" [ref=e401] [cursor=pointer]:
              - /url: https://fedrowanie.siecobywatelska.pl/media_internal/tinycontent/uploads/11_KLAUZULA.-.dane.z.udip.na.stronach.www.i.w.mediach.spol.pdf
        - generic [ref=e402]:
          - text: "silnik:"
          - link "jawne.info.pl" [ref=e403] [cursor=pointer]:
            - /url: http://jawne.info.pl
          - text: "| v1.5.77.deps |"
          - link "GitHub" [ref=e404] [cursor=pointer]:
            - /url: https://github.com/watchdogpolska/feder
          - text: –
          - link "efd8a3b" [ref=e406] [cursor=pointer]:
            - /url: https://github.com/watchdogpolska/feder/compare/efd8a3b2...master
          - text: "|"
          - link "API" [ref=e407] [cursor=pointer]:
            - /url: /api/
        - generic [ref=e409]: Ta strona wykorzystuje cookies.
  - list [ref=e411]:
    - listitem [ref=e412]:
      - link "Ukryj »" [ref=e413] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e414]:
      - link "Toggle Theme" [ref=e415] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e418]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e419]
      - link "Historia /monitoringi/monitoring-sadow-apelacyjnych/~edytuj" [ref=e420] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e421]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e422]
      - link "Wersje Django 5.2.17" [ref=e423] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e424]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e425]
      - 'link "Czas CPU: 174.73ms (180.41ms)" [ref=e426] [cursor=pointer]':
        - /url: "#"
    - listitem [ref=e427]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e428]
      - link "Ustawienia" [ref=e429] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e430]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e431]
      - link "Nagłówki" [ref=e432] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e433]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e434]
      - link "Zapytania MonitoringUpdateView" [ref=e435] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e436]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e437]
      - link "SQL 8 queries in 5.92ms" [ref=e438] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e439]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e440]
      - link "Pliki statyczne 5 użytych plików" [ref=e441] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e442]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e443]
      - link "Templatki monitorings/monitoring_form.html" [ref=e444] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e445]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e446]
      - link "Alerty" [ref=e447] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e448]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e449]
      - link "Cache 2 wywołania w 0.18ms" [ref=e450] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e451]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e452]
      - link "Sygnały 88 odbiorców 15 sygnałów" [ref=e453] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e454]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e455]
      - link "Gmina" [ref=e456] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e457]:
      - checkbox "Enable for next and successive requests" [ref=e458]
      - generic [ref=e459]: Przechwycone przekierowania
    - listitem [ref=e460]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e461]
      - link "Profilowanie" [ref=e462] [cursor=pointer]:
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