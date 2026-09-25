# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: visual_baseline.spec.ts >> mobile (375px) >> monitorings-mass-message - visual baseline
- Location: tests/bs3-bs5_migration/visual_baseline.spec.ts:10:11

# Error details

```
Error: expect(page).toHaveScreenshot(expected) failed

  Expected an image 375px by 2507px, received 375px by 2454px. 215387 pixels (ratio 0.23 of all image pixels) are different.

  Snapshot: monitorings-mass-message-mobile.png

Call log:
  - Expect "toHaveScreenshot(monitorings-mass-message-mobile.png)" with timeout 5000ms
    - verifying given screenshot expectation
  - taking page screenshot
    - disabled all CSS animations
  - waiting for fonts to load...
  - fonts loaded
  - Expected an image 375px by 2507px, received 375px by 2454px. 215387 pixels (ratio 0.23 of all image pixels) are different.
  - waiting 100ms before taking screenshot
  - taking page screenshot
    - disabled all CSS animations
  - waiting for fonts to load...
  - fonts loaded
  - captured a stable screenshot
  - Expected an image 375px by 2507px, received 375px by 2454px. 215387 pixels (ratio 0.23 of all image pixels) are different.

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
        - text: Wyślij wiadomość masową dla
        - link "Monitoring sądów apelacyjnych" [ref=e92] [cursor=pointer]:
          - /url: /monitoringi/monitoring-sadow-apelacyjnych
      - generic [ref=e94]:
        - generic [ref=e95]:
          - group "Wiadomość" [ref=e97]:
            - group "Tagi odbiorców*" [ref=e100]:
              - generic [ref=e103]:
                - checkbox "test (0)" [ref=e104]
                - generic [ref=e105]: test (0)
            - generic [ref=e106]:
              - generic [ref=e107]: Temat*
              - textbox "Temat*" [ref=e108]: "Re: Wniosek o udostępnienie informacji publicznej"
            - generic [ref=e109]:
              - generic [ref=e110]: Treść w formacie HTML
              - application [ref=e111]:
                - generic [ref=e112]:
                  - generic [ref=e113]:
                    - menubar [ref=e115]:
                      - menuitem "Plik" [ref=e116] [cursor=pointer]
                      - menuitem "Edytuj" [ref=e118] [cursor=pointer]
                      - menuitem "Widok" [ref=e120] [cursor=pointer]
                      - menuitem "Wstaw" [ref=e122] [cursor=pointer]
                      - menuitem "Format" [ref=e124] [cursor=pointer]
                      - menuitem "Narzędzia" [ref=e126] [cursor=pointer]
                      - menuitem "Tabela" [ref=e128] [cursor=pointer]
                      - menuitem "Pomoc" [ref=e130] [cursor=pointer]
                    - group [ref=e132]:
                      - group [ref=e133]:
                        - toolbar [ref=e134]:
                          - button "Cofnij" [disabled] [ref=e135]
                          - button "Powtórz" [disabled] [ref=e139]
                        - toolbar [ref=e143]:
                          - button "Wysokość Linii" [ref=e144] [cursor=pointer]
                        - toolbar [ref=e151]:
                          - button "Odkryj lub ukryj dodatkowe elementy na pasku narzędzi" [ref=e152] [cursor=pointer]
                  - iframe [ref=e158]:
                    - generic "Obszar tekstu sformatowanego. Naciśnij ALT-0, aby uzyskać pomoc." [ref=f1e1]:
                      - paragraph [ref=f1e2]: "Prosimy o odpowiedź na adres {{EMAIL}}"
                      - paragraph [ref=f1e3]: "-----"
                - generic [ref=e159]:
                  - generic [ref=e160]:
                    - navigation [ref=e161]:
                      - button "p" [ref=e162]
                    - generic [ref=e163]:
                      - button "6 sł." [ref=e164] [cursor=pointer]
                      - link "Build with TinyMCE" [ref=e166]:
                        - /url: https://www.tiny.cloud/powered-by-tiny?utm_campaign=poweredby&utm_source=tiny&utm_medium=referral&utm_content=v7
                        - text: Build with
                  - generic "Naciśnij klawisze strzałek w górę i w dół, aby zmienić rozmiar edytora." [ref=e173]
              - generic [ref=e177]: "Użyj: {{EMAIL}} aby umieścić adres odpowiedzi, {{ADRESAT}} aby umieścić nazwę adressata."
          - group "Wiadomość cd." [ref=e179]:
            - generic [ref=e181]:
              - generic [ref=e182]: Cytat w formacie HTML
              - application [ref=e183]:
                - generic [ref=e184]:
                  - generic [ref=e185]:
                    - menubar [ref=e187]:
                      - menuitem "Plik" [ref=e188] [cursor=pointer]
                      - menuitem "Edytuj" [ref=e190] [cursor=pointer]
                      - menuitem "Widok" [ref=e192] [cursor=pointer]
                      - menuitem "Wstaw" [ref=e194] [cursor=pointer]
                      - menuitem "Format" [ref=e196] [cursor=pointer]
                      - menuitem "Narzędzia" [ref=e198] [cursor=pointer]
                      - menuitem "Tabela" [ref=e200] [cursor=pointer]
                      - menuitem "Pomoc" [ref=e202] [cursor=pointer]
                    - group [ref=e204]:
                      - group [ref=e205]:
                        - toolbar [ref=e206]:
                          - button "Cofnij" [disabled] [ref=e207]
                          - button "Powtórz" [disabled] [ref=e211]
                        - toolbar [ref=e215]:
                          - button "Wysokość Linii" [ref=e216] [cursor=pointer]
                        - toolbar [ref=e223]:
                          - button "Odkryj lub ukryj dodatkowe elementy na pasku narzędzi" [ref=e224] [cursor=pointer]
                  - iframe [ref=e230]:
                    - generic "Obszar tekstu sformatowanego. Naciśnij ALT-0, aby uzyskać pomoc." [ref=f2e1]:
                      - paragraph [ref=f2e2]: "W nawiązaniu do pisma z dnia 2017-08-11 z adresu {{EMAIL}}:"
                      - blockquote [ref=f2e3]:
                        - paragraph [ref=f2e4]: "Stowarzyszenie Sieć Obywatelska Watchdog Polska wnosi o udostępnienie poprzez przesłanie następujących informacji: - adresu strony internetowej, na której znajdują się orzeczenia dyscyplinarne, wydane wobec sędziów przez tutejszy Sąd, - rejestru umów, zawartych w imieniu Sądu od 1 stycznia 2017 r. do 31 lipca 2017 r., zawierającego informacje w zakresie co najmniej dat zawartych umów, przedmiotach umów, stronach umów, kwotach umów, - adresu strony internetowej, na której znajduje się dokumentacja przebiegu i efektów kontroli, przeprowadzonych w sądzie, oraz wystąpienia, stanowiska, wnioski i opinie podmiotów ją przeprowadzających, - kalendarz spotkań prezesa (prezes) sądu, które odbył (odbyła) w lipcu 2017 r., - skany orzeczeń, zapadłych w wyniku wniesienia środka odwoławczego od orzeczenia sądu niższej instancji, zapadłego z oskarżenia o czyn zabroniony art. 231 Kodeksu karnego - wydanych przez Sąd w 2017 r., - skany orzeczeń, zapadłych w wyniku wniesienia środka odwoławczego od orzeczenia sądu niższej instancji, zapadłego z oskarżenia o czyn zabroniony art. 212 Kodeksu karnego - wydanych przez Sąd w 2017 r. Stowarzyszenie wnosi o udostępnienie wskazanych informacji w formie elektronicznej, na adres e-mail {{EMAIL}}. Katarzyna Batko-Tołuć, Bartosz Wilk - członkowie zarządu, zgodnie z zasadami reprezentacji"
                      - paragraph [ref=f2e5]
                - generic [ref=e231]:
                  - generic [ref=e232]:
                    - navigation [ref=e233]:
                      - button "p" [ref=e234]
                    - generic [ref=e235]:
                      - button "185 sł." [ref=e236] [cursor=pointer]
                      - link "Build with TinyMCE" [ref=e238]:
                        - /url: https://www.tiny.cloud/powered-by-tiny?utm_campaign=poweredby&utm_source=tiny&utm_medium=referral&utm_content=v7
                        - text: Build with
                  - generic "Naciśnij klawisze strzałek w górę i w dół, aby zmienić rozmiar edytora." [ref=e245]
            - generic [ref=e249]:
              - generic [ref=e250]: Komentarz od redakcji
              - textbox "Komentarz od redakcji" [ref=e251]
        - generic [ref=e253]:
          - button "Zapisz szkic" [ref=e254] [cursor=pointer]
          - button "Wyślij wiadomość" [ref=e255] [cursor=pointer]
        - table [ref=e256]:
          - rowgroup [ref=e257]:
            - row [ref=e258]:
              - columnheader "Plik*" [ref=e259]
              - columnheader "Usuń" [ref=e260]
          - rowgroup [ref=e261]:
            - row [ref=e262]:
              - cell [ref=e263]:
                - button "Choose File" [ref=e265] [cursor=pointer]
              - cell [ref=e266]:
                - checkbox [ref=e267]
            - row [ref=e268]:
              - cell [ref=e269]:
                - button "Choose File" [ref=e271] [cursor=pointer]
              - cell [ref=e272]:
                - checkbox [ref=e273]
            - row [ref=e274]:
              - cell [ref=e275]:
                - button "Choose File" [ref=e277] [cursor=pointer]
              - cell [ref=e278]:
                - checkbox [ref=e279]
      - generic [ref=e280]:
        - generic [ref=e281]:
          - text: Sieć Obywatelska - Watchdog Polska ul. Szpitalna 5/5 00-031 Warszawa
          - paragraph [ref=e282]:
            - link "Klauzula RODO" [ref=e283] [cursor=pointer]:
              - /url: https://fedrowanie.siecobywatelska.pl/media_internal/tinycontent/uploads/11_KLAUZULA.-.dane.z.udip.na.stronach.www.i.w.mediach.spol.pdf
        - generic [ref=e284]:
          - text: "silnik:"
          - link "jawne.info.pl" [ref=e285] [cursor=pointer]:
            - /url: http://jawne.info.pl
          - text: "| v1.5.77.deps |"
          - link "GitHub" [ref=e286] [cursor=pointer]:
            - /url: https://github.com/watchdogpolska/feder
          - text: –
          - link "efd8a3b" [ref=e288] [cursor=pointer]:
            - /url: https://github.com/watchdogpolska/feder/compare/efd8a3b2...master
          - text: "|"
          - link "API" [ref=e289] [cursor=pointer]:
            - /url: /api/
        - generic [ref=e291]: Ta strona wykorzystuje cookies.
  - list [ref=e293]:
    - listitem [ref=e294]:
      - link "Ukryj »" [ref=e295] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e296]:
      - link "Toggle Theme" [ref=e297] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e300]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e301]
      - link "Historia /monitoringi/monitoring-sadow-apelacyjnych/~wiadomosc-masowa" [ref=e302] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e303]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e304]
      - link "Wersje Django 5.2.17" [ref=e305] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e306]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e307]
      - 'link "Czas CPU: 384.10ms (387.50ms)" [ref=e308] [cursor=pointer]':
        - /url: "#"
    - listitem [ref=e309]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e310]
      - link "Ustawienia" [ref=e311] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e312]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e313]
      - link "Nagłówki" [ref=e314] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e315]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e316]
      - link "Zapytania MassMessageView" [ref=e317] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e318]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e319]
      - link "SQL 9 queries in 3.70ms" [ref=e320] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e321]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e322]
      - link "Pliki statyczne 5 użytych plików" [ref=e323] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e324]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e325]
      - link "Templatki letters/_letter_reply_body.html" [ref=e326] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e327]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e328]
      - link "Alerty" [ref=e329] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e330]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e331]
      - link "Cache 2 wywołania w 0.18ms" [ref=e332] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e333]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e334]
      - link "Sygnały 88 odbiorców 15 sygnałów" [ref=e335] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e336]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e337]
      - link "Gmina" [ref=e338] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e339]:
      - checkbox "Enable for next and successive requests" [ref=e340]
      - generic [ref=e341]: Przechwycone przekierowania
    - listitem [ref=e342]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e343]
      - link "Profilowanie" [ref=e344] [cursor=pointer]:
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