# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: visual_baseline.spec.ts >> desktop (2560px) >> monitorings-mass-message - visual baseline
- Location: tests/bs3-bs5_migration/visual_baseline.spec.ts:10:11

# Error details

```
Error: expect(page).toHaveScreenshot(expected) failed

  Expected an image 2560px by 1507px, received 2560px by 1501px. 419197 pixels (ratio 0.11 of all image pixels) are different.

  Snapshot: monitorings-mass-message-desktop.png

Call log:
  - Expect "toHaveScreenshot(monitorings-mass-message-desktop.png)" with timeout 5000ms
    - verifying given screenshot expectation
  - taking page screenshot
    - disabled all CSS animations
  - waiting for fonts to load...
  - fonts loaded
  - Expected an image 2560px by 1507px, received 2560px by 1501px. 419197 pixels (ratio 0.11 of all image pixels) are different.
  - waiting 100ms before taking screenshot
  - taking page screenshot
    - disabled all CSS animations
  - waiting for fonts to load...
  - fonts loaded
  - captured a stable screenshot
  - Expected an image 2560px by 1507px, received 2560px by 1501px. 419197 pixels (ratio 0.11 of all image pixels) are different.

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
        - text: Wyślij wiadomość masową dla
        - link "Monitoring sądów apelacyjnych" [ref=e84] [cursor=pointer]:
          - /url: /monitoringi/monitoring-sadow-apelacyjnych
      - generic [ref=e86]:
        - generic [ref=e87]:
          - group "Wiadomość" [ref=e89]:
            - group "Tagi odbiorców*" [ref=e92]:
              - generic [ref=e95]:
                - checkbox "test (0)" [ref=e96]
                - generic [ref=e97]: test (0)
            - generic [ref=e98]:
              - generic [ref=e99]: Temat*
              - textbox "Temat*" [ref=e100]: "Re: Wniosek o udostępnienie informacji publicznej"
            - generic [ref=e101]:
              - generic [ref=e102]: Treść w formacie HTML
              - application [ref=e103]:
                - generic [ref=e104]:
                  - generic [ref=e105]:
                    - menubar [ref=e107]:
                      - menuitem "Plik" [ref=e108] [cursor=pointer]
                      - menuitem "Edytuj" [ref=e110] [cursor=pointer]
                      - menuitem "Widok" [ref=e112] [cursor=pointer]
                      - menuitem "Wstaw" [ref=e114] [cursor=pointer]
                      - menuitem "Format" [ref=e116] [cursor=pointer]
                      - menuitem "Narzędzia" [ref=e118] [cursor=pointer]
                      - menuitem "Tabela" [ref=e120] [cursor=pointer]
                      - menuitem "Pomoc" [ref=e122] [cursor=pointer]
                    - group [ref=e124]:
                      - group [ref=e125]:
                        - toolbar [ref=e126]:
                          - button "Cofnij" [disabled] [ref=e127]
                          - button "Powtórz" [disabled] [ref=e131]
                        - toolbar [ref=e135]:
                          - button "Wysokość Linii" [ref=e136] [cursor=pointer]
                        - toolbar [ref=e143]:
                          - button "Pogrubienie" [ref=e144] [cursor=pointer]
                          - button "Kursywa" [ref=e148] [cursor=pointer]
                          - button "Kolor tła Czarny" [ref=e152]
                        - toolbar [ref=e160]:
                          - button "Wyrównaj do lewej" [ref=e161] [cursor=pointer]
                          - button "Wyrównaj do środka" [ref=e165] [cursor=pointer]
                          - button "Wyrównaj do prawej" [ref=e169] [cursor=pointer]
                          - button "Wyjustuj" [ref=e173] [cursor=pointer]
                        - toolbar [ref=e177]:
                          - button "Lista wypunktowana" [ref=e178] [cursor=pointer]
                          - button "Lista numerowana" [ref=e182] [cursor=pointer]
                          - button "Zmniejsz wcięcie" [disabled] [ref=e186]
                          - button "Zwiększ wcięcie" [ref=e190] [cursor=pointer]
                        - toolbar [ref=e194]:
                          - button "Znak Specjalny" [ref=e195] [cursor=pointer]
                        - toolbar [ref=e199]:
                          - button "Wyczyść formatowanie" [ref=e200] [cursor=pointer]
                        - toolbar [ref=e204]:
                          - button "Pomoc" [ref=e205] [cursor=pointer]
                  - iframe [ref=e214]:
                    - generic "Obszar tekstu sformatowanego. Naciśnij ALT-0, aby uzyskać pomoc." [ref=f1e1]:
                      - paragraph [ref=f1e2]: "Prosimy o odpowiedź na adres {{EMAIL}}"
                      - paragraph [ref=f1e3]: "-----"
                - generic [ref=e215]:
                  - generic [ref=e216]:
                    - navigation [ref=e217]:
                      - button "p" [ref=e218]
                    - generic [ref=e219]: Naciśnij Alt+0, aby uzyskać pomoc
                    - generic [ref=e220]:
                      - button "6 sł." [ref=e221] [cursor=pointer]
                      - link "Build with TinyMCE" [ref=e223]:
                        - /url: https://www.tiny.cloud/powered-by-tiny?utm_campaign=poweredby&utm_source=tiny&utm_medium=referral&utm_content=v7
                        - text: Build with
                  - generic "Naciśnij klawisze strzałek w górę i w dół, aby zmienić rozmiar edytora." [ref=e230]
              - generic [ref=e234]: "Użyj: {{EMAIL}} aby umieścić adres odpowiedzi, {{ADRESAT}} aby umieścić nazwę adressata."
          - group "Wiadomość cd." [ref=e236]:
            - generic [ref=e238]:
              - generic [ref=e239]: Cytat w formacie HTML
              - application [ref=e240]:
                - generic [ref=e241]:
                  - generic [ref=e242]:
                    - menubar [ref=e244]:
                      - menuitem "Plik" [ref=e245] [cursor=pointer]
                      - menuitem "Edytuj" [ref=e247] [cursor=pointer]
                      - menuitem "Widok" [ref=e249] [cursor=pointer]
                      - menuitem "Wstaw" [ref=e251] [cursor=pointer]
                      - menuitem "Format" [ref=e253] [cursor=pointer]
                      - menuitem "Narzędzia" [ref=e255] [cursor=pointer]
                      - menuitem "Tabela" [ref=e257] [cursor=pointer]
                      - menuitem "Pomoc" [ref=e259] [cursor=pointer]
                    - group [ref=e261]:
                      - group [ref=e262]:
                        - toolbar [ref=e263]:
                          - button "Cofnij" [disabled] [ref=e264]
                          - button "Powtórz" [disabled] [ref=e268]
                        - toolbar [ref=e272]:
                          - button "Wysokość Linii" [ref=e273] [cursor=pointer]
                        - toolbar [ref=e280]:
                          - button "Pogrubienie" [ref=e281] [cursor=pointer]
                          - button "Kursywa" [ref=e285] [cursor=pointer]
                          - button "Kolor tła Czarny" [ref=e289]
                        - toolbar [ref=e297]:
                          - button "Wyrównaj do lewej" [ref=e298] [cursor=pointer]
                          - button "Wyrównaj do środka" [ref=e302] [cursor=pointer]
                          - button "Wyrównaj do prawej" [ref=e306] [cursor=pointer]
                          - button "Wyjustuj" [ref=e310] [cursor=pointer]
                        - toolbar [ref=e314]:
                          - button "Lista wypunktowana" [ref=e315] [cursor=pointer]
                          - button "Lista numerowana" [ref=e319] [cursor=pointer]
                          - button "Zmniejsz wcięcie" [disabled] [ref=e323]
                          - button "Zwiększ wcięcie" [ref=e327] [cursor=pointer]
                        - toolbar [ref=e331]:
                          - button "Znak Specjalny" [ref=e332] [cursor=pointer]
                        - toolbar [ref=e336]:
                          - button "Wyczyść formatowanie" [ref=e337] [cursor=pointer]
                        - toolbar [ref=e341]:
                          - button "Pomoc" [ref=e342] [cursor=pointer]
                  - iframe [ref=e351]:
                    - generic "Obszar tekstu sformatowanego. Naciśnij ALT-0, aby uzyskać pomoc." [ref=f2e1]:
                      - paragraph [ref=f2e2]: "W nawiązaniu do pisma z dnia 2017-08-11 z adresu {{EMAIL}}:"
                      - blockquote [ref=f2e3]:
                        - paragraph [ref=f2e4]: "Stowarzyszenie Sieć Obywatelska Watchdog Polska wnosi o udostępnienie poprzez przesłanie następujących informacji: - adresu strony internetowej, na której znajdują się orzeczenia dyscyplinarne, wydane wobec sędziów przez tutejszy Sąd, - rejestru umów, zawartych w imieniu Sądu od 1 stycznia 2017 r. do 31 lipca 2017 r., zawierającego informacje w zakresie co najmniej dat zawartych umów, przedmiotach umów, stronach umów, kwotach umów, - adresu strony internetowej, na której znajduje się dokumentacja przebiegu i efektów kontroli, przeprowadzonych w sądzie, oraz wystąpienia, stanowiska, wnioski i opinie podmiotów ją przeprowadzających, - kalendarz spotkań prezesa (prezes) sądu, które odbył (odbyła) w lipcu 2017 r., - skany orzeczeń, zapadłych w wyniku wniesienia środka odwoławczego od orzeczenia sądu niższej instancji, zapadłego z oskarżenia o czyn zabroniony art. 231 Kodeksu karnego - wydanych przez Sąd w 2017 r., - skany orzeczeń, zapadłych w wyniku wniesienia środka odwoławczego od orzeczenia sądu niższej instancji, zapadłego z oskarżenia o czyn zabroniony art. 212 Kodeksu karnego - wydanych przez Sąd w 2017 r. Stowarzyszenie wnosi o udostępnienie wskazanych informacji w formie elektronicznej, na adres e-mail {{EMAIL}}. Katarzyna Batko-Tołuć, Bartosz Wilk - członkowie zarządu, zgodnie z zasadami reprezentacji"
                      - paragraph [ref=f2e5]
                - generic [ref=e352]:
                  - generic [ref=e353]:
                    - navigation [ref=e354]:
                      - button "p" [ref=e355]
                    - generic [ref=e356]: Naciśnij Alt+0, aby uzyskać pomoc
                    - generic [ref=e357]:
                      - button "185 sł." [ref=e358] [cursor=pointer]
                      - link "Build with TinyMCE" [ref=e360]:
                        - /url: https://www.tiny.cloud/powered-by-tiny?utm_campaign=poweredby&utm_source=tiny&utm_medium=referral&utm_content=v7
                        - text: Build with
                  - generic "Naciśnij klawisze strzałek w górę i w dół, aby zmienić rozmiar edytora." [ref=e367]
            - generic [ref=e371]:
              - generic [ref=e372]: Komentarz od redakcji
              - textbox "Komentarz od redakcji" [ref=e373]
        - generic [ref=e375]:
          - button "Zapisz szkic" [ref=e376] [cursor=pointer]
          - button "Wyślij wiadomość" [ref=e377] [cursor=pointer]
        - table [ref=e378]:
          - rowgroup [ref=e379]:
            - row [ref=e380]:
              - columnheader "Plik*" [ref=e381]
              - columnheader "Usuń" [ref=e382]
          - rowgroup [ref=e383]:
            - row [ref=e384]:
              - cell [ref=e385]:
                - button "Choose File" [ref=e387] [cursor=pointer]
              - cell [ref=e388]:
                - checkbox [ref=e389]
            - row [ref=e390]:
              - cell [ref=e391]:
                - button "Choose File" [ref=e393] [cursor=pointer]
              - cell [ref=e394]:
                - checkbox [ref=e395]
            - row [ref=e396]:
              - cell [ref=e397]:
                - button "Choose File" [ref=e399] [cursor=pointer]
              - cell [ref=e400]:
                - checkbox [ref=e401]
      - generic [ref=e402]:
        - generic [ref=e403]:
          - text: Sieć Obywatelska - Watchdog Polska ul. Szpitalna 5/5 00-031 Warszawa
          - paragraph [ref=e404]:
            - link "Klauzula RODO" [ref=e405] [cursor=pointer]:
              - /url: https://fedrowanie.siecobywatelska.pl/media_internal/tinycontent/uploads/11_KLAUZULA.-.dane.z.udip.na.stronach.www.i.w.mediach.spol.pdf
        - generic [ref=e406]:
          - text: "silnik:"
          - link "jawne.info.pl" [ref=e407] [cursor=pointer]:
            - /url: http://jawne.info.pl
          - text: "| v1.5.77.deps |"
          - link "GitHub" [ref=e408] [cursor=pointer]:
            - /url: https://github.com/watchdogpolska/feder
          - text: –
          - link "efd8a3b" [ref=e410] [cursor=pointer]:
            - /url: https://github.com/watchdogpolska/feder/compare/efd8a3b2...master
          - text: "|"
          - link "API" [ref=e411] [cursor=pointer]:
            - /url: /api/
        - generic [ref=e413]: Ta strona wykorzystuje cookies.
  - list [ref=e415]:
    - listitem [ref=e416]:
      - link "Ukryj »" [ref=e417] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e418]:
      - link "Toggle Theme" [ref=e419] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e422]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e423]
      - link "Historia /monitoringi/monitoring-sadow-apelacyjnych/~wiadomosc-masowa" [ref=e424] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e425]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e426]
      - link "Wersje Django 5.2.17" [ref=e427] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e428]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e429]
      - 'link "Czas CPU: 378.47ms (381.62ms)" [ref=e430] [cursor=pointer]':
        - /url: "#"
    - listitem [ref=e431]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e432]
      - link "Ustawienia" [ref=e433] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e434]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e435]
      - link "Nagłówki" [ref=e436] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e437]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e438]
      - link "Zapytania MassMessageView" [ref=e439] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e440]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e441]
      - link "SQL 9 queries in 3.72ms" [ref=e442] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e443]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e444]
      - link "Pliki statyczne 5 użytych plików" [ref=e445] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e446]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e447]
      - link "Templatki letters/_letter_reply_body.html" [ref=e448] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e449]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e450]
      - link "Alerty" [ref=e451] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e452]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e453]
      - link "Cache 2 wywołania w 0.14ms" [ref=e454] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e455]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e456]
      - link "Sygnały 88 odbiorców 15 sygnałów" [ref=e457] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e458]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e459]
      - link "Gmina" [ref=e460] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e461]:
      - checkbox "Enable for next and successive requests" [ref=e462]
      - generic [ref=e463]: Przechwycone przekierowania
    - listitem [ref=e464]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e465]
      - link "Profilowanie" [ref=e466] [cursor=pointer]:
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