# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: visual_baseline.spec.ts >> desktop (2560px) >> letters-reply - visual baseline
- Location: tests/bs3-bs5_migration/visual_baseline.spec.ts:10:11

# Error details

```
Error: expect(page).toHaveScreenshot(expected) failed

  396561 pixels (ratio 0.11 of all image pixels) are different.

  Snapshot: letters-reply-desktop.png

Call log:
  - Expect "toHaveScreenshot(letters-reply-desktop.png)" with timeout 5000ms
    - verifying given screenshot expectation
  - taking page screenshot
    - disabled all CSS animations
  - waiting for fonts to load...
  - fonts loaded
  - 396561 pixels (ratio 0.11 of all image pixels) are different.
  - waiting 100ms before taking screenshot
  - taking page screenshot
    - disabled all CSS animations
  - waiting for fonts to load...
  - fonts loaded
  - captured a stable screenshot
  - 396561 pixels (ratio 0.11 of all image pixels) are different.

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
        - listitem [ref=e64]:
          - text: /
          - 'link "Monitoring sądów apelacyjnych #1" [ref=e65] [cursor=pointer]':
            - /url: /sprawy/monitoring-sadow-apelacyjnych-1
        - listitem [ref=e66]:
          - text: /
          - link "Wniosek o udostępnienie informacji publicznej" [ref=e67] [cursor=pointer]:
            - /url: /listy/7302
        - listitem [ref=e68]: / Odpowiedź
      - heading [level=2] [ref=e69]:
        - text: Odpowiedź na
        - link "Wniosek o udostępnienie informacji publicznej" [ref=e71] [cursor=pointer]:
          - /url: /listy/7302
      - generic [ref=e73]:
        - generic [ref=e74]:
          - group "Wiadomość" [ref=e76]:
            - generic [ref=e78]:
              - generic [ref=e79]: Temat*
              - textbox "Temat*" [ref=e80]: "Re: Wniosek o udostępnienie informacji publicznej"
            - generic [ref=e81]:
              - generic [ref=e82]: Treść w formacie HTML
              - application [ref=e83]:
                - generic [ref=e84]:
                  - generic [ref=e85]:
                    - menubar [ref=e87]:
                      - menuitem "Plik" [ref=e88] [cursor=pointer]
                      - menuitem "Edytuj" [ref=e90] [cursor=pointer]
                      - menuitem "Widok" [ref=e92] [cursor=pointer]
                      - menuitem "Wstaw" [ref=e94] [cursor=pointer]
                      - menuitem "Format" [ref=e96] [cursor=pointer]
                      - menuitem "Narzędzia" [ref=e98] [cursor=pointer]
                      - menuitem "Tabela" [ref=e100] [cursor=pointer]
                      - menuitem "Pomoc" [ref=e102] [cursor=pointer]
                    - group [ref=e104]:
                      - group [ref=e105]:
                        - toolbar [ref=e106]:
                          - button "Cofnij" [disabled] [ref=e107]
                          - button "Powtórz" [disabled] [ref=e111]
                        - toolbar [ref=e115]:
                          - button "Wysokość Linii" [ref=e116] [cursor=pointer]
                        - toolbar [ref=e123]:
                          - button "Pogrubienie" [ref=e124] [cursor=pointer]
                          - button "Kursywa" [ref=e128] [cursor=pointer]
                          - button "Kolor tła Czarny" [ref=e132]
                        - toolbar [ref=e140]:
                          - button "Wyrównaj do lewej" [ref=e141] [cursor=pointer]
                          - button "Wyrównaj do środka" [ref=e145] [cursor=pointer]
                          - button "Wyrównaj do prawej" [ref=e149] [cursor=pointer]
                          - button "Wyjustuj" [ref=e153] [cursor=pointer]
                        - toolbar [ref=e157]:
                          - button "Lista wypunktowana" [ref=e158] [cursor=pointer]
                          - button "Lista numerowana" [ref=e162] [cursor=pointer]
                          - button "Zmniejsz wcięcie" [disabled] [ref=e166]
                          - button "Zwiększ wcięcie" [ref=e170] [cursor=pointer]
                        - toolbar [ref=e174]:
                          - button "Znak Specjalny" [ref=e175] [cursor=pointer]
                        - toolbar [ref=e179]:
                          - button "Wyczyść formatowanie" [ref=e180] [cursor=pointer]
                        - toolbar [ref=e184]:
                          - button "Pomoc" [ref=e185] [cursor=pointer]
                  - iframe [ref=e194]:
                    - generic "Obszar tekstu sformatowanego. Naciśnij ALT-0, aby uzyskać pomoc." [ref=f2e1]:
                      - paragraph [ref=f2e2]
                      - paragraph [ref=f2e3]: "Prosimy o odpowiedź na adres {{EMAIL}}"
                      - paragraph [ref=f2e4]: "-----"
                - generic [ref=e195]:
                  - generic [ref=e196]:
                    - navigation [ref=e197]:
                      - button "p" [ref=e198]
                    - generic [ref=e199]: Naciśnij Alt+0, aby uzyskać pomoc
                    - generic [ref=e200]:
                      - button "6 sł." [ref=e201] [cursor=pointer]
                      - link "Build with TinyMCE" [ref=e203]:
                        - /url: https://www.tiny.cloud/powered-by-tiny?utm_campaign=poweredby&utm_source=tiny&utm_medium=referral&utm_content=v7
                        - text: Build with
                  - generic "Naciśnij klawisze strzałek w górę i w dół, aby zmienić rozmiar edytora." [ref=e210]
          - group "Wiadomość cd." [ref=e215]:
            - generic [ref=e217]:
              - generic [ref=e218]: "*"
              - iframe [ref=e219]:
                - generic [ref=f1e1]:
                  - paragraph [ref=f1e2]: "W nawiązaniu do pisma z dnia 2017-08-11 z adresu sprawa-2684@fedrowanie.siecobywatelska.pl:"
                  - blockquote [ref=f1e3]:
                    - paragraph [ref=f1e4]: "Stowarzyszenie Sieć Obywatelska Watchdog Polska wnosi o udostępnienie poprzez przesłanie następujących informacji: - adresu strony internetowej, na której znajdują się orzeczenia dyscyplinarne, wydane wobec sędziów przez tutejszy Sąd, - rejestru umów, zawartych w imieniu Sądu od 1 stycznia 2017 r. do 31 lipca 2017 r., zawierającego informacje w zakresie co najmniej dat zawartych umów, przedmiotach umów, stronach umów, kwotach umów, - adresu strony internetowej, na której znajduje się dokumentacja przebiegu i efektów kontroli, przeprowadzonych w sądzie, oraz wystąpienia, stanowiska, wnioski i opinie podmiotów ją przeprowadzających, - kalendarz spotkań prezesa (prezes) sądu, które odbył (odbyła) w lipcu 2017 r., - skany orzeczeń, zapadłych w wyniku wniesienia środka odwoławczego od orzeczenia sądu niższej instancji, zapadłego z oskarżenia o czyn zabroniony art. 231 Kodeksu karnego - wydanych przez Sąd w 2017 r., - skany orzeczeń, zapadłych w wyniku wniesienia środka odwoławczego od orzeczenia sądu niższej instancji, zapadłego z oskarżenia o czyn zabroniony art. 212 Kodeksu karnego - wydanych przez Sąd w 2017 r. Stowarzyszenie wnosi o udostępnienie wskazanych informacji w formie elektronicznej, na adres e-mail {{EMAIL}}. Katarzyna Batko-Tołuć, Bartosz Wilk - członkowie zarządu, zgodnie z zasadami reprezentacji"
                  - paragraph
            - generic [ref=e220]:
              - generic [ref=e221]: Komentarz od redakcji
              - textbox "Komentarz od redakcji" [ref=e222]
        - generic [ref=e224]:
          - button "Zapisz szkic" [ref=e225] [cursor=pointer]
          - button "Wyślij odpowiedź" [ref=e226] [cursor=pointer]
        - table [ref=e227]:
          - rowgroup [ref=e228]:
            - row [ref=e229]:
              - columnheader "Plik*" [ref=e230]
              - columnheader "Usuń" [ref=e231]
          - rowgroup [ref=e232]:
            - row [ref=e233]:
              - cell [ref=e234]:
                - button "Choose File" [ref=e236] [cursor=pointer]
              - cell [ref=e237]:
                - checkbox [ref=e238]
            - row [ref=e239]:
              - cell [ref=e240]:
                - button "Choose File" [ref=e242] [cursor=pointer]
              - cell [ref=e243]:
                - checkbox [ref=e244]
            - row [ref=e245]:
              - cell [ref=e246]:
                - button "Choose File" [ref=e248] [cursor=pointer]
              - cell [ref=e249]:
                - checkbox [ref=e250]
      - generic [ref=e251]:
        - generic [ref=e252]:
          - text: Sieć Obywatelska - Watchdog Polska ul. Szpitalna 5/5 00-031 Warszawa
          - paragraph [ref=e253]:
            - link "Klauzula RODO" [ref=e254] [cursor=pointer]:
              - /url: https://fedrowanie.siecobywatelska.pl/media_internal/tinycontent/uploads/11_KLAUZULA.-.dane.z.udip.na.stronach.www.i.w.mediach.spol.pdf
        - generic [ref=e255]:
          - text: "silnik:"
          - link "jawne.info.pl" [ref=e256] [cursor=pointer]:
            - /url: http://jawne.info.pl
          - text: "| v1.5.77.deps |"
          - link "GitHub" [ref=e257] [cursor=pointer]:
            - /url: https://github.com/watchdogpolska/feder
          - text: –
          - link "efd8a3b" [ref=e259] [cursor=pointer]:
            - /url: https://github.com/watchdogpolska/feder/compare/efd8a3b2...master
          - text: "|"
          - link "API" [ref=e260] [cursor=pointer]:
            - /url: /api/
        - generic [ref=e262]: Ta strona wykorzystuje cookies.
  - list [ref=e264]:
    - listitem [ref=e265]:
      - link "Ukryj »" [ref=e266] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e267]:
      - link "Toggle Theme" [ref=e268] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e271]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e272]
      - link "Historia /listy/7302/~odpowiedz" [ref=e273] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e274]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e275]
      - link "Wersje Django 5.2.17" [ref=e276] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e277]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e278]
      - 'link "Czas CPU: 199.25ms (201.97ms)" [ref=e279] [cursor=pointer]':
        - /url: "#"
    - listitem [ref=e280]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e281]
      - link "Ustawienia" [ref=e282] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e283]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e284]
      - link "Nagłówki" [ref=e285] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e286]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e287]
      - link "Zapytania LetterReplyView" [ref=e288] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e289]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e290]
      - link "SQL 6 queries in 2.57ms" [ref=e291] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e292]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e293]
      - link "Pliki statyczne 5 użytych plików" [ref=e294] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e295]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e296]
      - link "Templatki letters/_letter_reply_body.html" [ref=e297] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e298]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e299]
      - link "Alerty" [ref=e300] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e301]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e302]
      - link "Cache 2 wywołania w 0.18ms" [ref=e303] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e304]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e305]
      - link "Sygnały 88 odbiorców 15 sygnałów" [ref=e306] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e307]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e308]
      - link "Gmina" [ref=e309] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e310]:
      - checkbox "Enable for next and successive requests" [ref=e311]
      - generic [ref=e312]: Przechwycone przekierowania
    - listitem [ref=e313]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e314]
      - link "Profilowanie" [ref=e315] [cursor=pointer]:
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