# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: visual_baseline.spec.ts >> mobile (375px) >> letters-reply - visual baseline
- Location: tests/bs3-bs5_migration/visual_baseline.spec.ts:10:11

# Error details

```
Error: expect(page).toHaveScreenshot(expected) failed

  Expected an image 375px by 2208px, received 375px by 2251px. 196909 pixels (ratio 0.24 of all image pixels) are different.

  Snapshot: letters-reply-mobile.png

Call log:
  - Expect "toHaveScreenshot(letters-reply-mobile.png)" with timeout 5000ms
    - verifying given screenshot expectation
  - taking page screenshot
    - disabled all CSS animations
  - waiting for fonts to load...
  - fonts loaded
  - Expected an image 375px by 2208px, received 375px by 2251px. 196909 pixels (ratio 0.24 of all image pixels) are different.
  - waiting 100ms before taking screenshot
  - taking page screenshot
    - disabled all CSS animations
  - waiting for fonts to load...
  - fonts loaded
  - captured a stable screenshot
  - Expected an image 375px by 2208px, received 375px by 2251px. 196909 pixels (ratio 0.24 of all image pixels) are different.

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
        - listitem [ref=e70]:
          - link "Monitoring sądów apelacyjnych" [ref=e71] [cursor=pointer]:
            - /url: /monitoringi/monitoring-sadow-apelacyjnych
        - listitem [ref=e72]:
          - text: /
          - 'link "Monitoring sądów apelacyjnych #1" [ref=e73] [cursor=pointer]':
            - /url: /sprawy/monitoring-sadow-apelacyjnych-1
        - listitem [ref=e74]:
          - text: /
          - link "Wniosek o udostępnienie informacji publicznej" [ref=e75] [cursor=pointer]:
            - /url: /listy/7302
        - listitem [ref=e76]: / Odpowiedź
      - heading [level=2] [ref=e77]:
        - text: Odpowiedź na
        - link "Wniosek o udostępnienie informacji publicznej" [ref=e79] [cursor=pointer]:
          - /url: /listy/7302
      - generic [ref=e81]:
        - generic [ref=e82]:
          - group "Wiadomość" [ref=e84]:
            - generic [ref=e86]:
              - generic [ref=e87]: Temat*
              - textbox "Temat*" [ref=e88]: "Re: Wniosek o udostępnienie informacji publicznej"
            - generic [ref=e89]:
              - generic [ref=e90]: Treść w formacie HTML
              - application [ref=e91]:
                - generic [ref=e92]:
                  - generic [ref=e93]:
                    - menubar [ref=e95]:
                      - menuitem "Plik" [ref=e96] [cursor=pointer]
                      - menuitem "Edytuj" [ref=e98] [cursor=pointer]
                      - menuitem "Widok" [ref=e100] [cursor=pointer]
                      - menuitem "Wstaw" [ref=e102] [cursor=pointer]
                      - menuitem "Format" [ref=e104] [cursor=pointer]
                      - menuitem "Narzędzia" [ref=e106] [cursor=pointer]
                      - menuitem "Tabela" [ref=e108] [cursor=pointer]
                      - menuitem "Pomoc" [ref=e110] [cursor=pointer]
                    - group [ref=e112]:
                      - group [ref=e113]:
                        - toolbar [ref=e114]:
                          - button "Cofnij" [disabled] [ref=e115]
                          - button "Powtórz" [disabled] [ref=e119]
                        - toolbar [ref=e123]:
                          - button "Wysokość Linii" [ref=e124] [cursor=pointer]
                        - toolbar [ref=e131]:
                          - button "Odkryj lub ukryj dodatkowe elementy na pasku narzędzi" [ref=e132] [cursor=pointer]
                  - iframe [ref=e138]:
                    - generic "Obszar tekstu sformatowanego. Naciśnij ALT-0, aby uzyskać pomoc." [ref=f2e1]:
                      - paragraph [ref=f2e2]
                      - paragraph [ref=f2e3]: "Prosimy o odpowiedź na adres {{EMAIL}}"
                      - paragraph [ref=f2e4]: "-----"
                - generic [ref=e139]:
                  - generic [ref=e140]:
                    - navigation [ref=e141]:
                      - button "p" [ref=e142]
                    - generic [ref=e143]:
                      - button "6 sł." [ref=e144] [cursor=pointer]
                      - link "Build with TinyMCE" [ref=e146]:
                        - /url: https://www.tiny.cloud/powered-by-tiny?utm_campaign=poweredby&utm_source=tiny&utm_medium=referral&utm_content=v7
                        - text: Build with
                  - generic "Naciśnij klawisze strzałek w górę i w dół, aby zmienić rozmiar edytora." [ref=e153]
          - group "Wiadomość cd." [ref=e158]:
            - generic [ref=e160]:
              - generic [ref=e161]: "*"
              - iframe [ref=e162]:
                - generic [ref=f1e1]:
                  - paragraph [ref=f1e2]: "W nawiązaniu do pisma z dnia 2017-08-11 z adresu sprawa-2684@fedrowanie.siecobywatelska.pl:"
                  - blockquote [ref=f1e3]:
                    - paragraph [ref=f1e4]: "Stowarzyszenie Sieć Obywatelska Watchdog Polska wnosi o udostępnienie poprzez przesłanie następujących informacji: - adresu strony internetowej, na której znajdują się orzeczenia dyscyplinarne, wydane wobec sędziów przez tutejszy Sąd, - rejestru umów, zawartych w imieniu Sądu od 1 stycznia 2017 r. do 31 lipca 2017 r., zawierającego informacje w zakresie co najmniej dat zawartych umów, przedmiotach umów, stronach umów, kwotach umów, - adresu strony internetowej, na której znajduje się dokumentacja przebiegu i efektów kontroli, przeprowadzonych w sądzie, oraz wystąpienia, stanowiska, wnioski i opinie podmiotów ją przeprowadzających, - kalendarz spotkań prezesa (prezes) sądu, które odbył (odbyła) w lipcu 2017 r., - skany orzeczeń, zapadłych w wyniku wniesienia środka odwoławczego od orzeczenia sądu niższej instancji, zapadłego z oskarżenia o czyn zabroniony art. 231 Kodeksu karnego - wydanych przez Sąd w 2017 r., - skany orzeczeń, zapadłych w wyniku wniesienia środka odwoławczego od orzeczenia sądu niższej instancji, zapadłego z oskarżenia o czyn zabroniony art. 212 Kodeksu karnego - wydanych przez Sąd w 2017 r. Stowarzyszenie wnosi o udostępnienie wskazanych informacji w formie elektronicznej, na adres e-mail {{EMAIL}}. Katarzyna Batko-Tołuć, Bartosz Wilk - członkowie zarządu, zgodnie z zasadami reprezentacji"
                  - paragraph
            - generic [ref=e163]:
              - generic [ref=e164]: Komentarz od redakcji
              - textbox "Komentarz od redakcji" [ref=e165]
        - generic [ref=e167]:
          - button "Zapisz szkic" [ref=e168] [cursor=pointer]
          - button "Wyślij odpowiedź" [ref=e169] [cursor=pointer]
        - table [ref=e170]:
          - rowgroup [ref=e171]:
            - row [ref=e172]:
              - columnheader "Plik*" [ref=e173]
              - columnheader "Usuń" [ref=e174]
          - rowgroup [ref=e175]:
            - row [ref=e176]:
              - cell [ref=e177]:
                - button "Choose File" [ref=e179] [cursor=pointer]
              - cell [ref=e180]:
                - checkbox [ref=e181]
            - row [ref=e182]:
              - cell [ref=e183]:
                - button "Choose File" [ref=e185] [cursor=pointer]
              - cell [ref=e186]:
                - checkbox [ref=e187]
            - row [ref=e188]:
              - cell [ref=e189]:
                - button "Choose File" [ref=e191] [cursor=pointer]
              - cell [ref=e192]:
                - checkbox [ref=e193]
      - generic [ref=e194]:
        - generic [ref=e195]:
          - text: Sieć Obywatelska - Watchdog Polska ul. Szpitalna 5/5 00-031 Warszawa
          - paragraph [ref=e196]:
            - link "Klauzula RODO" [ref=e197] [cursor=pointer]:
              - /url: https://fedrowanie.siecobywatelska.pl/media_internal/tinycontent/uploads/11_KLAUZULA.-.dane.z.udip.na.stronach.www.i.w.mediach.spol.pdf
        - generic [ref=e198]:
          - text: "silnik:"
          - link "jawne.info.pl" [ref=e199] [cursor=pointer]:
            - /url: http://jawne.info.pl
          - text: "| v1.5.77.deps |"
          - link "GitHub" [ref=e200] [cursor=pointer]:
            - /url: https://github.com/watchdogpolska/feder
          - text: –
          - link "efd8a3b" [ref=e202] [cursor=pointer]:
            - /url: https://github.com/watchdogpolska/feder/compare/efd8a3b2...master
          - text: "|"
          - link "API" [ref=e203] [cursor=pointer]:
            - /url: /api/
        - generic [ref=e205]: Ta strona wykorzystuje cookies.
  - list [ref=e207]:
    - listitem [ref=e208]:
      - link "Ukryj »" [ref=e209] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e210]:
      - link "Toggle Theme" [ref=e211] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e214]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e215]
      - link "Historia /listy/7302/~odpowiedz" [ref=e216] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e217]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e218]
      - link "Wersje Django 5.2.17" [ref=e219] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e220]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e221]
      - 'link "Czas CPU: 226.62ms (219.47ms)" [ref=e222] [cursor=pointer]':
        - /url: "#"
    - listitem [ref=e223]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e224]
      - link "Ustawienia" [ref=e225] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e226]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e227]
      - link "Nagłówki" [ref=e228] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e229]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e230]
      - link "Zapytania LetterReplyView" [ref=e231] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e232]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e233]
      - link "SQL 6 queries in 2.10ms" [ref=e234] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e235]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e236]
      - link "Pliki statyczne 5 użytych plików" [ref=e237] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e238]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e239]
      - link "Templatki letters/_letter_reply_body.html" [ref=e240] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e241]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e242]
      - link "Alerty" [ref=e243] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e244]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e245]
      - link "Cache 2 wywołania w 0.13ms" [ref=e246] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e247]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e248]
      - link "Sygnały 88 odbiorców 15 sygnałów" [ref=e249] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e250]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e251]
      - link "Gmina" [ref=e252] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e253]:
      - checkbox "Enable for next and successive requests" [ref=e254]
      - generic [ref=e255]: Przechwycone przekierowania
    - listitem [ref=e256]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e257]
      - link "Profilowanie" [ref=e258] [cursor=pointer]:
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