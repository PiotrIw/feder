# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: visual_baseline.spec.ts >> desktop (2560px) >> cases-details - visual baseline
- Location: tests/bs3-bs5_migration/visual_baseline.spec.ts:10:11

# Error details

```
Error: expect(page).toHaveScreenshot(expected) failed

  Expected an image 2560px by 2995px, received 2560px by 3167px. 545260 pixels (ratio 0.07 of all image pixels) are different.

  Snapshot: cases-details-desktop.png

Call log:
  - Expect "toHaveScreenshot(cases-details-desktop.png)" with timeout 5000ms
    - verifying given screenshot expectation
  - taking page screenshot
    - disabled all CSS animations
  - waiting for fonts to load...
  - fonts loaded
  - Expected an image 2560px by 2995px, received 2560px by 3167px. 545260 pixels (ratio 0.07 of all image pixels) are different.
  - waiting 100ms before taking screenshot
  - taking page screenshot
    - disabled all CSS animations
  - waiting for fonts to load...
  - fonts loaded
  - captured a stable screenshot
  - Expected an image 2560px by 2995px, received 2560px by 3167px. 545260 pixels (ratio 0.07 of all image pixels) are different.

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
        - listitem [ref=e64]: "/ Monitoring sądów apelacyjnych #1"
      - generic [ref=e66]:
        - link "Edytuj" [ref=e67] [cursor=pointer]:
          - /url: /sprawy/monitoring-sadow-apelacyjnych-1/~edytuj
        - link "Usuń" [ref=e68] [cursor=pointer]:
          - /url: /sprawy/monitoring-sadow-apelacyjnych-1/~usun
        - link "Zobacz dzienniki" [ref=e69] [cursor=pointer]:
          - /url: /listy/logi/spraw-2684
        - button "Dodaj przesyłkę pocztową" [ref=e71] [cursor=pointer]
        - link "Dodaj list" [ref=e73] [cursor=pointer]:
          - /url: /listy/~utworz-2684
      - 'heading "Monitoring sądów apelacyjnych #1" [level=1] [ref=e76]'
      - generic [ref=e78]:
        - generic [ref=e79]:
          - table [ref=e81]:
            - rowgroup [ref=e82]:
              - row [ref=e83]:
                - cell [ref=e84]:
                  - paragraph [ref=e85]: "Instytucja:"
                - cell [ref=e87]:
                  - paragraph [ref=e88]:
                    - link "Sąd Apelacyjny w Białymstoku" [ref=e89] [cursor=pointer]:
                      - /url: /instytucje/sad-apelacyjny-w-bialymstoku
              - row [ref=e90]:
                - cell [ref=e91]:
                  - paragraph [ref=e92]: "Email instytucji:"
                - cell [ref=e94]:
                  - paragraph [ref=e95]: boi@bialystok.sa.gov.pl
              - row [ref=e96]:
                - cell [ref=e97]:
                  - paragraph [ref=e98]: "Monitoring:"
                - cell [ref=e100]:
                  - paragraph [ref=e101]:
                    - link "Monitoring sądów apelacyjnych" [ref=e102] [cursor=pointer]:
                      - /url: /monitoringi/monitoring-sadow-apelacyjnych
              - row [ref=e103]:
                - cell [ref=e104]:
                  - paragraph [ref=e105]: "Email sprawy:"
                - cell [ref=e107]:
                  - paragraph [ref=e108]: sprawa-2684@fedrowanie.siecobywatelska.pl
              - row [ref=e109]:
                - cell [ref=e110]:
                  - paragraph [ref=e111]: "Liczba listów:"
                - cell [ref=e113]:
                  - paragraph [ref=e114]: "4"
              - row [ref=e115]:
                - cell [ref=e116]:
                  - paragraph [ref=e117]: "Liczba spamu:"
                - cell [ref=e119]:
                  - paragraph [ref=e120]: "0"
              - row [ref=e121]:
                - cell [ref=e122]:
                  - paragraph [ref=e123]: "Status pierwszego wniosku:"
                - cell [ref=e125]:
                  - paragraph
              - row [ref=e126]:
                - cell [ref=e127]:
                  - paragraph [ref=e128]: "Status ostatniego wniosku:"
                - cell [ref=e130]:
                  - paragraph
              - row [ref=e131]:
                - cell [ref=e132]:
                  - paragraph [ref=e133]: "Otrzymano potwierdzenie:"
                - cell [ref=e135]:
                  - paragraph [ref=e136]
              - row [ref=e138]:
                - cell [ref=e139]:
                  - paragraph [ref=e140]: "Otrzymano odpowiedź:"
                - cell [ref=e142]:
                  - paragraph [ref=e143]
              - row [ref=e145]:
                - cell [ref=e146]:
                  - paragraph [ref=e147]: "Poddany kwarantannie:"
                - cell [ref=e149]:
                  - paragraph [ref=e150]
          - heading "Znormalizowana odpowiedź" [level=4] [ref=e153]
        - generic [ref=e154]:
          - heading "Treść" [level=4] [ref=e155]
          - list [ref=e157]:
            - listitem [ref=e158]:
              - generic [ref=e160]:
                - heading [level=3] [ref=e161]:
                  - link "Wniosek o udostępnienie informacji publicznej" [ref=e163] [cursor=pointer]:
                    - /url: /listy/7302
                  - generic [ref=e164]:
                    - text: przez
                    - link "adobrawy" [ref=e165] [cursor=pointer]:
                      - /url: /uzytkownik/adobrawy/
                    - time [ref=e166]: 11 sierpnia 2017 02:48
                - generic [ref=e167]:
                  - paragraph [ref=e168]: "Stowarzyszenie Sieć Obywatelska Watchdog Polska wnosi o udostępnienie poprzez przesłanie następujących informacji:"
                  - paragraph [ref=e169]: "- adresu strony internetowej, na której znajdują się orzeczenia dyscyplinarne, wydane wobec sędziów przez tutejszy Sąd,- rejestru umów, zawartych w imieniu Sądu od 1 stycznia 2017 r. do 31 lipca 2017 r., zawierającego informacje w zakresie co najmniej dat zawartych umów, przedmiotach umów, stronach umów, kwotach umów,- adresu strony internetowej, na której znajduje się dokumentacja przebiegu i efektów kontroli, przeprowadzonych w sądzie, oraz wystąpienia, stanowiska, wnioski i opinie podmiotów ją przeprowadzających,- kalendarz spotkań prezesa (prezes) sądu, które odbył (odbyła) w lipcu 2017 r.,- skany orzeczeń, zapadłych w wyniku wniesienia środka odwoławczego od orzeczenia sądu niższej instancji, zapadłego z oskarżenia o czyn zabroniony art. 231 Kodeksu karnego - wydanych przez Sąd w 2017 r.,- skany orzeczeń, zapadłych w wyniku wniesienia środka odwoławczego od orzeczenia sądu niższej instancji, zapadłego z oskarżenia o czyn zabroniony art. 212 Kodeksu karnego - wydanych przez Sąd w 2017 r."
                  - paragraph [ref=e170]: "Stowarzyszenie wnosi o udostępnienie wskazanych informacji w formie elektronicznej, na adres e-mail {{EMAIL}}."
                  - paragraph [ref=e171]: Katarzyna Batko-Tołuć, Bartosz Wilk - członkowie zarządu, zgodnie z zasadami reprezentacji
            - listitem [ref=e172]:
              - generic [ref=e174]:
                - heading [level=3] [ref=e175]:
                  - 'link "Read: Wniosek o udostępnienie informacji publicznej" [ref=e177] [cursor=pointer]':
                    - /url: /listy/7680
                  - generic [ref=e178]:
                    - text: przez
                    - link "Sąd Apelacyjny w Białymstoku" [ref=e179] [cursor=pointer]:
                      - /url: /instytucje/sad-apelacyjny-w-bialymstoku
                    - time [ref=e180]: 11 sierpnia 2017 07:15
                - generic [ref=e181]:
                  - paragraph [ref=e182]: Twoja wiadomość
                  - paragraph [ref=e183]: "Do: Adamik Dariusz Temat: Wniosek o udostępnienie informacji publicznej Wysłano: 11 sierpnia 2017 02:48:11 (UTC+01:00) Sarajewo, Skopie, Warszawa, Zagrzeb"
                  - paragraph [ref=e184]: "została przeczytana: 11 sierpnia 2017 07:00:27 (UTC+01:00) Sarajewo, Skopie, Warszawa, Zagrzeb."
            - listitem [ref=e185]:
              - generic [ref=e187]:
                - heading [level=3] [ref=e188]:
                  - link "A-061-79/17 dot. wniosku o udostępnienie informacji publicznej" [ref=e190] [cursor=pointer]:
                    - /url: /listy/8927
                  - generic [ref=e191]:
                    - text: przez
                    - link "Sąd Apelacyjny w Białymstoku" [ref=e192] [cursor=pointer]:
                      - /url: /instytucje/sad-apelacyjny-w-bialymstoku
                    - time [ref=e193]: 25 sierpnia 2017 12:15
                - paragraph [ref=e195]: "<html xmlns:v=\"urn:schemas-microsoft-com:vml\" xmlns:o=\"urn:schemas-microsoft-com:office:office\" xmlns:w=\"urn:schemas-microsoft-com:office:word\" xmlns:m=\"http://schemas.microsoft.com/office/2004/12/omml\" xmlns=\"http://www.w3.org/TR/REC-html40\"><head><meta http-equiv=\"Content-Type\" content=\"text/html; charset=iso-8859-2\"><meta name=\"Generator\" content=\"Microsoft Word 14 (filtered medium)\"><style><!--/* Font Definitions */@font-face {font-family:Calibri; panose-1:2 15 5 2 2 2 4 3 2 4;}/* Style Definitions */p.MsoNormal, li.MsoNormal, div.MsoNormal {margin:0cm; margin-bottom:.0001pt; font-size:11.0pt; font-family:\"Calibri\",\"sans-serif\"; mso-fareast-language:EN-US;}a:link, span.MsoHyperlink {mso-style-priority:99; color:blue; text-decoration:underline;}a:visited, span.MsoHyperlinkFollowed {mso-style-priority:99; color:purple; text-decoration:underline;}span.Stylwiadomocie-mail17 {mso-style-type:personal-compose; font-family:\"Calibri\",\"sans-serif\"; color:windowtext;}.MsoChpDefault {mso-style-type:export-only; font-family:\"Calibri\",\"sans-serif\"; mso-fareast-language:EN-US;}@page WordSection1 {size:612.0pt 792.0pt; margin:70.85pt 70.85pt 70.85pt 70.85pt;}div.WordSection1 {page:WordSection1;}--></style><!--[if gte mso 9]><xml><o:shapedefaults v:ext=\"edit\" spidmax=\"1026\" /></xml><![endif]--><!--[if gte mso 9]><xml><o:shapelayout v:ext=\"edit\"><o:idmap v:ext=\"edit\" data=\"1\" /></o:shapelayout></xml><![endif]--></head><body lang=\"PL\" link=\"blue\" vlink=\"purple\"><div class=\"WordSection1\"><p class=\"MsoNormal\"><o:p>&nbsp;</o:p></p></div></body></html>"
                - generic [ref=e196]:
                  - heading "Załączniki" [level=4] [ref=e197]
                  - list [ref=e198]:
                    - table [ref=e199]:
                      - rowgroup [ref=e200]:
                        - row [ref=e201]:
                          - cell [ref=e202]:
                            - listitem [ref=e203]:
                              - text: A-061-79-17.pdf (
                              - link "Nie wykryte" [ref=e204] [cursor=pointer]:
                                - /url: https://www.virustotal.com/file/377180e8afe93845b47d6851eee14701004d84ad03050bb62bb461f5f04508c3/analysis/1576790519/
                              - text: )
                              - link "Pobierz" [ref=e205] [cursor=pointer]:
                                - /url: /listy/zalacznik/7978/8927
                          - cell "Pokaż treść" [ref=e207]:
                            - 'generic "Uwaga: treść załączników została odczytana maszynowo, więc może zawierać błędy związane z nieprawidłowym odczytaniem znaków, a także błędną interpretacji układu tekstu na stronie. Jeśli nie masz stosownych uprawnień i potrzebujesz dostępu do oryginału, skontaktuj się z biurem SOWP." [ref=e208]'
                            - button "Pokaż treść" [ref=e209] [cursor=pointer]
                        - row [ref=e211]:
                          - cell [ref=e212]:
                            - listitem [ref=e213]:
                              - text: Umowy.pdf (
                              - link "Nie wykryte" [ref=e214] [cursor=pointer]:
                                - /url: https://www.virustotal.com/file/9ce65c2ab2d8474dbe2ada4e8c3681e7dcb988e58776b6c36a23c7be87afed0a/analysis/1576790541/
                              - text: )
                              - link "Pobierz" [ref=e215] [cursor=pointer]:
                                - /url: /listy/zalacznik/7979/8927
                          - cell "Pokaż treść" [ref=e217]:
                            - 'generic "Uwaga: treść załączników została odczytana maszynowo, więc może zawierać błędy związane z nieprawidłowym odczytaniem znaków, a także błędną interpretacji układu tekstu na stronie. Jeśli nie masz stosownych uprawnień i potrzebujesz dostępu do oryginału, skontaktuj się z biurem SOWP." [ref=e218]'
                            - button "Pokaż treść" [ref=e219] [cursor=pointer]
            - listitem [ref=e221]:
              - generic [ref=e223]:
                - heading [level=3] [ref=e224]:
                  - 'link "Nieprzeczytane: Wniosek o udostępnienie informacji publicznej" [ref=e226] [cursor=pointer]':
                    - /url: /listy/16858
                  - generic [ref=e227]:
                    - text: przez
                    - link "Sąd Apelacyjny w Białymstoku" [ref=e228] [cursor=pointer]:
                      - /url: /instytucje/sad-apelacyjny-w-bialymstoku
                    - time [ref=e229]: 16 lipca 2018 08:58
                - generic [ref=e230]:
                  - paragraph [ref=e231]: Twoja wiadomość
                  - paragraph [ref=e232]: "Do: Jarosławska Marzanna Temat: Wniosek o udostępnienie informacji publicznej Wysłano: 11 sierpnia 2017 02:48:11 (UTC+01:00) Sarajewo, Skopie, Warszawa, Zagrzeb"
                  - paragraph [ref=e233]: "została usunięta nieprzeczytana: 28 maja 2018 16:53:54 (UTC+01:00) Sarajewo, Skopie, Warszawa, Zagrzeb."
      - generic [ref=e234]:
        - generic [ref=e235]:
          - text: Sieć Obywatelska - Watchdog Polska ul. Szpitalna 5/5 00-031 Warszawa
          - paragraph [ref=e236]:
            - link "Klauzula RODO" [ref=e237] [cursor=pointer]:
              - /url: https://fedrowanie.siecobywatelska.pl/media_internal/tinycontent/uploads/11_KLAUZULA.-.dane.z.udip.na.stronach.www.i.w.mediach.spol.pdf
        - generic [ref=e238]:
          - text: "silnik:"
          - link "jawne.info.pl" [ref=e239] [cursor=pointer]:
            - /url: http://jawne.info.pl
          - text: "| v1.5.77.deps |"
          - link "GitHub" [ref=e240] [cursor=pointer]:
            - /url: https://github.com/watchdogpolska/feder
          - text: –
          - link "efd8a3b" [ref=e242] [cursor=pointer]:
            - /url: https://github.com/watchdogpolska/feder/compare/efd8a3b2...master
          - text: "|"
          - link "API" [ref=e243] [cursor=pointer]:
            - /url: /api/
        - generic [ref=e245]: Ta strona wykorzystuje cookies.
  - list [ref=e247]:
    - listitem [ref=e248]:
      - link "Ukryj »" [ref=e249] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e250]:
      - link "Toggle Theme" [ref=e251] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e254]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e255]
      - link "Historia /sprawy/monitoring-sadow-apelacyjnych-1" [ref=e256] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e257]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e258]
      - link "Wersje Django 5.2.17" [ref=e259] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e260]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e261]
      - 'link "Czas CPU: 167.91ms (163.45ms)" [ref=e262] [cursor=pointer]':
        - /url: "#"
    - listitem [ref=e263]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e264]
      - link "Ustawienia" [ref=e265] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e266]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e267]
      - link "Nagłówki" [ref=e268] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e269]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e270]
      - link "Zapytania CaseDetailView" [ref=e271] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e272]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e273]
      - link "SQL 18 queries in 11.61ms" [ref=e274] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e275]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e276]
      - link "Pliki statyczne 3 użyte plików" [ref=e277] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e278]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e279]
      - link "Templatki cases/case_detail.html" [ref=e280] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e281]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e282]
      - link "Alerty" [ref=e283] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e284]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e285]
      - link "Cache 2 wywołania w 0.16ms" [ref=e286] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e287]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e288]
      - link "Sygnały 88 odbiorców 15 sygnałów" [ref=e289] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e290]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e291]
      - link "Gmina" [ref=e292] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e293]:
      - checkbox "Enable for next and successive requests" [ref=e294]
      - generic [ref=e295]: Przechwycone przekierowania
    - listitem [ref=e296]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e297]
      - link "Profilowanie" [ref=e298] [cursor=pointer]:
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