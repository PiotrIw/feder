# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: layout_assertions.spec.ts >> Layout sanity - mobile >> cases-details - no horizontal overflow on mobile
- Location: tests/bs3-bs5_migration/layout_assertions.spec.ts:30:9

# Error details

```
Error: Mobile layout has horizontal scroll

expect(received).toBeLessThanOrEqual(expected)

Expected: <= 20
Received:    165
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
        - listitem [ref=e72]: "/ Monitoring sądów apelacyjnych #1"
      - generic [ref=e74]:
        - link "Edytuj" [ref=e75] [cursor=pointer]:
          - /url: /sprawy/monitoring-sadow-apelacyjnych-1/~edytuj
        - link "Usuń" [ref=e76] [cursor=pointer]:
          - /url: /sprawy/monitoring-sadow-apelacyjnych-1/~usun
        - link "Zobacz dzienniki" [ref=e77] [cursor=pointer]:
          - /url: /listy/logi/spraw-2684
        - button "Dodaj przesyłkę pocztową" [ref=e79] [cursor=pointer]
        - link "Dodaj list" [ref=e81] [cursor=pointer]:
          - /url: /listy/~utworz-2684
      - 'heading "Monitoring sądów apelacyjnych #1" [level=1] [ref=e84]'
      - generic [ref=e86]:
        - generic [ref=e87]:
          - table [ref=e89]:
            - rowgroup [ref=e90]:
              - row [ref=e91]:
                - cell [ref=e92]:
                  - paragraph [ref=e93]: "Instytucja:"
                - cell [ref=e95]:
                  - paragraph [ref=e96]:
                    - link "Sąd Apelacyjny w Białymstoku" [ref=e97] [cursor=pointer]:
                      - /url: /instytucje/sad-apelacyjny-w-bialymstoku
              - row [ref=e98]:
                - cell [ref=e99]:
                  - paragraph [ref=e100]: "Email instytucji:"
                - cell [ref=e102]:
                  - paragraph [ref=e103]: boi@bialystok.sa.gov.pl
              - row [ref=e104]:
                - cell [ref=e105]:
                  - paragraph [ref=e106]: "Monitoring:"
                - cell [ref=e108]:
                  - paragraph [ref=e109]:
                    - link "Monitoring sądów apelacyjnych" [ref=e110] [cursor=pointer]:
                      - /url: /monitoringi/monitoring-sadow-apelacyjnych
              - row [ref=e111]:
                - cell [ref=e112]:
                  - paragraph [ref=e113]: "Email sprawy:"
                - cell [ref=e115]:
                  - paragraph [ref=e116]: sprawa-2684@fedrowanie.siecobywatelska.pl
              - row [ref=e117]:
                - cell [ref=e118]:
                  - paragraph [ref=e119]: "Liczba listów:"
                - cell [ref=e121]:
                  - paragraph [ref=e122]: "4"
              - row [ref=e123]:
                - cell [ref=e124]:
                  - paragraph [ref=e125]: "Liczba spamu:"
                - cell [ref=e127]:
                  - paragraph [ref=e128]: "0"
              - row [ref=e129]:
                - cell [ref=e130]:
                  - paragraph [ref=e131]: "Status pierwszego wniosku:"
                - cell [ref=e133]:
                  - paragraph
              - row [ref=e134]:
                - cell [ref=e135]:
                  - paragraph [ref=e136]: "Status ostatniego wniosku:"
                - cell [ref=e138]:
                  - paragraph
              - row [ref=e139]:
                - cell [ref=e140]:
                  - paragraph [ref=e141]: "Otrzymano potwierdzenie:"
                - cell [ref=e143]:
                  - paragraph [ref=e144]
              - row [ref=e146]:
                - cell [ref=e147]:
                  - paragraph [ref=e148]: "Otrzymano odpowiedź:"
                - cell [ref=e150]:
                  - paragraph [ref=e151]
              - row [ref=e153]:
                - cell [ref=e154]:
                  - paragraph [ref=e155]: "Poddany kwarantannie:"
                - cell [ref=e157]:
                  - paragraph [ref=e158]
          - heading "Znormalizowana odpowiedź" [level=4] [ref=e161]
        - generic [ref=e162]:
          - heading "Treść" [level=4] [ref=e163]
          - list [ref=e165]:
            - listitem [ref=e166]:
              - generic [ref=e168]:
                - heading [level=3] [ref=e169]:
                  - link "Wniosek o udostępnienie informacji publicznej" [ref=e171] [cursor=pointer]:
                    - /url: /listy/7302
                  - generic [ref=e172]:
                    - text: przez
                    - link "adobrawy" [ref=e173] [cursor=pointer]:
                      - /url: /uzytkownik/adobrawy/
                    - time [ref=e174]: 11 sierpnia 2017 02:48
                - generic [ref=e175]:
                  - paragraph [ref=e176]: "Stowarzyszenie Sieć Obywatelska Watchdog Polska wnosi o udostępnienie poprzez przesłanie następujących informacji:"
                  - paragraph [ref=e177]: "- adresu strony internetowej, na której znajdują się orzeczenia dyscyplinarne, wydane wobec sędziów przez tutejszy Sąd,- rejestru umów, zawartych w imieniu Sądu od 1 stycznia 2017 r. do 31 lipca 2017 r., zawierającego informacje w zakresie co najmniej dat zawartych umów, przedmiotach umów, stronach umów, kwotach umów,- adresu strony internetowej, na której znajduje się dokumentacja przebiegu i efektów kontroli, przeprowadzonych w sądzie, oraz wystąpienia, stanowiska, wnioski i opinie podmiotów ją przeprowadzających,- kalendarz spotkań prezesa (prezes) sądu, które odbył (odbyła) w lipcu 2017 r.,- skany orzeczeń, zapadłych w wyniku wniesienia środka odwoławczego od orzeczenia sądu niższej instancji, zapadłego z oskarżenia o czyn zabroniony art. 231 Kodeksu karnego - wydanych przez Sąd w 2017 r.,- skany orzeczeń, zapadłych w wyniku wniesienia środka odwoławczego od orzeczenia sądu niższej instancji, zapadłego z oskarżenia o czyn zabroniony art. 212 Kodeksu karnego - wydanych przez Sąd w 2017 r."
                  - paragraph [ref=e178]: "Stowarzyszenie wnosi o udostępnienie wskazanych informacji w formie elektronicznej, na adres e-mail {{EMAIL}}."
                  - paragraph [ref=e179]: Katarzyna Batko-Tołuć, Bartosz Wilk - członkowie zarządu, zgodnie z zasadami reprezentacji
            - listitem [ref=e180]:
              - generic [ref=e182]:
                - heading [level=3] [ref=e183]:
                  - 'link "Read: Wniosek o udostępnienie informacji publicznej" [ref=e185] [cursor=pointer]':
                    - /url: /listy/7680
                  - generic [ref=e186]:
                    - text: przez
                    - link "Sąd Apelacyjny w Białymstoku" [ref=e187] [cursor=pointer]:
                      - /url: /instytucje/sad-apelacyjny-w-bialymstoku
                    - time [ref=e188]: 11 sierpnia 2017 07:15
                - generic [ref=e189]:
                  - paragraph [ref=e190]: Twoja wiadomość
                  - paragraph [ref=e191]: "Do: Adamik Dariusz Temat: Wniosek o udostępnienie informacji publicznej Wysłano: 11 sierpnia 2017 02:48:11 (UTC+01:00) Sarajewo, Skopie, Warszawa, Zagrzeb"
                  - paragraph [ref=e192]: "została przeczytana: 11 sierpnia 2017 07:00:27 (UTC+01:00) Sarajewo, Skopie, Warszawa, Zagrzeb."
            - listitem [ref=e193]:
              - generic [ref=e195]:
                - heading [level=3] [ref=e196]:
                  - link "A-061-79/17 dot. wniosku o udostępnienie informacji publicznej" [ref=e198] [cursor=pointer]:
                    - /url: /listy/8927
                  - generic [ref=e199]:
                    - text: przez
                    - link "Sąd Apelacyjny w Białymstoku" [ref=e200] [cursor=pointer]:
                      - /url: /instytucje/sad-apelacyjny-w-bialymstoku
                    - time [ref=e201]: 25 sierpnia 2017 12:15
                - paragraph [ref=e203]: "<html xmlns:v=\"urn:schemas-microsoft-com:vml\" xmlns:o=\"urn:schemas-microsoft-com:office:office\" xmlns:w=\"urn:schemas-microsoft-com:office:word\" xmlns:m=\"http://schemas.microsoft.com/office/2004/12/omml\" xmlns=\"http://www.w3.org/TR/REC-html40\"><head><meta http-equiv=\"Content-Type\" content=\"text/html; charset=iso-8859-2\"><meta name=\"Generator\" content=\"Microsoft Word 14 (filtered medium)\"><style><!--/* Font Definitions */@font-face {font-family:Calibri; panose-1:2 15 5 2 2 2 4 3 2 4;}/* Style Definitions */p.MsoNormal, li.MsoNormal, div.MsoNormal {margin:0cm; margin-bottom:.0001pt; font-size:11.0pt; font-family:\"Calibri\",\"sans-serif\"; mso-fareast-language:EN-US;}a:link, span.MsoHyperlink {mso-style-priority:99; color:blue; text-decoration:underline;}a:visited, span.MsoHyperlinkFollowed {mso-style-priority:99; color:purple; text-decoration:underline;}span.Stylwiadomocie-mail17 {mso-style-type:personal-compose; font-family:\"Calibri\",\"sans-serif\"; color:windowtext;}.MsoChpDefault {mso-style-type:export-only; font-family:\"Calibri\",\"sans-serif\"; mso-fareast-language:EN-US;}@page WordSection1 {size:612.0pt 792.0pt; margin:70.85pt 70.85pt 70.85pt 70.85pt;}div.WordSection1 {page:WordSection1;}--></style><!--[if gte mso 9]><xml><o:shapedefaults v:ext=\"edit\" spidmax=\"1026\" /></xml><![endif]--><!--[if gte mso 9]><xml><o:shapelayout v:ext=\"edit\"><o:idmap v:ext=\"edit\" data=\"1\" /></o:shapelayout></xml><![endif]--></head><body lang=\"PL\" link=\"blue\" vlink=\"purple\"><div class=\"WordSection1\"><p class=\"MsoNormal\"><o:p>&nbsp;</o:p></p></div></body></html>"
                - generic [ref=e204]:
                  - heading "Załączniki" [level=4] [ref=e205]
                  - list [ref=e206]:
                    - table [ref=e207]:
                      - rowgroup [ref=e208]:
                        - row [ref=e209]:
                          - cell [ref=e210]:
                            - listitem [ref=e211]:
                              - text: A-061-79-17.pdf (
                              - link "Nie wykryte" [ref=e212] [cursor=pointer]:
                                - /url: https://www.virustotal.com/file/377180e8afe93845b47d6851eee14701004d84ad03050bb62bb461f5f04508c3/analysis/1576790519/
                              - text: )
                              - link "Pobierz" [ref=e213] [cursor=pointer]:
                                - /url: /listy/zalacznik/7978/8927
                          - cell "Pokaż treść" [ref=e215]:
                            - 'generic "Uwaga: treść załączników została odczytana maszynowo, więc może zawierać błędy związane z nieprawidłowym odczytaniem znaków, a także błędną interpretacji układu tekstu na stronie. Jeśli nie masz stosownych uprawnień i potrzebujesz dostępu do oryginału, skontaktuj się z biurem SOWP." [ref=e216]'
                            - button "Pokaż treść" [ref=e217] [cursor=pointer]
                        - row [ref=e219]:
                          - cell [ref=e220]:
                            - listitem [ref=e221]:
                              - text: Umowy.pdf (
                              - link "Nie wykryte" [ref=e222] [cursor=pointer]:
                                - /url: https://www.virustotal.com/file/9ce65c2ab2d8474dbe2ada4e8c3681e7dcb988e58776b6c36a23c7be87afed0a/analysis/1576790541/
                              - text: )
                              - link "Pobierz" [ref=e223] [cursor=pointer]:
                                - /url: /listy/zalacznik/7979/8927
                          - cell "Pokaż treść" [ref=e225]:
                            - 'generic "Uwaga: treść załączników została odczytana maszynowo, więc może zawierać błędy związane z nieprawidłowym odczytaniem znaków, a także błędną interpretacji układu tekstu na stronie. Jeśli nie masz stosownych uprawnień i potrzebujesz dostępu do oryginału, skontaktuj się z biurem SOWP." [ref=e226]'
                            - button "Pokaż treść" [ref=e227] [cursor=pointer]
            - listitem [ref=e229]:
              - generic [ref=e231]:
                - heading [level=3] [ref=e232]:
                  - 'link "Nieprzeczytane: Wniosek o udostępnienie informacji publicznej" [ref=e234] [cursor=pointer]':
                    - /url: /listy/16858
                  - generic [ref=e235]:
                    - text: przez
                    - link "Sąd Apelacyjny w Białymstoku" [ref=e236] [cursor=pointer]:
                      - /url: /instytucje/sad-apelacyjny-w-bialymstoku
                    - time [ref=e237]: 16 lipca 2018 08:58
                - generic [ref=e238]:
                  - paragraph [ref=e239]: Twoja wiadomość
                  - paragraph [ref=e240]: "Do: Jarosławska Marzanna Temat: Wniosek o udostępnienie informacji publicznej Wysłano: 11 sierpnia 2017 02:48:11 (UTC+01:00) Sarajewo, Skopie, Warszawa, Zagrzeb"
                  - paragraph [ref=e241]: "została usunięta nieprzeczytana: 28 maja 2018 16:53:54 (UTC+01:00) Sarajewo, Skopie, Warszawa, Zagrzeb."
      - generic [ref=e242]:
        - generic [ref=e243]:
          - text: Sieć Obywatelska - Watchdog Polska ul. Szpitalna 5/5 00-031 Warszawa
          - paragraph [ref=e244]:
            - link "Klauzula RODO" [ref=e245] [cursor=pointer]:
              - /url: https://fedrowanie.siecobywatelska.pl/media_internal/tinycontent/uploads/11_KLAUZULA.-.dane.z.udip.na.stronach.www.i.w.mediach.spol.pdf
        - generic [ref=e246]:
          - text: "silnik:"
          - link "jawne.info.pl" [ref=e247] [cursor=pointer]:
            - /url: http://jawne.info.pl
          - text: "| v1.5.77.deps |"
          - link "GitHub" [ref=e248] [cursor=pointer]:
            - /url: https://github.com/watchdogpolska/feder
          - text: –
          - link "efd8a3b" [ref=e250] [cursor=pointer]:
            - /url: https://github.com/watchdogpolska/feder/compare/efd8a3b2...master
          - text: "|"
          - link "API" [ref=e251] [cursor=pointer]:
            - /url: /api/
        - generic [ref=e253]: Ta strona wykorzystuje cookies.
  - list [ref=e255]:
    - listitem [ref=e256]:
      - link "Ukryj »" [ref=e257] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e258]:
      - link "Toggle Theme" [ref=e259] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e262]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e263]
      - link "Historia /sprawy/monitoring-sadow-apelacyjnych-1" [ref=e264] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e265]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e266]
      - link "Wersje Django 5.2.17" [ref=e267] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e268]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e269]
      - 'link "Czas CPU: 139.86ms (146.32ms)" [ref=e270] [cursor=pointer]':
        - /url: "#"
    - listitem [ref=e271]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e272]
      - link "Ustawienia" [ref=e273] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e274]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e275]
      - link "Nagłówki" [ref=e276] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e277]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e278]
      - link "Zapytania CaseDetailView" [ref=e279] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e280]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e281]
      - link "SQL 18 queries in 8.83ms" [ref=e282] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e283]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e284]
      - link "Pliki statyczne 3 użyte plików" [ref=e285] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e286]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e287]
      - link "Templatki cases/case_detail.html" [ref=e288] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e289]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e290]
      - link "Alerty" [ref=e291] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e292]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e293]
      - link "Cache 2 wywołania w 0.12ms" [ref=e294] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e295]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e296]
      - link "Sygnały 88 odbiorców 15 sygnałów" [ref=e297] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e298]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e299]
      - link "Gmina" [ref=e300] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e301]:
      - checkbox "Enable for next and successive requests" [ref=e302]
      - generic [ref=e303]: Przechwycone przekierowania
    - listitem [ref=e304]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e305]
      - link "Profilowanie" [ref=e306] [cursor=pointer]:
        - /url: "#"
```

# Test source

```ts
  1  | import { test, expect } from '@playwright/test';
  2  | import { PAGES } from './pages';
  3  | import { VIEWPORTS } from './viewports';
  4  | 
  5  | // Bootstrap 3's `.row` negative margins (-15px) routinely push scrollWidth ~4px past
  6  | // clientWidth on every page even with no visible scrollbar - that's cosmetic BS3 grid
  7  | // noise, not real overflow. A genuinely overflowing wide table measured ~278px over.
  8  | // This tolerance separates the two instead of flagging every single page.
  9  | const OVERFLOW_TOLERANCE_PX = 20;
  10 | 
  11 | test.describe('Layout sanity - desktop', () => {
  12 |   test.use({ viewport: VIEWPORTS.desktop });
  13 | 
  14 |   for (const page of PAGES) {
  15 |     test(`${page.name} - no horizontal overflow`, async ({ page: pw }) => {
  16 |       await pw.goto(page.path);
  17 |       await pw.waitForLoadState('networkidle');
  18 |       const overflowPx = await pw.evaluate(() =>
  19 |         document.documentElement.scrollWidth - document.documentElement.clientWidth
  20 |       );
  21 |       expect(overflowPx, 'Page has horizontal scroll').toBeLessThanOrEqual(OVERFLOW_TOLERANCE_PX);
  22 |     });
  23 |   }
  24 | });
  25 | 
  26 | test.describe('Layout sanity - mobile', () => {
  27 |   test.use({ viewport: VIEWPORTS.mobile });
  28 | 
  29 |   for (const page of PAGES) {
  30 |     test(`${page.name} - no horizontal overflow on mobile`, async ({ page: pw }) => {
  31 |       await pw.goto(page.path);
  32 |       await pw.waitForLoadState('networkidle');
  33 |       const overflowPx = await pw.evaluate(() =>
  34 |         document.documentElement.scrollWidth - document.documentElement.clientWidth
  35 |       );
> 36 |       expect(overflowPx, 'Mobile layout has horizontal scroll').toBeLessThanOrEqual(OVERFLOW_TOLERANCE_PX);
     |                                                                 ^ Error: Mobile layout has horizontal scroll
  37 |     });
  38 |   }
  39 | });
  40 | 
  41 | // This app's desktop layout (feder/main/templates/base.html) is a permanent left
  42 | // `.sidebar` next to `.content`, not a top navbar - `.navbar` is `display: none` above
  43 | // the mobile breakpoint (it only reappears, with `.navbar-toggle`, on small screens).
  44 | // So "nav above content" doesn't apply on desktop; the real desktop invariant is
  45 | // "sidebar sits to the left of content", checked below instead.
  46 | test.describe('Navigation structure', () => {
  47 |   test.use({ viewport: VIEWPORTS.desktop });
  48 | 
  49 |   test('sidebar is left of main content on desktop', async ({ page: pw }) => {
  50 |     await pw.goto('/');
  51 |     await pw.waitForLoadState('networkidle');
  52 |     const sidebarBox = await pw.locator('.sidebar').first().boundingBox();
  53 |     const contentBox = await pw.locator('.content').first().boundingBox();
  54 |     expect(sidebarBox).toBeTruthy();
  55 |     expect(contentBox).toBeTruthy();
  56 |     expect(sidebarBox!.x + sidebarBox!.width).toBeLessThanOrEqual(contentBox!.x + 5);
  57 |   });
  58 | 
  59 |   test('navbar collapses on mobile', async ({ page: pw }) => {
  60 |     await pw.setViewportSize(VIEWPORTS.mobile);
  61 |     await pw.goto('/');
  62 |     await pw.waitForLoadState('networkidle');
  63 |     const toggle = pw.locator('.navbar-toggle, .navbar-toggler');
  64 |     await expect(toggle).toBeVisible();
  65 |   });
  66 | });
  67 | 
```