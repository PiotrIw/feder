# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: visual_baseline.spec.ts >> mobile (375px) >> monitorings-update - visual baseline
- Location: tests/bs3-bs5_migration/visual_baseline.spec.ts:10:11

# Error details

```
Error: expect(page).toHaveScreenshot(expected) failed

  Expected an image 375px by 2712px, received 375px by 2667px. 219516 pixels (ratio 0.22 of all image pixels) are different.

  Snapshot: monitorings-update-mobile.png

Call log:
  - Expect "toHaveScreenshot(monitorings-update-mobile.png)" with timeout 5000ms
    - verifying given screenshot expectation
  - taking page screenshot
    - disabled all CSS animations
  - waiting for fonts to load...
  - fonts loaded
  - Expected an image 375px by 2712px, received 375px by 2667px. 219516 pixels (ratio 0.22 of all image pixels) are different.
  - waiting 100ms before taking screenshot
  - taking page screenshot
    - disabled all CSS animations
  - waiting for fonts to load...
  - fonts loaded
  - captured a stable screenshot
  - Expected an image 375px by 2712px, received 375px by 2667px. 219516 pixels (ratio 0.22 of all image pixels) are different.

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
        - listitem [ref=e72]: / Zaktualizuj monitoring
      - generic [ref=e74]:
        - link "Edytuj" [ref=e75] [cursor=pointer]:
          - /url: /monitoringi/monitoring-sadow-apelacyjnych/~edytuj
        - link "Aktualizuj wyniki" [ref=e76] [cursor=pointer]:
          - /url: /monitoringi/monitoring-sadow-apelacyjnych/~results-update
        - link "Przypisz" [ref=e77] [cursor=pointer]:
          - /url: /monitoringi/monitoring-sadow-apelacyjnych/~przypisz
        - link "Usuń" [ref=e78] [cursor=pointer]:
          - /url: /monitoringi/monitoring-sadow-apelacyjnych/~usun
        - link "Utwórz sprawę" [ref=e79] [cursor=pointer]:
          - /url: /sprawy/~utworz-5
        - link "Wiadomość masowa" [ref=e80] [cursor=pointer]:
          - /url: /monitoringi/monitoring-sadow-apelacyjnych/~wiadomosc-masowa
        - link "Uprawnienia" [ref=e81] [cursor=pointer]:
          - /url: /monitoringi/monitoring-sadow-apelacyjnych/~uprawnienia
        - link "Lista alertów" [ref=e82] [cursor=pointer]:
          - /url: /alerty/monitoring-5
        - link "Zobacz dzienniki" [ref=e83] [cursor=pointer]:
          - /url: /listy/logi/monitoring-5
        - link "Zobacz tagi" [ref=e85] [cursor=pointer]:
          - /url: /sprawy/tagi/monitoring-5
        - link "Zobacz raport" [ref=e87] [cursor=pointer]:
          - /url: /monitoringi/monitoring-sadow-apelacyjnych/raport
        - link "Zobacz tabelę spraw" [ref=e89] [cursor=pointer]:
          - /url: /monitoringi/monitoring-sadow-apelacyjnych/monitoring_cases_table
      - heading [level=2] [ref=e92]:
        - link "Monitoring sądów apelacyjnych" [ref=e94] [cursor=pointer]:
          - /url: /monitoringi/monitoring-sadow-apelacyjnych
      - generic [ref=e96]:
        - generic [ref=e97]:
          - group "Monitoring" [ref=e99]:
            - generic [ref=e101]:
              - generic [ref=e102]: Nazwa*
              - textbox "Nazwa*" [ref=e103]: Monitoring sądów apelacyjnych
            - generic [ref=e104]:
              - generic [ref=e105]: Opis
              - textbox "Opis" [ref=e106]
            - generic [ref=e108]:
              - checkbox "Powiadamiaj o alertach" [checked] [ref=e109]
              - generic [ref=e110]: Powiadamiaj o alertach
              - generic [ref=e111]: Powiadom o nowych alertach osoby, które mogą je widzieć
            - generic [ref=e113]:
              - checkbox "Czy publicznie widoczny?" [checked] [ref=e114]
              - generic [ref=e115]: Czy publicznie widoczny?
            - generic [ref=e117]:
              - checkbox "Czy ukrywać nowe sprawy przy przypisywaniu?" [ref=e118]
              - generic [ref=e119]: Czy ukrywać nowe sprawy przy przypisywaniu?
            - generic [ref=e121]:
              - checkbox "Korzystaj z LLM" [ref=e122]
              - generic [ref=e123]: Korzystaj z LLM
              - generic [ref=e124]: Przed włączeniem upewnij się, że treść wniosku nie będzie już zmieniana. Zawsze możesz wrócić do edycji i włączyć później.
          - group "Szablon" [ref=e126]:
            - generic [ref=e128]:
              - generic [ref=e129]: Temat*
              - textbox "Temat*" [ref=e130]: Wniosek o udostępnienie informacji publicznej
            - generic [ref=e131]:
              - generic [ref=e132]: Szablon*
              - application [ref=e133]:
                - generic [ref=e134]:
                  - generic [ref=e135]:
                    - menubar [ref=e137]:
                      - menuitem "Plik" [ref=e138] [cursor=pointer]
                      - menuitem "Edytuj" [ref=e140] [cursor=pointer]
                      - menuitem "Widok" [ref=e142] [cursor=pointer]
                      - menuitem "Wstaw" [ref=e144] [cursor=pointer]
                      - menuitem "Format" [ref=e146] [cursor=pointer]
                      - menuitem "Narzędzia" [ref=e148] [cursor=pointer]
                      - menuitem "Tabela" [ref=e150] [cursor=pointer]
                      - menuitem "Pomoc" [ref=e152] [cursor=pointer]
                    - group [ref=e154]:
                      - group [ref=e155]:
                        - toolbar [ref=e156]:
                          - button "Cofnij" [disabled] [ref=e157]
                          - button "Powtórz" [disabled] [ref=e161]
                        - toolbar [ref=e165]:
                          - button "Wysokość Linii" [ref=e166] [cursor=pointer]
                        - toolbar [ref=e173]:
                          - button "Odkryj lub ukryj dodatkowe elementy na pasku narzędzi" [ref=e174] [cursor=pointer]
                  - iframe [ref=e180]:
                    - generic "Obszar tekstu sformatowanego. Naciśnij ALT-0, aby uzyskać pomoc." [ref=f1e1]:
                      - paragraph [ref=f1e2]: "Stowarzyszenie Sieć Obywatelska Watchdog Polska wnosi o udostępnienie poprzez przesłanie następujących informacji: - adresu strony internetowej, na której znajdują się orzeczenia dyscyplinarne, wydane wobec sędziów przez tutejszy Sąd, - rejestru umów, zawartych w imieniu Sądu od 1 stycznia 2017 r. do 31 lipca 2017 r., zawierającego informacje w zakresie co najmniej dat zawartych umów, przedmiotach umów, stronach umów, kwotach umów, - adresu strony internetowej, na której znajduje się dokumentacja przebiegu i efektów kontroli, przeprowadzonych w sądzie, oraz wystąpienia, stanowiska, wnioski i opinie podmiotów ją przeprowadzających, - kalendarz spotkań prezesa (prezes) sądu, które odbył (odbyła) w lipcu 2017 r., - skany orzeczeń, zapadłych w wyniku wniesienia środka odwoławczego od orzeczenia sądu niższej instancji, zapadłego z oskarżenia o czyn zabroniony art. 231 Kodeksu karnego - wydanych przez Sąd w 2017 r., - skany orzeczeń, zapadłych w wyniku wniesienia środka odwoławczego od orzeczenia sądu niższej instancji, zapadłego z oskarżenia o czyn zabroniony art. 212 Kodeksu karnego - wydanych przez Sąd w 2017 r. Stowarzyszenie wnosi o udostępnienie wskazanych informacji w formie elektronicznej, na adres e-mail {{EMAIL}}. Katarzyna Batko-Tołuć, Bartosz Wilk - członkowie zarządu, zgodnie z zasadami reprezentacji"
                - generic [ref=e181]:
                  - generic [ref=e182]:
                    - navigation [ref=e183]:
                      - button "p" [ref=e184]
                    - generic [ref=e185]:
                      - button "175 sł." [ref=e186] [cursor=pointer]
                      - link "Build with TinyMCE" [ref=e188]:
                        - /url: https://www.tiny.cloud/powered-by-tiny?utm_campaign=poweredby&utm_source=tiny&utm_medium=referral&utm_content=v7
                        - text: Build with
                  - generic "Naciśnij klawisze strzałek w górę i w dół, aby zmienić rozmiar edytora." [ref=e195]
              - generic [ref=e199]: "Użyj: {{EMAIL}} aby umieścić adres odpowiedzi, {{ADRESAT}} aby umieścić nazwę adressata."
            - generic [ref=e200]:
              - generic [ref=e201]: Podpis w e-mail*
              - application [ref=e202]:
                - generic [ref=e203]:
                  - generic [ref=e204]:
                    - menubar [ref=e206]:
                      - menuitem "Plik" [ref=e207] [cursor=pointer]
                      - menuitem "Edytuj" [ref=e209] [cursor=pointer]
                      - menuitem "Widok" [ref=e211] [cursor=pointer]
                      - menuitem "Wstaw" [ref=e213] [cursor=pointer]
                      - menuitem "Format" [ref=e215] [cursor=pointer]
                      - menuitem "Narzędzia" [ref=e217] [cursor=pointer]
                      - menuitem "Tabela" [ref=e219] [cursor=pointer]
                      - menuitem "Pomoc" [ref=e221] [cursor=pointer]
                    - group [ref=e223]:
                      - group [ref=e224]:
                        - toolbar [ref=e225]:
                          - button "Cofnij" [disabled] [ref=e226]
                          - button "Powtórz" [disabled] [ref=e230]
                        - toolbar [ref=e234]:
                          - button "Wysokość Linii" [ref=e235] [cursor=pointer]
                        - toolbar [ref=e242]:
                          - button "Odkryj lub ukryj dodatkowe elementy na pasku narzędzi" [ref=e243] [cursor=pointer]
                  - iframe [ref=e249]:
                    - generic "Obszar tekstu sformatowanego. Naciśnij ALT-0, aby uzyskać pomoc." [ref=f2e1]:
                      - paragraph [ref=f2e2]: "---"
                - generic [ref=e250]:
                  - generic [ref=e251]:
                    - navigation [ref=e252]:
                      - button "p" [ref=e253]
                    - generic [ref=e254]:
                      - button "0 sł." [ref=e255] [cursor=pointer]
                      - link "Build with TinyMCE" [ref=e257]:
                        - /url: https://www.tiny.cloud/powered-by-tiny?utm_campaign=poweredby&utm_source=tiny&utm_medium=referral&utm_content=v7
                        - text: Build with
                  - generic "Naciśnij klawisze strzałek w górę i w dół, aby zmienić rozmiar edytora." [ref=e264]
              - generic [ref=e268]: Podpis w stopce e-maili, w tym w odpowiedziach na e-maile
            - generic [ref=e269]:
              - generic [ref=e270]: Domain*
              - combobox "Domain*" [ref=e271]:
                - option "---------"
                - option "fedrowanie.siecobywatelska.pl" [selected]
                - option "pokot.pl"
                - option "monitoring.bartoszwilk.pl"
                - option "info.lasyiobywatele.pl"
                - option "nijakowski.pl"
                - option "fedr.uratujzwierze.pl"
                - option "pytania.ofop.eu"
                - option "info.szkolajestnasza.pl"
              - generic [ref=e272]: Domena użyta do wysłania wiadomości
        - button "Aktualizuj" [ref=e275] [cursor=pointer]
      - generic [ref=e276]:
        - generic [ref=e277]:
          - text: Sieć Obywatelska - Watchdog Polska ul. Szpitalna 5/5 00-031 Warszawa
          - paragraph [ref=e278]:
            - link "Klauzula RODO" [ref=e279] [cursor=pointer]:
              - /url: https://fedrowanie.siecobywatelska.pl/media_internal/tinycontent/uploads/11_KLAUZULA.-.dane.z.udip.na.stronach.www.i.w.mediach.spol.pdf
        - generic [ref=e280]:
          - text: "silnik:"
          - link "jawne.info.pl" [ref=e281] [cursor=pointer]:
            - /url: http://jawne.info.pl
          - text: "| v1.5.77.deps |"
          - link "GitHub" [ref=e282] [cursor=pointer]:
            - /url: https://github.com/watchdogpolska/feder
          - text: –
          - link "efd8a3b" [ref=e284] [cursor=pointer]:
            - /url: https://github.com/watchdogpolska/feder/compare/efd8a3b2...master
          - text: "|"
          - link "API" [ref=e285] [cursor=pointer]:
            - /url: /api/
        - generic [ref=e287]: Ta strona wykorzystuje cookies.
  - list [ref=e289]:
    - listitem [ref=e290]:
      - link "Ukryj »" [ref=e291] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e292]:
      - link "Toggle Theme" [ref=e293] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e296]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e297]
      - link "Historia /monitoringi/monitoring-sadow-apelacyjnych/~edytuj" [ref=e298] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e299]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e300]
      - link "Wersje Django 5.2.17" [ref=e301] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e302]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e303]
      - 'link "Czas CPU: 323.90ms (326.60ms)" [ref=e304] [cursor=pointer]':
        - /url: "#"
    - listitem [ref=e305]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e306]
      - link "Ustawienia" [ref=e307] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e308]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e309]
      - link "Nagłówki" [ref=e310] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e311]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e312]
      - link "Zapytania MonitoringUpdateView" [ref=e313] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e314]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e315]
      - link "SQL 8 queries in 2.72ms" [ref=e316] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e317]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e318]
      - link "Pliki statyczne 5 użytych plików" [ref=e319] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e320]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e321]
      - link "Templatki monitorings/monitoring_form.html" [ref=e322] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e323]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e324]
      - link "Alerty" [ref=e325] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e326]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e327]
      - link "Cache 2 wywołania w 0.13ms" [ref=e328] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e329]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e330]
      - link "Sygnały 88 odbiorców 15 sygnałów" [ref=e331] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e332]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e333]
      - link "Gmina" [ref=e334] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e335]:
      - checkbox "Enable for next and successive requests" [ref=e336]
      - generic [ref=e337]: Przechwycone przekierowania
    - listitem [ref=e338]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e339]
      - link "Profilowanie" [ref=e340] [cursor=pointer]:
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