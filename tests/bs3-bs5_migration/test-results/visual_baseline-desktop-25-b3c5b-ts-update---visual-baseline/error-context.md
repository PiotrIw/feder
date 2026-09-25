# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: visual_baseline.spec.ts >> desktop (2560px) >> monitorings-results-update - visual baseline
- Location: tests/bs3-bs5_migration/visual_baseline.spec.ts:10:11

# Error details

```
Error: expect(page).toHaveScreenshot(expected) failed

  354088 pixels (ratio 0.10 of all image pixels) are different.

  Snapshot: monitorings-results-update-desktop.png

Call log:
  - Expect "toHaveScreenshot(monitorings-results-update-desktop.png)" with timeout 5000ms
    - verifying given screenshot expectation
  - taking page screenshot
    - disabled all CSS animations
  - waiting for fonts to load...
  - fonts loaded
  - 354088 pixels (ratio 0.10 of all image pixels) are different.
  - waiting 100ms before taking screenshot
  - taking page screenshot
    - disabled all CSS animations
  - waiting for fonts to load...
  - fonts loaded
  - captured a stable screenshot
  - 354088 pixels (ratio 0.10 of all image pixels) are different.

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
          - group "Informacje o monitoringu" [ref=e91]:
            - generic [ref=e93]:
              - generic [ref=e94]: Nazwa*
              - textbox "Nazwa*" [ref=e95]: Monitoring sądów apelacyjnych
            - generic [ref=e96]:
              - generic [ref=e97]: Opis
              - textbox "Opis" [ref=e98]
          - group "Wyniki monitoringu" [ref=e100]:
            - generic [ref=e102]:
              - generic [ref=e103]: Temat*
              - textbox "Temat*" [ref=e104]: Wniosek o udostępnienie informacji publicznej
            - generic [ref=e105]:
              - generic [ref=e106]: Wyniki
              - application [ref=e107]:
                - generic [ref=e108]:
                  - generic [ref=e109]:
                    - menubar [ref=e111]:
                      - menuitem "Plik" [ref=e112] [cursor=pointer]
                      - menuitem "Edytuj" [ref=e114] [cursor=pointer]
                      - menuitem "Widok" [ref=e116] [cursor=pointer]
                      - menuitem "Wstaw" [ref=e118] [cursor=pointer]
                      - menuitem "Format" [ref=e120] [cursor=pointer]
                      - menuitem "Narzędzia" [ref=e122] [cursor=pointer]
                      - menuitem "Tabela" [ref=e124] [cursor=pointer]
                      - menuitem "Pomoc" [ref=e126] [cursor=pointer]
                    - group [ref=e128]:
                      - group [ref=e129]:
                        - toolbar [ref=e130]:
                          - button "Cofnij" [disabled] [ref=e131]
                          - button "Powtórz" [disabled] [ref=e135]
                        - toolbar [ref=e139]:
                          - button "Wysokość Linii" [ref=e140] [cursor=pointer]
                        - toolbar [ref=e147]:
                          - button "Pogrubienie" [ref=e148] [cursor=pointer]
                          - button "Kursywa" [ref=e152] [cursor=pointer]
                          - button "Kolor tła Czarny" [ref=e156]
                        - toolbar [ref=e164]:
                          - button "Wyrównaj do lewej" [ref=e165] [cursor=pointer]
                          - button "Wyrównaj do środka" [ref=e169] [cursor=pointer]
                          - button "Wyrównaj do prawej" [ref=e173] [cursor=pointer]
                          - button "Wyjustuj" [ref=e177] [cursor=pointer]
                        - toolbar [ref=e181]:
                          - button "Lista wypunktowana" [ref=e182] [cursor=pointer]
                          - button "Lista numerowana" [ref=e186] [cursor=pointer]
                          - button "Zmniejsz wcięcie" [disabled] [ref=e190]
                          - button "Zwiększ wcięcie" [ref=e194] [cursor=pointer]
                        - toolbar [ref=e198]:
                          - button "Znak Specjalny" [ref=e199] [cursor=pointer]
                        - toolbar [ref=e203]:
                          - button "Wyczyść formatowanie" [ref=e204] [cursor=pointer]
                        - toolbar [ref=e208]:
                          - button "Pomoc" [ref=e209] [cursor=pointer]
                  - iframe [ref=e218]:
                    - generic "Obszar tekstu sformatowanego. Naciśnij ALT-0, aby uzyskać pomoc." [ref=f1e1]:
                      - paragraph [ref=f1e2]
                - generic [ref=e219]:
                  - generic [ref=e220]:
                    - navigation [ref=e221]:
                      - button "p" [ref=e222]
                    - generic [ref=e223]: Naciśnij Alt+0, aby uzyskać pomoc
                    - generic [ref=e224]:
                      - button "0 sł." [ref=e225] [cursor=pointer]
                      - link "Build with TinyMCE" [ref=e227]:
                        - /url: https://www.tiny.cloud/powered-by-tiny?utm_campaign=poweredby&utm_source=tiny&utm_medium=referral&utm_content=v7
                        - text: Build with
                  - generic "Naciśnij klawisze strzałek w górę i w dół, aby zmienić rozmiar edytora." [ref=e234]
              - generic [ref=e238]: Wyniki monitoringu i otrzymanych odpowiedzi
        - button "Aktualizuj" [ref=e241] [cursor=pointer]
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
      - link "Historia /monitoringi/monitoring-sadow-apelacyjnych/~results-update" [ref=e264] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e265]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e266]
      - link "Wersje Django 5.2.17" [ref=e267] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e268]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e269]
      - 'link "Czas CPU: 279.10ms (281.58ms)" [ref=e270] [cursor=pointer]':
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
      - link "Zapytania MonitoringResultsUpdateView" [ref=e279] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e280]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e281]
      - link "SQL 7 queries in 2.43ms" [ref=e282] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e283]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e284]
      - link "Pliki statyczne 5 użytych plików" [ref=e285] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e286]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e287]
      - link "Templatki monitorings/monitoring_form.html" [ref=e288] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e289]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e290]
      - link "Alerty" [ref=e291] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e292]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e293]
      - link "Cache 2 wywołania w 0.15ms" [ref=e294] [cursor=pointer]:
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