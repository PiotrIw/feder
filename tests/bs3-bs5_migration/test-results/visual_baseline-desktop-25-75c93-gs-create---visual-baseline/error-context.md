# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: visual_baseline.spec.ts >> desktop (2560px) >> monitorings-create - visual baseline
- Location: tests/bs3-bs5_migration/visual_baseline.spec.ts:10:11

# Error details

```
Error: expect(page).toHaveScreenshot(expected) failed

  Expected an image 2560px by 1676px, received 2560px by 1698px. 359399 pixels (ratio 0.09 of all image pixels) are different.

  Snapshot: monitorings-create-desktop.png

Call log:
  - Expect "toHaveScreenshot(monitorings-create-desktop.png)" with timeout 5000ms
    - verifying given screenshot expectation
  - taking page screenshot
    - disabled all CSS animations
  - waiting for fonts to load...
  - fonts loaded
  - Expected an image 2560px by 1676px, received 2560px by 1698px. 359399 pixels (ratio 0.09 of all image pixels) are different.
  - waiting 100ms before taking screenshot
  - taking page screenshot
    - disabled all CSS animations
  - waiting for fonts to load...
  - fonts loaded
  - captured a stable screenshot
  - Expected an image 2560px by 1676px, received 2560px by 1698px. 359399 pixels (ratio 0.09 of all image pixels) are different.

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
        - listitem [ref=e62]: Dodaj monitoring
      - heading "Dodaj monitoring" [level=2] [ref=e64]
      - generic [ref=e67]:
        - generic [ref=e68]:
          - group "Monitoring" [ref=e70]:
            - generic [ref=e72]:
              - generic [ref=e73]: Nazwa*
              - textbox "Nazwa*" [ref=e74]
            - generic [ref=e75]:
              - generic [ref=e76]: Opis
              - textbox "Opis" [ref=e77]
            - generic [ref=e79]:
              - checkbox "Czy publicznie widoczny?" [checked] [ref=e80]
              - generic [ref=e81]: Czy publicznie widoczny?
            - generic [ref=e83]:
              - checkbox "Czy ukrywać nowe sprawy przy przypisywaniu?" [ref=e84]
              - generic [ref=e85]: Czy ukrywać nowe sprawy przy przypisywaniu?
            - generic [ref=e87]:
              - checkbox "Korzystaj z LLM" [ref=e88]
              - generic [ref=e89]: Korzystaj z LLM
              - generic [ref=e90]: Przed włączeniem upewnij się, że treść wniosku nie będzie już zmieniana. Zawsze możesz wrócić do edycji i włączyć później.
          - group "Szablon" [ref=e92]:
            - generic [ref=e94]:
              - generic [ref=e95]: Temat*
              - textbox "Temat*" [ref=e96]
            - generic [ref=e97]:
              - generic [ref=e98]: Szablon*
              - application [ref=e99]:
                - generic [ref=e100]:
                  - generic [ref=e101]:
                    - menubar [ref=e103]:
                      - menuitem "Plik" [ref=e104] [cursor=pointer]
                      - menuitem "Edytuj" [ref=e106] [cursor=pointer]
                      - menuitem "Widok" [ref=e108] [cursor=pointer]
                      - menuitem "Wstaw" [ref=e110] [cursor=pointer]
                      - menuitem "Format" [ref=e112] [cursor=pointer]
                      - menuitem "Narzędzia" [ref=e114] [cursor=pointer]
                      - menuitem "Tabela" [ref=e116] [cursor=pointer]
                      - menuitem "Pomoc" [ref=e118] [cursor=pointer]
                    - group [ref=e120]:
                      - group [ref=e121]:
                        - toolbar [ref=e122]:
                          - button "Cofnij" [disabled] [ref=e123]
                          - button "Powtórz" [disabled] [ref=e127]
                        - toolbar [ref=e131]:
                          - button "Wysokość Linii" [ref=e132] [cursor=pointer]
                        - toolbar [ref=e139]:
                          - button "Pogrubienie" [ref=e140] [cursor=pointer]
                          - button "Kursywa" [ref=e144] [cursor=pointer]
                          - button "Kolor tła Czarny" [ref=e148]
                        - toolbar [ref=e156]:
                          - button "Wyrównaj do lewej" [ref=e157] [cursor=pointer]
                          - button "Wyrównaj do środka" [ref=e161] [cursor=pointer]
                          - button "Wyrównaj do prawej" [ref=e165] [cursor=pointer]
                          - button "Wyjustuj" [ref=e169] [cursor=pointer]
                        - toolbar [ref=e173]:
                          - button "Lista wypunktowana" [ref=e174] [cursor=pointer]
                          - button "Lista numerowana" [ref=e178] [cursor=pointer]
                          - button "Zmniejsz wcięcie" [disabled] [ref=e182]
                          - button "Zwiększ wcięcie" [ref=e186] [cursor=pointer]
                        - toolbar [ref=e190]:
                          - button "Znak Specjalny" [ref=e191] [cursor=pointer]
                        - toolbar [ref=e195]:
                          - button "Wyczyść formatowanie" [ref=e196] [cursor=pointer]
                        - toolbar [ref=e200]:
                          - button "Pomoc" [ref=e201] [cursor=pointer]
                  - iframe [ref=e210]:
                    - generic "Obszar tekstu sformatowanego. Naciśnij ALT-0, aby uzyskać pomoc." [ref=f1e1]:
                      - paragraph [ref=f1e2]: "Prosimy o odpowiedź na adres {{EMAIL}}"
                - generic [ref=e211]:
                  - generic [ref=e212]:
                    - navigation [ref=e213]:
                      - button "p" [ref=e214]
                    - generic [ref=e215]: Naciśnij Alt+0, aby uzyskać pomoc
                    - generic [ref=e216]:
                      - button "6 sł." [ref=e217] [cursor=pointer]
                      - link "Build with TinyMCE" [ref=e219]:
                        - /url: https://www.tiny.cloud/powered-by-tiny?utm_campaign=poweredby&utm_source=tiny&utm_medium=referral&utm_content=v7
                        - text: Build with
                  - generic "Naciśnij klawisze strzałek w górę i w dół, aby zmienić rozmiar edytora." [ref=e226]
              - generic [ref=e230]: "Użyj: {{EMAIL}} aby umieścić adres odpowiedzi, {{ADRESAT}} aby umieścić nazwę adressata."
            - generic [ref=e231]:
              - generic [ref=e232]: Podpis w e-mail*
              - application [ref=e233]:
                - generic [ref=e234]:
                  - generic [ref=e235]:
                    - menubar [ref=e237]:
                      - menuitem "Plik" [ref=e238] [cursor=pointer]
                      - menuitem "Edytuj" [ref=e240] [cursor=pointer]
                      - menuitem "Widok" [ref=e242] [cursor=pointer]
                      - menuitem "Wstaw" [ref=e244] [cursor=pointer]
                      - menuitem "Format" [ref=e246] [cursor=pointer]
                      - menuitem "Narzędzia" [ref=e248] [cursor=pointer]
                      - menuitem "Tabela" [ref=e250] [cursor=pointer]
                      - menuitem "Pomoc" [ref=e252] [cursor=pointer]
                    - group [ref=e254]:
                      - group [ref=e255]:
                        - toolbar [ref=e256]:
                          - button "Cofnij" [disabled] [ref=e257]
                          - button "Powtórz" [disabled] [ref=e261]
                        - toolbar [ref=e265]:
                          - button "Wysokość Linii" [ref=e266] [cursor=pointer]
                        - toolbar [ref=e273]:
                          - button "Pogrubienie" [ref=e274] [cursor=pointer]
                          - button "Kursywa" [ref=e278] [cursor=pointer]
                          - button "Kolor tła Czarny" [ref=e282]
                        - toolbar [ref=e290]:
                          - button "Wyrównaj do lewej" [ref=e291] [cursor=pointer]
                          - button "Wyrównaj do środka" [ref=e295] [cursor=pointer]
                          - button "Wyrównaj do prawej" [ref=e299] [cursor=pointer]
                          - button "Wyjustuj" [ref=e303] [cursor=pointer]
                        - toolbar [ref=e307]:
                          - button "Lista wypunktowana" [ref=e308] [cursor=pointer]
                          - button "Lista numerowana" [ref=e312] [cursor=pointer]
                          - button "Zmniejsz wcięcie" [disabled] [ref=e316]
                          - button "Zwiększ wcięcie" [ref=e320] [cursor=pointer]
                        - toolbar [ref=e324]:
                          - button "Znak Specjalny" [ref=e325] [cursor=pointer]
                        - toolbar [ref=e329]:
                          - button "Wyczyść formatowanie" [ref=e330] [cursor=pointer]
                        - toolbar [ref=e334]:
                          - button "Pomoc" [ref=e335] [cursor=pointer]
                  - iframe [ref=e344]:
                    - generic "Obszar tekstu sformatowanego. Naciśnij ALT-0, aby uzyskać pomoc." [ref=f2e1]:
                      - paragraph [ref=f2e2]
                - generic [ref=e345]:
                  - generic [ref=e346]:
                    - navigation [ref=e347]:
                      - button "p" [ref=e348]
                    - generic [ref=e349]: Naciśnij Alt+0, aby uzyskać pomoc
                    - generic [ref=e350]:
                      - button "0 sł." [ref=e351] [cursor=pointer]
                      - link "Build with TinyMCE" [ref=e353]:
                        - /url: https://www.tiny.cloud/powered-by-tiny?utm_campaign=poweredby&utm_source=tiny&utm_medium=referral&utm_content=v7
                        - text: Build with
                  - generic "Naciśnij klawisze strzałek w górę i w dół, aby zmienić rozmiar edytora." [ref=e360]
              - generic [ref=e364]: Podpis w stopce e-maili, w tym w odpowiedziach na e-maile
            - generic [ref=e365]:
              - generic [ref=e366]: Domain*
              - combobox "Domain*" [ref=e367]:
                - option "---------" [selected]
                - option "fedrowanie.siecobywatelska.pl"
                - option "pokot.pl"
                - option "monitoring.bartoszwilk.pl"
                - option "info.lasyiobywatele.pl"
                - option "nijakowski.pl"
                - option "fedr.uratujzwierze.pl"
                - option "pytania.ofop.eu"
                - option "info.szkolajestnasza.pl"
              - generic [ref=e368]: Domena użyta do wysłania wiadomości
        - button "Zapisz" [ref=e371] [cursor=pointer]
      - generic [ref=e372]:
        - generic [ref=e373]:
          - text: Sieć Obywatelska - Watchdog Polska ul. Szpitalna 5/5 00-031 Warszawa
          - paragraph [ref=e374]:
            - link "Klauzula RODO" [ref=e375] [cursor=pointer]:
              - /url: https://fedrowanie.siecobywatelska.pl/media_internal/tinycontent/uploads/11_KLAUZULA.-.dane.z.udip.na.stronach.www.i.w.mediach.spol.pdf
        - generic [ref=e376]:
          - text: "silnik:"
          - link "jawne.info.pl" [ref=e377] [cursor=pointer]:
            - /url: http://jawne.info.pl
          - text: "| v1.5.77.deps |"
          - link "GitHub" [ref=e378] [cursor=pointer]:
            - /url: https://github.com/watchdogpolska/feder
          - text: –
          - link "efd8a3b" [ref=e380] [cursor=pointer]:
            - /url: https://github.com/watchdogpolska/feder/compare/efd8a3b2...master
          - text: "|"
          - link "API" [ref=e381] [cursor=pointer]:
            - /url: /api/
        - generic [ref=e383]: Ta strona wykorzystuje cookies.
  - list [ref=e385]:
    - listitem [ref=e386]:
      - link "Ukryj »" [ref=e387] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e388]:
      - link "Toggle Theme" [ref=e389] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e392]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e393]
      - link "Historia /monitoringi/~utworz" [ref=e394] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e395]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e396]
      - link "Wersje Django 5.2.17" [ref=e397] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e398]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e399]
      - 'link "Czas CPU: 181.74ms (173.43ms)" [ref=e400] [cursor=pointer]':
        - /url: "#"
    - listitem [ref=e401]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e402]
      - link "Ustawienia" [ref=e403] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e404]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e405]
      - link "Nagłówki" [ref=e406] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e407]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e408]
      - link "Zapytania MonitoringCreateView" [ref=e409] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e410]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e411]
      - link "SQL 5 queries in 1.91ms" [ref=e412] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e413]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e414]
      - link "Pliki statyczne 5 użytych plików" [ref=e415] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e416]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e417]
      - link "Templatki monitorings/monitoring_form.html" [ref=e418] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e419]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e420]
      - link "Alerty" [ref=e421] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e422]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e423]
      - link "Cache 2 wywołania w 0.17ms" [ref=e424] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e425]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e426]
      - link "Sygnały 88 odbiorców 15 sygnałów" [ref=e427] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e428]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e429]
      - link "Gmina" [ref=e430] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e431]:
      - checkbox "Enable for next and successive requests" [ref=e432]
      - generic [ref=e433]: Przechwycone przekierowania
    - listitem [ref=e434]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e435]
      - link "Profilowanie" [ref=e436] [cursor=pointer]:
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