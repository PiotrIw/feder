# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: visual_baseline.spec.ts >> mobile (375px) >> monitorings-create - visual baseline
- Location: tests/bs3-bs5_migration/visual_baseline.spec.ts:10:11

# Error details

```
Error: expect(page).toHaveScreenshot(expected) failed

  Expected an image 375px by 2389px, received 375px by 2427px. 195029 pixels (ratio 0.22 of all image pixels) are different.

  Snapshot: monitorings-create-mobile.png

Call log:
  - Expect "toHaveScreenshot(monitorings-create-mobile.png)" with timeout 5000ms
    - verifying given screenshot expectation
  - taking page screenshot
    - disabled all CSS animations
  - waiting for fonts to load...
  - fonts loaded
  - Expected an image 375px by 2389px, received 375px by 2427px. 195029 pixels (ratio 0.22 of all image pixels) are different.
  - waiting 100ms before taking screenshot
  - taking page screenshot
    - disabled all CSS animations
  - waiting for fonts to load...
  - fonts loaded
  - captured a stable screenshot
  - Expected an image 375px by 2389px, received 375px by 2427px. 195029 pixels (ratio 0.22 of all image pixels) are different.

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
        - listitem [ref=e70]: Dodaj monitoring
      - heading "Dodaj monitoring" [level=2] [ref=e72]
      - generic [ref=e75]:
        - generic [ref=e76]:
          - group "Monitoring" [ref=e78]:
            - generic [ref=e80]:
              - generic [ref=e81]: Nazwa*
              - textbox "Nazwa*" [ref=e82]
            - generic [ref=e83]:
              - generic [ref=e84]: Opis
              - textbox "Opis" [ref=e85]
            - generic [ref=e87]:
              - checkbox "Czy publicznie widoczny?" [checked] [ref=e88]
              - generic [ref=e89]: Czy publicznie widoczny?
            - generic [ref=e91]:
              - checkbox "Czy ukrywać nowe sprawy przy przypisywaniu?" [ref=e92]
              - generic [ref=e93]: Czy ukrywać nowe sprawy przy przypisywaniu?
            - generic [ref=e95]:
              - checkbox "Korzystaj z LLM" [ref=e96]
              - generic [ref=e97]: Korzystaj z LLM
              - generic [ref=e98]: Przed włączeniem upewnij się, że treść wniosku nie będzie już zmieniana. Zawsze możesz wrócić do edycji i włączyć później.
          - group "Szablon" [ref=e100]:
            - generic [ref=e102]:
              - generic [ref=e103]: Temat*
              - textbox "Temat*" [ref=e104]
            - generic [ref=e105]:
              - generic [ref=e106]: Szablon*
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
                          - button "Odkryj lub ukryj dodatkowe elementy na pasku narzędzi" [ref=e148] [cursor=pointer]
                  - iframe [ref=e154]:
                    - generic "Obszar tekstu sformatowanego. Naciśnij ALT-0, aby uzyskać pomoc." [ref=f1e1]:
                      - paragraph [ref=f1e2]: "Prosimy o odpowiedź na adres {{EMAIL}}"
                - generic [ref=e155]:
                  - generic [ref=e156]:
                    - navigation [ref=e157]:
                      - button "p" [ref=e158]
                    - generic [ref=e159]:
                      - button "6 sł." [ref=e160] [cursor=pointer]
                      - link "Build with TinyMCE" [ref=e162]:
                        - /url: https://www.tiny.cloud/powered-by-tiny?utm_campaign=poweredby&utm_source=tiny&utm_medium=referral&utm_content=v7
                        - text: Build with
                  - generic "Naciśnij klawisze strzałek w górę i w dół, aby zmienić rozmiar edytora." [ref=e169]
              - generic [ref=e173]: "Użyj: {{EMAIL}} aby umieścić adres odpowiedzi, {{ADRESAT}} aby umieścić nazwę adressata."
            - generic [ref=e174]:
              - generic [ref=e175]: Podpis w e-mail*
              - application [ref=e176]:
                - generic [ref=e177]:
                  - generic [ref=e178]:
                    - menubar [ref=e180]:
                      - menuitem "Plik" [ref=e181] [cursor=pointer]
                      - menuitem "Edytuj" [ref=e183] [cursor=pointer]
                      - menuitem "Widok" [ref=e185] [cursor=pointer]
                      - menuitem "Wstaw" [ref=e187] [cursor=pointer]
                      - menuitem "Format" [ref=e189] [cursor=pointer]
                      - menuitem "Narzędzia" [ref=e191] [cursor=pointer]
                      - menuitem "Tabela" [ref=e193] [cursor=pointer]
                      - menuitem "Pomoc" [ref=e195] [cursor=pointer]
                    - group [ref=e197]:
                      - group [ref=e198]:
                        - toolbar [ref=e199]:
                          - button "Cofnij" [disabled] [ref=e200]
                          - button "Powtórz" [disabled] [ref=e204]
                        - toolbar [ref=e208]:
                          - button "Wysokość Linii" [ref=e209] [cursor=pointer]
                        - toolbar [ref=e216]:
                          - button "Odkryj lub ukryj dodatkowe elementy na pasku narzędzi" [ref=e217] [cursor=pointer]
                  - iframe [ref=e223]:
                    - generic "Obszar tekstu sformatowanego. Naciśnij ALT-0, aby uzyskać pomoc." [ref=f2e1]:
                      - paragraph [ref=f2e2]
                - generic [ref=e224]:
                  - generic [ref=e225]:
                    - navigation [ref=e226]:
                      - button "p" [ref=e227]
                    - generic [ref=e228]:
                      - button "0 sł." [ref=e229] [cursor=pointer]
                      - link "Build with TinyMCE" [ref=e231]:
                        - /url: https://www.tiny.cloud/powered-by-tiny?utm_campaign=poweredby&utm_source=tiny&utm_medium=referral&utm_content=v7
                        - text: Build with
                  - generic "Naciśnij klawisze strzałek w górę i w dół, aby zmienić rozmiar edytora." [ref=e238]
              - generic [ref=e242]: Podpis w stopce e-maili, w tym w odpowiedziach na e-maile
            - generic [ref=e243]:
              - generic [ref=e244]: Domain*
              - combobox "Domain*" [ref=e245]:
                - option "---------" [selected]
                - option "fedrowanie.siecobywatelska.pl"
                - option "pokot.pl"
                - option "monitoring.bartoszwilk.pl"
                - option "info.lasyiobywatele.pl"
                - option "nijakowski.pl"
                - option "fedr.uratujzwierze.pl"
                - option "pytania.ofop.eu"
                - option "info.szkolajestnasza.pl"
              - generic [ref=e246]: Domena użyta do wysłania wiadomości
        - button "Zapisz" [ref=e249] [cursor=pointer]
      - generic [ref=e250]:
        - generic [ref=e251]:
          - text: Sieć Obywatelska - Watchdog Polska ul. Szpitalna 5/5 00-031 Warszawa
          - paragraph [ref=e252]:
            - link "Klauzula RODO" [ref=e253] [cursor=pointer]:
              - /url: https://fedrowanie.siecobywatelska.pl/media_internal/tinycontent/uploads/11_KLAUZULA.-.dane.z.udip.na.stronach.www.i.w.mediach.spol.pdf
        - generic [ref=e254]:
          - text: "silnik:"
          - link "jawne.info.pl" [ref=e255] [cursor=pointer]:
            - /url: http://jawne.info.pl
          - text: "| v1.5.77.deps |"
          - link "GitHub" [ref=e256] [cursor=pointer]:
            - /url: https://github.com/watchdogpolska/feder
          - text: –
          - link "efd8a3b" [ref=e258] [cursor=pointer]:
            - /url: https://github.com/watchdogpolska/feder/compare/efd8a3b2...master
          - text: "|"
          - link "API" [ref=e259] [cursor=pointer]:
            - /url: /api/
        - generic [ref=e261]: Ta strona wykorzystuje cookies.
  - list [ref=e263]:
    - listitem [ref=e264]:
      - link "Ukryj »" [ref=e265] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e266]:
      - link "Toggle Theme" [ref=e267] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e270]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e271]
      - link "Historia /monitoringi/~utworz" [ref=e272] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e273]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e274]
      - link "Wersje Django 5.2.17" [ref=e275] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e276]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e277]
      - 'link "Czas CPU: 307.74ms (305.83ms)" [ref=e278] [cursor=pointer]':
        - /url: "#"
    - listitem [ref=e279]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e280]
      - link "Ustawienia" [ref=e281] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e282]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e283]
      - link "Nagłówki" [ref=e284] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e285]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e286]
      - link "Zapytania MonitoringCreateView" [ref=e287] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e288]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e289]
      - link "SQL 5 queries in 1.53ms" [ref=e290] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e291]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e292]
      - link "Pliki statyczne 5 użytych plików" [ref=e293] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e294]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e295]
      - link "Templatki monitorings/monitoring_form.html" [ref=e296] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e297]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e298]
      - link "Alerty" [ref=e299] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e300]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e301]
      - link "Cache 2 wywołania w 0.12ms" [ref=e302] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e303]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e304]
      - link "Sygnały 88 odbiorców 15 sygnałów" [ref=e305] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e306]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e307]
      - link "Gmina" [ref=e308] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e309]:
      - checkbox "Enable for next and successive requests" [ref=e310]
      - generic [ref=e311]: Przechwycone przekierowania
    - listitem [ref=e312]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e313]
      - link "Profilowanie" [ref=e314] [cursor=pointer]:
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