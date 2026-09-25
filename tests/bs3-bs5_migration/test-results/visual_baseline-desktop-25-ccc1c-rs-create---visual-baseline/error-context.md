# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: visual_baseline.spec.ts >> desktop (2560px) >> letters-create - visual baseline
- Location: tests/bs3-bs5_migration/visual_baseline.spec.ts:10:11

# Error details

```
Error: expect(page).toHaveScreenshot(expected) failed

  Expected an image 2560px by 1465px, received 2560px by 1496px. 356298 pixels (ratio 0.10 of all image pixels) are different.

  Snapshot: letters-create-desktop.png

Call log:
  - Expect "toHaveScreenshot(letters-create-desktop.png)" with timeout 5000ms
    - verifying given screenshot expectation
  - taking page screenshot
    - disabled all CSS animations
  - waiting for fonts to load...
  - fonts loaded
  - Expected an image 2560px by 1465px, received 2560px by 1496px. 356298 pixels (ratio 0.10 of all image pixels) are different.
  - waiting 100ms before taking screenshot
  - taking page screenshot
    - disabled all CSS animations
  - waiting for fonts to load...
  - fonts loaded
  - captured a stable screenshot
  - Expected an image 2560px by 1465px, received 2560px by 1496px. 356298 pixels (ratio 0.10 of all image pixels) are different.

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
        - listitem
        - listitem [ref=e62]: / Szkic
        - listitem [ref=e63]:
          - text: /
          - link:
            - /url: ""
        - listitem [ref=e64]: / Edytuj
      - generic [ref=e66]:
        - heading [level=2] [ref=e68]:
          - text: Nowy list
          - generic [ref=e70]:
            - text: przez
            - link "claude_ai" [ref=e71] [cursor=pointer]:
              - /url: claude_ai
            - text: w sprawie
            - 'link "Monitoring sądów apelacyjnych #1" [ref=e72] [cursor=pointer]':
              - /url: /sprawy/monitoring-sadow-apelacyjnych-1
            - text: z
            - link "Sąd Apelacyjny w Białymstoku" [ref=e73] [cursor=pointer]:
              - /url: /instytucje/sad-apelacyjny-w-bialymstoku
        - generic [ref=e75]:
          - generic [ref=e76]:
            - generic [ref=e77]: Temat*
            - textbox "Temat*" [ref=e78]
          - generic [ref=e79]:
            - generic [ref=e80]: Treść w formacie HTML
            - application [ref=e81]:
              - generic [ref=e82]:
                - generic [ref=e83]:
                  - menubar [ref=e85]:
                    - menuitem "Plik" [ref=e86] [cursor=pointer]
                    - menuitem "Edytuj" [ref=e88] [cursor=pointer]
                    - menuitem "Widok" [ref=e90] [cursor=pointer]
                    - menuitem "Wstaw" [ref=e92] [cursor=pointer]
                    - menuitem "Format" [ref=e94] [cursor=pointer]
                    - menuitem "Narzędzia" [ref=e96] [cursor=pointer]
                    - menuitem "Tabela" [ref=e98] [cursor=pointer]
                    - menuitem "Pomoc" [ref=e100] [cursor=pointer]
                  - group [ref=e102]:
                    - group [ref=e103]:
                      - toolbar [ref=e104]:
                        - button "Cofnij" [disabled] [ref=e105]
                        - button "Powtórz" [disabled] [ref=e109]
                      - toolbar [ref=e113]:
                        - button "Wysokość Linii" [ref=e114] [cursor=pointer]
                      - toolbar [ref=e121]:
                        - button "Pogrubienie" [ref=e122] [cursor=pointer]
                        - button "Kursywa" [ref=e126] [cursor=pointer]
                        - button "Kolor tła Czarny" [ref=e130]
                      - toolbar [ref=e138]:
                        - button "Wyrównaj do lewej" [ref=e139] [cursor=pointer]
                        - button "Wyrównaj do środka" [ref=e143] [cursor=pointer]
                        - button "Wyrównaj do prawej" [ref=e147] [cursor=pointer]
                        - button "Wyjustuj" [ref=e151] [cursor=pointer]
                      - toolbar [ref=e155]:
                        - button "Lista wypunktowana" [ref=e156] [cursor=pointer]
                        - button "Lista numerowana" [ref=e160] [cursor=pointer]
                        - button "Zmniejsz wcięcie" [disabled] [ref=e164]
                        - button "Zwiększ wcięcie" [ref=e168] [cursor=pointer]
                      - toolbar [ref=e172]:
                        - button "Znak Specjalny" [ref=e173] [cursor=pointer]
                      - toolbar [ref=e177]:
                        - button "Wyczyść formatowanie" [ref=e178] [cursor=pointer]
                      - toolbar [ref=e182]:
                        - button "Pomoc" [ref=e183] [cursor=pointer]
                - iframe [ref=e192]:
                  - generic "Obszar tekstu sformatowanego. Naciśnij ALT-0, aby uzyskać pomoc." [ref=f1e1]:
                    - paragraph [ref=f1e2]
                    - paragraph [ref=f1e3]: "Prosimy o odpowiedź na adres {{EMAIL}}"
                    - paragraph [ref=f1e4]: "-----"
              - generic [ref=e193]:
                - generic [ref=e194]:
                  - navigation [ref=e195]:
                    - button "p" [ref=e196]
                  - generic [ref=e197]: Naciśnij Alt+0, aby uzyskać pomoc
                  - generic [ref=e198]:
                    - button "6 sł." [ref=e199] [cursor=pointer]
                    - link "Build with TinyMCE" [ref=e201]:
                      - /url: https://www.tiny.cloud/powered-by-tiny?utm_campaign=poweredby&utm_source=tiny&utm_medium=referral&utm_content=v7
                      - text: Build with
                - generic "Naciśnij klawisze strzałek w górę i w dół, aby zmienić rozmiar edytora." [ref=e208]
          - generic [ref=e212]:
            - generic [ref=e213]: Sprawa*
            - combobox [aria-hidden] [ref=e214]
            - combobox [ref=e217] [cursor=pointer]:
              - 'textbox "Monitoring sądów apelacyjnych #1" [ref=e218]'
          - generic [ref=e219]:
            - generic [ref=e220]: Ocena AI listu
            - combobox "Ocena AI listu" [ref=e221]
          - generic [ref=e222]:
            - generic [ref=e223]: Komentarz od redakcji
            - textbox "Komentarz od redakcji" [ref=e224]
          - button "Zapisz" [ref=e227] [cursor=pointer]
      - generic [ref=e228]:
        - generic [ref=e229]:
          - text: Sieć Obywatelska - Watchdog Polska ul. Szpitalna 5/5 00-031 Warszawa
          - paragraph [ref=e230]:
            - link "Klauzula RODO" [ref=e231] [cursor=pointer]:
              - /url: https://fedrowanie.siecobywatelska.pl/media_internal/tinycontent/uploads/11_KLAUZULA.-.dane.z.udip.na.stronach.www.i.w.mediach.spol.pdf
        - generic [ref=e232]:
          - text: "silnik:"
          - link "jawne.info.pl" [ref=e233] [cursor=pointer]:
            - /url: http://jawne.info.pl
          - text: "| v1.5.77.deps |"
          - link "GitHub" [ref=e234] [cursor=pointer]:
            - /url: https://github.com/watchdogpolska/feder
          - text: –
          - link "efd8a3b" [ref=e236] [cursor=pointer]:
            - /url: https://github.com/watchdogpolska/feder/compare/efd8a3b2...master
          - text: "|"
          - link "API" [ref=e237] [cursor=pointer]:
            - /url: /api/
        - generic [ref=e239]: Ta strona wykorzystuje cookies.
  - list [ref=e241]:
    - listitem [ref=e242]:
      - link "Ukryj »" [ref=e243] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e244]:
      - link "Toggle Theme" [ref=e245] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e248]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e249]
      - link "Historia /listy/~utworz-2684" [ref=e250] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e251]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e252]
      - link "Wersje Django 5.2.17" [ref=e253] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e254]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e255]
      - 'link "Czas CPU: 135.39ms (138.25ms)" [ref=e256] [cursor=pointer]':
        - /url: "#"
    - listitem [ref=e257]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e258]
      - link "Ustawienia" [ref=e259] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e260]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e261]
      - link "Nagłówki" [ref=e262] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e263]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e264]
      - link "Zapytania LetterCreateView" [ref=e265] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e266]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e267]
      - link "SQL 7 queries in 2.67ms" [ref=e268] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e269]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e270]
      - link "Pliki statyczne 12 użytych plików" [ref=e271] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e272]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e273]
      - link "Templatki letters/_letter_reply_body.html" [ref=e274] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e275]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e276]
      - link "Alerty" [ref=e277] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e278]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e279]
      - link "Cache 2 wywołania w 0.24ms" [ref=e280] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e281]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e282]
      - link "Sygnały 88 odbiorców 15 sygnałów" [ref=e283] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e284]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e285]
      - link "Gmina" [ref=e286] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e287]:
      - checkbox "Enable for next and successive requests" [ref=e288]
      - generic [ref=e289]: Przechwycone przekierowania
    - listitem [ref=e290]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e291]
      - link "Profilowanie" [ref=e292] [cursor=pointer]:
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