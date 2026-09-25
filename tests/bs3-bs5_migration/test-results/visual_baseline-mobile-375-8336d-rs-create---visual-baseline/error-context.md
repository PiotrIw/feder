# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: visual_baseline.spec.ts >> mobile (375px) >> letters-create - visual baseline
- Location: tests/bs3-bs5_migration/visual_baseline.spec.ts:10:11

# Error details

```
Error: expect(page).toHaveScreenshot(expected) failed

  Expected an image 386px by 1615px, received 383px by 1658px. 181572 pixels (ratio 0.29 of all image pixels) are different.

  Snapshot: letters-create-mobile.png

Call log:
  - Expect "toHaveScreenshot(letters-create-mobile.png)" with timeout 5000ms
    - verifying given screenshot expectation
  - taking page screenshot
    - disabled all CSS animations
  - waiting for fonts to load...
  - fonts loaded
  - Expected an image 386px by 1615px, received 383px by 1658px. 181572 pixels (ratio 0.29 of all image pixels) are different.
  - waiting 100ms before taking screenshot
  - taking page screenshot
    - disabled all CSS animations
  - waiting for fonts to load...
  - fonts loaded
  - captured a stable screenshot
  - Expected an image 386px by 1615px, received 383px by 1658px. 181572 pixels (ratio 0.29 of all image pixels) are different.

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
        - listitem
        - listitem [ref=e70]: / Szkic
        - listitem [ref=e71]:
          - text: /
          - link:
            - /url: ""
        - listitem [ref=e72]: / Edytuj
      - generic [ref=e74]:
        - heading [level=2] [ref=e76]:
          - text: Nowy list
          - generic [ref=e78]:
            - text: przez
            - link "claude_ai" [ref=e79] [cursor=pointer]:
              - /url: claude_ai
            - text: w sprawie
            - 'link "Monitoring sądów apelacyjnych #1" [ref=e80] [cursor=pointer]':
              - /url: /sprawy/monitoring-sadow-apelacyjnych-1
            - text: z
            - link "Sąd Apelacyjny w Białymstoku" [ref=e81] [cursor=pointer]:
              - /url: /instytucje/sad-apelacyjny-w-bialymstoku
        - generic [ref=e83]:
          - generic [ref=e84]:
            - generic [ref=e85]: Temat*
            - textbox "Temat*" [ref=e86]
          - generic [ref=e87]:
            - generic [ref=e88]: Treść w formacie HTML
            - application [ref=e89]:
              - generic [ref=e90]:
                - generic [ref=e91]:
                  - menubar [ref=e93]:
                    - menuitem "Plik" [ref=e94] [cursor=pointer]
                    - menuitem "Edytuj" [ref=e96] [cursor=pointer]
                    - menuitem "Widok" [ref=e98] [cursor=pointer]
                    - menuitem "Wstaw" [ref=e100] [cursor=pointer]
                    - menuitem "Format" [ref=e102] [cursor=pointer]
                    - menuitem "Narzędzia" [ref=e104] [cursor=pointer]
                    - menuitem "Tabela" [ref=e106] [cursor=pointer]
                    - menuitem "Pomoc" [ref=e108] [cursor=pointer]
                  - group [ref=e110]:
                    - group [ref=e111]:
                      - toolbar [ref=e112]:
                        - button "Cofnij" [disabled] [ref=e113]
                        - button "Powtórz" [disabled] [ref=e117]
                      - toolbar [ref=e121]:
                        - button "Wysokość Linii" [ref=e122] [cursor=pointer]
                      - toolbar [ref=e129]:
                        - button "Odkryj lub ukryj dodatkowe elementy na pasku narzędzi" [ref=e130] [cursor=pointer]
                - iframe [ref=e136]:
                  - generic "Obszar tekstu sformatowanego. Naciśnij ALT-0, aby uzyskać pomoc." [ref=f1e1]:
                    - paragraph [ref=f1e2]
                    - paragraph [ref=f1e3]: "Prosimy o odpowiedź na adres {{EMAIL}}"
                    - paragraph [ref=f1e4]: "-----"
              - generic [ref=e137]:
                - generic [ref=e138]:
                  - navigation [ref=e139]:
                    - button "p" [ref=e140]
                  - generic [ref=e141]:
                    - button "6 sł." [ref=e142] [cursor=pointer]
                    - link "Build with TinyMCE" [ref=e144]:
                      - /url: https://www.tiny.cloud/powered-by-tiny?utm_campaign=poweredby&utm_source=tiny&utm_medium=referral&utm_content=v7
                      - text: Build with
                - generic "Naciśnij klawisze strzałek w górę i w dół, aby zmienić rozmiar edytora." [ref=e151]
          - generic [ref=e155]:
            - generic [ref=e156]: Sprawa*
            - combobox [aria-hidden] [ref=e157]
            - combobox [ref=e160] [cursor=pointer]:
              - 'textbox "Monitoring sądów apelacyjnych #1" [ref=e161]'
          - generic [ref=e162]:
            - generic [ref=e163]: Ocena AI listu
            - combobox "Ocena AI listu" [ref=e164]
          - generic [ref=e165]:
            - generic [ref=e166]: Komentarz od redakcji
            - textbox "Komentarz od redakcji" [ref=e167]
          - button "Zapisz" [ref=e170] [cursor=pointer]
      - generic [ref=e171]:
        - generic [ref=e172]:
          - text: Sieć Obywatelska - Watchdog Polska ul. Szpitalna 5/5 00-031 Warszawa
          - paragraph [ref=e173]:
            - link "Klauzula RODO" [ref=e174] [cursor=pointer]:
              - /url: https://fedrowanie.siecobywatelska.pl/media_internal/tinycontent/uploads/11_KLAUZULA.-.dane.z.udip.na.stronach.www.i.w.mediach.spol.pdf
        - generic [ref=e175]:
          - text: "silnik:"
          - link "jawne.info.pl" [ref=e176] [cursor=pointer]:
            - /url: http://jawne.info.pl
          - text: "| v1.5.77.deps |"
          - link "GitHub" [ref=e177] [cursor=pointer]:
            - /url: https://github.com/watchdogpolska/feder
          - text: –
          - link "efd8a3b" [ref=e179] [cursor=pointer]:
            - /url: https://github.com/watchdogpolska/feder/compare/efd8a3b2...master
          - text: "|"
          - link "API" [ref=e180] [cursor=pointer]:
            - /url: /api/
        - generic [ref=e182]: Ta strona wykorzystuje cookies.
  - list [ref=e184]:
    - listitem [ref=e185]:
      - link "Ukryj »" [ref=e186] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e187]:
      - link "Toggle Theme" [ref=e188] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e191]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e192]
      - link "Historia /listy/~utworz-2684" [ref=e193] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e194]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e195]
      - link "Wersje Django 5.2.17" [ref=e196] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e197]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e198]
      - 'link "Czas CPU: 119.34ms (121.71ms)" [ref=e199] [cursor=pointer]':
        - /url: "#"
    - listitem [ref=e200]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e201]
      - link "Ustawienia" [ref=e202] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e203]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e204]
      - link "Nagłówki" [ref=e205] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e206]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e207]
      - link "Zapytania LetterCreateView" [ref=e208] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e209]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e210]
      - link "SQL 7 queries in 2.30ms" [ref=e211] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e212]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e213]
      - link "Pliki statyczne 12 użytych plików" [ref=e214] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e215]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e216]
      - link "Templatki letters/_letter_reply_body.html" [ref=e217] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e218]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e219]
      - link "Alerty" [ref=e220] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e221]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e222]
      - link "Cache 2 wywołania w 0.23ms" [ref=e223] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e224]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e225]
      - link "Sygnały 88 odbiorców 15 sygnałów" [ref=e226] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e227]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e228]
      - link "Gmina" [ref=e229] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e230]:
      - checkbox "Enable for next and successive requests" [ref=e231]
      - generic [ref=e232]: Przechwycone przekierowania
    - listitem [ref=e233]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e234]
      - link "Profilowanie" [ref=e235] [cursor=pointer]:
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