# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: visual_baseline.spec.ts >> desktop (2560px) >> letters-assign - visual baseline
- Location: tests/bs3-bs5_migration/visual_baseline.spec.ts:10:11

# Error details

```
Error: expect(page).toHaveScreenshot(expected) failed

  395955 pixels (ratio 0.11 of all image pixels) are different.

  Snapshot: letters-assign-desktop.png

Call log:
  - Expect "toHaveScreenshot(letters-assign-desktop.png)" with timeout 5000ms
    - verifying given screenshot expectation
  - taking page screenshot
    - disabled all CSS animations
  - waiting for fonts to load...
  - fonts loaded
  - 395955 pixels (ratio 0.11 of all image pixels) are different.
  - waiting 100ms before taking screenshot
  - taking page screenshot
    - disabled all CSS animations
  - waiting for fonts to load...
  - fonts loaded
  - captured a stable screenshot
  - 395955 pixels (ratio 0.11 of all image pixels) are different.

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
      - generic [ref=e61]:
        - heading [level=2] [ref=e63]:
          - link "Wniosek o udostępnienie informacji publicznej" [ref=e65] [cursor=pointer]:
            - /url: /listy/7302
          - time [ref=e67]: 11 sierpnia 2017 02:48
        - generic [ref=e68]:
          - paragraph [ref=e69]: "Stowarzyszenie Sieć Obywatelska Watchdog Polska wnosi o udostępnienie poprzez przesłanie następujących informacji:"
          - paragraph [ref=e70]: "- adresu strony internetowej, na której znajdują się orzeczenia dyscyplinarne, wydane wobec sędziów przez tutejszy Sąd,- rejestru umów, zawartych w imieniu Sądu od 1 stycznia 2017 r. do 31 lipca 2017 r., zawierającego informacje w zakresie co najmniej dat zawartych umów, przedmiotach umów, stronach umów, kwotach umów,- adresu strony internetowej, na której znajduje się dokumentacja przebiegu i efektów kontroli, przeprowadzonych w sądzie, oraz wystąpienia, stanowiska, wnioski i opinie podmiotów ją przeprowadzających,- kalendarz spotkań prezesa (prezes) sądu, które odbył (odbyła) w lipcu 2017 r.,- skany orzeczeń, zapadłych w wyniku wniesienia środka odwoławczego od orzeczenia sądu niższej instancji, zapadłego z oskarżenia o czyn zabroniony art. 231 Kodeksu karnego - wydanych przez Sąd w 2017 r.,- skany orzeczeń, zapadłych w wyniku wniesienia środka odwoławczego od orzeczenia sądu niższej instancji, zapadłego z oskarżenia o czyn zabroniony art. 212 Kodeksu karnego - wydanych przez Sąd w 2017 r."
          - paragraph [ref=e71]: "Stowarzyszenie wnosi o udostępnienie wskazanych informacji w formie elektronicznej, na adres e-mail {{EMAIL}}."
          - paragraph [ref=e72]: Katarzyna Batko-Tołuć, Bartosz Wilk - członkowie zarządu, zgodnie z zasadami reprezentacji
        - generic [ref=e73]:
          - generic [ref=e74]:
            - generic [ref=e75]:
              - generic [ref=e76]:
                - generic [ref=e77]: Numer sprawy*
                - combobox [aria-hidden] [ref=e78]
                - combobox [ref=e81] [cursor=pointer]:
                  - textbox
              - button "Przypisz" [ref=e84] [cursor=pointer]
            - link "Pobierz list" [ref=e85] [cursor=pointer]:
              - /url: /listy/7302-msg
          - link "Oznacz spam" [ref=e88] [cursor=pointer]:
            - /url: /listy/7302/~oznacz-spam
      - generic [ref=e90]:
        - generic [ref=e91]:
          - text: Sieć Obywatelska - Watchdog Polska ul. Szpitalna 5/5 00-031 Warszawa
          - paragraph [ref=e92]:
            - link "Klauzula RODO" [ref=e93] [cursor=pointer]:
              - /url: https://fedrowanie.siecobywatelska.pl/media_internal/tinycontent/uploads/11_KLAUZULA.-.dane.z.udip.na.stronach.www.i.w.mediach.spol.pdf
        - generic [ref=e94]:
          - text: "silnik:"
          - link "jawne.info.pl" [ref=e95] [cursor=pointer]:
            - /url: http://jawne.info.pl
          - text: "| v1.5.77.deps |"
          - link "GitHub" [ref=e96] [cursor=pointer]:
            - /url: https://github.com/watchdogpolska/feder
          - text: –
          - link "efd8a3b" [ref=e98] [cursor=pointer]:
            - /url: https://github.com/watchdogpolska/feder/compare/efd8a3b2...master
          - text: "|"
          - link "API" [ref=e99] [cursor=pointer]:
            - /url: /api/
        - generic [ref=e101]: Ta strona wykorzystuje cookies.
  - list [ref=e103]:
    - listitem [ref=e104]:
      - link "Ukryj »" [ref=e105] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e106]:
      - link "Toggle Theme" [ref=e107] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e110]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e111]
      - link "Historia /listy/~przypisz-7302" [ref=e112] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e113]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e114]
      - link "Wersje Django 5.2.17" [ref=e115] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e116]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e117]
      - 'link "Czas CPU: 284.01ms (281.58ms)" [ref=e118] [cursor=pointer]':
        - /url: "#"
    - listitem [ref=e119]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e120]
      - link "Ustawienia" [ref=e121] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e122]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e123]
      - link "Nagłówki" [ref=e124] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e125]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e126]
      - link "Zapytania AssignLetterFormView" [ref=e127] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e128]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e129]
      - link "SQL 10 queries in 2.98ms" [ref=e130] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e131]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e132]
      - link "Pliki statyczne 10 użytych plików" [ref=e133] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e134]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e135]
      - link "Templatki letters/letter_assign.html" [ref=e136] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e137]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e138]
      - link "Alerty" [ref=e139] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e140]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e141]
      - link "Cache 2 wywołania w 0.14ms" [ref=e142] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e143]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e144]
      - link "Sygnały 88 odbiorców 15 sygnałów" [ref=e145] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e146]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e147]
      - link "Gmina" [ref=e148] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e149]:
      - checkbox "Enable for next and successive requests" [ref=e150]
      - generic [ref=e151]: Przechwycone przekierowania
    - listitem [ref=e152]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e153]
      - link "Profilowanie" [ref=e154] [cursor=pointer]:
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