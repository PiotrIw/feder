# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: visual_baseline.spec.ts >> desktop (2560px) >> letters-update - visual baseline
- Location: tests/bs3-bs5_migration/visual_baseline.spec.ts:10:11

# Error details

```
Error: expect(page).toHaveScreenshot(expected) failed

  Expected an image 2560px by 1527px, received 2560px by 1540px. 390800 pixels (ratio 0.10 of all image pixels) are different.

  Snapshot: letters-update-desktop.png

Call log:
  - Expect "toHaveScreenshot(letters-update-desktop.png)" with timeout 5000ms
    - verifying given screenshot expectation
  - taking page screenshot
    - disabled all CSS animations
  - waiting for fonts to load...
  - fonts loaded
  - Expected an image 2560px by 1527px, received 2560px by 1540px. 390800 pixels (ratio 0.10 of all image pixels) are different.
  - waiting 100ms before taking screenshot
  - taking page screenshot
    - disabled all CSS animations
  - waiting for fonts to load...
  - fonts loaded
  - captured a stable screenshot
  - Expected an image 2560px by 1527px, received 2560px by 1540px. 390800 pixels (ratio 0.10 of all image pixels) are different.

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
        - listitem [ref=e64]:
          - text: /
          - 'link "Monitoring sądów apelacyjnych #1" [ref=e65] [cursor=pointer]':
            - /url: /sprawy/monitoring-sadow-apelacyjnych-1
        - listitem [ref=e66]:
          - text: /
          - link "Wniosek o udostępnienie informacji publicznej" [ref=e67] [cursor=pointer]:
            - /url: /listy/7302
        - listitem [ref=e68]: / Edytuj
      - generic [ref=e72]:
        - link "Edytuj" [ref=e73] [cursor=pointer]:
          - /url: /listy/7302/~edytuj
        - link "Usuń" [ref=e74] [cursor=pointer]:
          - /url: /listy/7302/~usun
        - link "Odpowiedź" [ref=e75] [cursor=pointer]:
          - /url: /listy/7302/~odpowiedz
        - link "Wyślij ponownie" [ref=e77] [cursor=pointer]:
          - /url: /listy/7302/~resend
      - generic [ref=e80]:
        - heading [level=2] [ref=e82]:
          - link "Wniosek o udostępnienie informacji publicznej" [ref=e84] [cursor=pointer]:
            - /url: /listy/7302
          - generic [ref=e85]:
            - text: przez
            - link "adobrawy" [ref=e86] [cursor=pointer]:
              - /url: /uzytkownik/adobrawy/
            - time [ref=e87]: 11 sierpnia 2017 02:48
            - text: w sprawie
            - 'link "Monitoring sądów apelacyjnych #1" [ref=e88] [cursor=pointer]':
              - /url: /sprawy/monitoring-sadow-apelacyjnych-1
            - text: z
            - link "Sąd Apelacyjny w Białymstoku" [ref=e89] [cursor=pointer]:
              - /url: /instytucje/sad-apelacyjny-w-bialymstoku
        - generic [ref=e91]:
          - generic [ref=e92]:
            - generic [ref=e93]: Temat*
            - textbox "Temat*" [ref=e94]: Wniosek o udostępnienie informacji publicznej
          - generic [ref=e95]:
            - generic [ref=e96]: Treść w formacie HTML*
            - iframe [ref=e97]:
              - paragraph [ref=f1e2]: "Stowarzyszenie Sieć Obywatelska Watchdog Polska wnosi o udostępnienie poprzez przesłanie następujących informacji: - adresu strony internetowej, na której znajdują się orzeczenia dyscyplinarne, wydane wobec sędziów przez tutejszy Sąd, - rejestru umów, zawartych w imieniu Sądu od 1 stycznia 2017 r. do 31 lipca 2017 r., zawierającego informacje w zakresie co najmniej dat zawartych umów, przedmiotach umów, stronach umów, kwotach umów, - adresu strony internetowej, na której znajduje się dokumentacja przebiegu i efektów kontroli, przeprowadzonych w sądzie, oraz wystąpienia, stanowiska, wnioski i opinie podmiotów ją przeprowadzających, - kalendarz spotkań prezesa (prezes) sądu, które odbył (odbyła) w lipcu 2017 r., - skany orzeczeń, zapadłych w wyniku wniesienia środka odwoławczego od orzeczenia sądu niższej instancji, zapadłego z oskarżenia o czyn zabroniony art. 231 Kodeksu karnego - wydanych przez Sąd w 2017 r., - skany orzeczeń, zapadłych w wyniku wniesienia środka odwoławczego od orzeczenia sądu niższej instancji, zapadłego z oskarżenia o czyn zabroniony art. 212 Kodeksu karnego - wydanych przez Sąd w 2017 r. Stowarzyszenie wnosi o udostępnienie wskazanych informacji w formie elektronicznej, na adres e-mail {{EMAIL}}. Katarzyna Batko-Tołuć, Bartosz Wilk - członkowie zarządu, zgodnie z zasadami reprezentacji"
          - generic [ref=e98]:
            - generic [ref=e99]: Sprawa*
            - combobox [aria-hidden] [ref=e100]
            - combobox [ref=e103] [cursor=pointer]:
              - 'textbox "Monitoring sądów apelacyjnych #1" [ref=e104]'
          - generic [ref=e105]:
            - generic [ref=e106]: Ocena AI listu
            - combobox "Ocena AI listu" [ref=e107]:
              - option "None" [selected]
              - option "A) email jest odpowiedzią z Sąd Apelacyjny w Białymstoku i zawiera odpowiedzi na pytania z wniosku o informację publiczną."
              - option "B) email jest odpowiedzią z Sąd Apelacyjny w Białymstoku i zawiera odmowę odpowiedzi na pytania z wniosku o informację publiczną."
              - option "C) email jest odpowiedzią z Sąd Apelacyjny w Białymstoku i zawiera informację o przedłużeniu terminu na odpowiedź."
              - option "D) email jest potwierdzeniem dostarczenia lub otwarcia maila z Sąd Apelacyjny w Białymstoku i nie zawiera odpowiedzi na pytania z wniosku o informację publiczną."
              - option "E) email jest odpowiedzią z innej instytucji lub na inny wniosek."
              - option "F) email nie jest odpowiedzią z Sąd Apelacyjny w Białymstoku i jest spamem."
              - option "G) nie można ustalić kategorii odpowiedzi."
          - generic [ref=e108]:
            - generic [ref=e109]: Komentarz od redakcji
            - textbox "Komentarz od redakcji" [ref=e110]
          - button "Aktualizuj" [ref=e113] [cursor=pointer]
      - generic [ref=e114]:
        - generic [ref=e115]:
          - text: Sieć Obywatelska - Watchdog Polska ul. Szpitalna 5/5 00-031 Warszawa
          - paragraph [ref=e116]:
            - link "Klauzula RODO" [ref=e117] [cursor=pointer]:
              - /url: https://fedrowanie.siecobywatelska.pl/media_internal/tinycontent/uploads/11_KLAUZULA.-.dane.z.udip.na.stronach.www.i.w.mediach.spol.pdf
        - generic [ref=e118]:
          - text: "silnik:"
          - link "jawne.info.pl" [ref=e119] [cursor=pointer]:
            - /url: http://jawne.info.pl
          - text: "| v1.5.77.deps |"
          - link "GitHub" [ref=e120] [cursor=pointer]:
            - /url: https://github.com/watchdogpolska/feder
          - text: –
          - link "efd8a3b" [ref=e122] [cursor=pointer]:
            - /url: https://github.com/watchdogpolska/feder/compare/efd8a3b2...master
          - text: "|"
          - link "API" [ref=e123] [cursor=pointer]:
            - /url: /api/
        - generic [ref=e125]: Ta strona wykorzystuje cookies.
  - list [ref=e127]:
    - listitem [ref=e128]:
      - link "Ukryj »" [ref=e129] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e130]:
      - link "Toggle Theme" [ref=e131] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e134]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e135]
      - link "Historia /listy/7302/~edytuj" [ref=e136] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e137]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e138]
      - link "Wersje Django 5.2.17" [ref=e139] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e140]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e141]
      - 'link "Czas CPU: 140.27ms (144.37ms)" [ref=e142] [cursor=pointer]':
        - /url: "#"
    - listitem [ref=e143]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e144]
      - link "Ustawienia" [ref=e145] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e146]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e147]
      - link "Nagłówki" [ref=e148] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e149]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e150]
      - link "Zapytania LetterUpdateView" [ref=e151] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e152]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e153]
      - link "SQL 13 queries in 4.98ms" [ref=e154] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e155]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e156]
      - link "Pliki statyczne 10 użytych plików" [ref=e157] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e158]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e159]
      - link "Templatki letters/letter_form.html" [ref=e160] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e161]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e162]
      - link "Alerty" [ref=e163] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e164]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e165]
      - link "Cache 2 wywołania w 0.12ms" [ref=e166] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e167]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e168]
      - link "Sygnały 88 odbiorców 15 sygnałów" [ref=e169] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e170]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e171]
      - link "Gmina" [ref=e172] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e173]:
      - checkbox "Enable for next and successive requests" [ref=e174]
      - generic [ref=e175]: Przechwycone przekierowania
    - listitem [ref=e176]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e177]
      - link "Profilowanie" [ref=e178] [cursor=pointer]:
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