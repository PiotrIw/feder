# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: visual_baseline.spec.ts >> mobile (375px) >> letters-details - visual baseline
- Location: tests/bs3-bs5_migration/visual_baseline.spec.ts:10:11

# Error details

```
Error: expect(page).toHaveScreenshot(expected) failed

  Expected an image 386px by 1748px, received 383px by 1856px. 221365 pixels (ratio 0.31 of all image pixels) are different.

  Snapshot: letters-details-mobile.png

Call log:
  - Expect "toHaveScreenshot(letters-details-mobile.png)" with timeout 5000ms
    - verifying given screenshot expectation
  - taking page screenshot
    - disabled all CSS animations
  - waiting for fonts to load...
  - fonts loaded
  - Expected an image 386px by 1748px, received 383px by 1856px. 221365 pixels (ratio 0.31 of all image pixels) are different.
  - waiting 100ms before taking screenshot
  - taking page screenshot
    - disabled all CSS animations
  - waiting for fonts to load...
  - fonts loaded
  - captured a stable screenshot
  - Expected an image 386px by 1748px, received 383px by 1856px. 221365 pixels (ratio 0.31 of all image pixels) are different.

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
        - listitem [ref=e72]:
          - text: /
          - 'link "Monitoring sądów apelacyjnych #1" [ref=e73] [cursor=pointer]':
            - /url: /sprawy/monitoring-sadow-apelacyjnych-1
        - listitem [ref=e74]: / Wniosek o udostępnienie informacji publicznej
      - generic [ref=e78]:
        - link "Edytuj" [ref=e79] [cursor=pointer]:
          - /url: /listy/7302/~edytuj
        - link "Usuń" [ref=e80] [cursor=pointer]:
          - /url: /listy/7302/~usun
        - link "Odpowiedź" [ref=e81] [cursor=pointer]:
          - /url: /listy/7302/~odpowiedz
        - link "Wyślij ponownie" [ref=e83] [cursor=pointer]:
          - /url: /listy/7302/~resend
      - generic [ref=e86]:
        - heading [level=2] [ref=e88]:
          - link "Wniosek o udostępnienie informacji publicznej" [ref=e90] [cursor=pointer]:
            - /url: /listy/7302
          - generic [ref=e91]:
            - text: przez
            - link "adobrawy" [ref=e92] [cursor=pointer]:
              - /url: /uzytkownik/adobrawy/
            - time [ref=e93]: 11 sierpnia 2017 02:48
            - text: w sprawie
            - 'link "Monitoring sądów apelacyjnych #1" [ref=e94] [cursor=pointer]':
              - /url: /sprawy/monitoring-sadow-apelacyjnych-1
            - text: z
            - link "Sąd Apelacyjny w Białymstoku" [ref=e95] [cursor=pointer]:
              - /url: /instytucje/sad-apelacyjny-w-bialymstoku
        - generic [ref=e96]:
          - button "Przełącz metadane M" [ref=e98] [cursor=pointer]:
            - generic [ref=e99]: Przełącz metadane
            - text: M
          - generic [ref=e103]:
            - paragraph [ref=e104]: "Stowarzyszenie Sieć Obywatelska Watchdog Polska wnosi o udostępnienie poprzez przesłanie następujących informacji:"
            - paragraph [ref=e105]: "- adresu strony internetowej, na której znajdują się orzeczenia dyscyplinarne, wydane wobec sędziów przez tutejszy Sąd,- rejestru umów, zawartych w imieniu Sądu od 1 stycznia 2017 r. do 31 lipca 2017 r., zawierającego informacje w zakresie co najmniej dat zawartych umów, przedmiotach umów, stronach umów, kwotach umów,- adresu strony internetowej, na której znajduje się dokumentacja przebiegu i efektów kontroli, przeprowadzonych w sądzie, oraz wystąpienia, stanowiska, wnioski i opinie podmiotów ją przeprowadzających,- kalendarz spotkań prezesa (prezes) sądu, które odbył (odbyła) w lipcu 2017 r.,- skany orzeczeń, zapadłych w wyniku wniesienia środka odwoławczego od orzeczenia sądu niższej instancji, zapadłego z oskarżenia o czyn zabroniony art. 231 Kodeksu karnego - wydanych przez Sąd w 2017 r.,- skany orzeczeń, zapadłych w wyniku wniesienia środka odwoławczego od orzeczenia sądu niższej instancji, zapadłego z oskarżenia o czyn zabroniony art. 212 Kodeksu karnego - wydanych przez Sąd w 2017 r."
            - paragraph [ref=e106]: "Stowarzyszenie wnosi o udostępnienie wskazanych informacji w formie elektronicznej, na adres e-mail {{EMAIL}}."
            - paragraph [ref=e107]: Katarzyna Batko-Tołuć, Bartosz Wilk - członkowie zarządu, zgodnie z zasadami reprezentacji
          - link "Pobierz list" [ref=e108] [cursor=pointer]:
            - /url: /listy/7302-msg
      - generic [ref=e110]:
        - generic [ref=e111]:
          - text: Sieć Obywatelska - Watchdog Polska ul. Szpitalna 5/5 00-031 Warszawa
          - paragraph [ref=e112]:
            - link "Klauzula RODO" [ref=e113] [cursor=pointer]:
              - /url: https://fedrowanie.siecobywatelska.pl/media_internal/tinycontent/uploads/11_KLAUZULA.-.dane.z.udip.na.stronach.www.i.w.mediach.spol.pdf
        - generic [ref=e114]:
          - text: "silnik:"
          - link "jawne.info.pl" [ref=e115] [cursor=pointer]:
            - /url: http://jawne.info.pl
          - text: "| v1.5.77.deps |"
          - link "GitHub" [ref=e116] [cursor=pointer]:
            - /url: https://github.com/watchdogpolska/feder
          - text: –
          - link "efd8a3b" [ref=e118] [cursor=pointer]:
            - /url: https://github.com/watchdogpolska/feder/compare/efd8a3b2...master
          - text: "|"
          - link "API" [ref=e119] [cursor=pointer]:
            - /url: /api/
        - generic [ref=e121]: Ta strona wykorzystuje cookies.
  - list [ref=e123]:
    - listitem [ref=e124]:
      - link "Ukryj »" [ref=e125] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e126]:
      - link "Toggle Theme" [ref=e127] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e130]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e131]
      - link "Historia /listy/7302" [ref=e132] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e133]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e134]
      - link "Wersje Django 5.2.17" [ref=e135] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e136]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e137]
      - 'link "Czas CPU: 102.17ms (105.47ms)" [ref=e138] [cursor=pointer]':
        - /url: "#"
    - listitem [ref=e139]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e140]
      - link "Ustawienia" [ref=e141] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e142]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e143]
      - link "Nagłówki" [ref=e144] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e145]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e146]
      - link "Zapytania LetterDetailView" [ref=e147] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e148]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e149]
      - link "SQL 10 queries in 3.78ms" [ref=e150] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e151]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e152]
      - link "Pliki statyczne 3 użyte plików" [ref=e153] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e154]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e155]
      - link "Templatki letters/letter_detail.html" [ref=e156] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e157]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e158]
      - link "Alerty" [ref=e159] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e160]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e161]
      - link "Cache 2 wywołania w 0.13ms" [ref=e162] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e163]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e164]
      - link "Sygnały 88 odbiorców 15 sygnałów" [ref=e165] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e166]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e167]
      - link "Gmina" [ref=e168] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e169]:
      - checkbox "Enable for next and successive requests" [ref=e170]
      - generic [ref=e171]: Przechwycone przekierowania
    - listitem [ref=e172]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e173]
      - link "Profilowanie" [ref=e174] [cursor=pointer]:
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