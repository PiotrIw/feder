# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: visual_baseline.spec.ts >> desktop (2560px) >> jst-list - visual baseline
- Location: tests/bs3-bs5_migration/visual_baseline.spec.ts:10:11

# Error details

```
Error: expect(page).toHaveScreenshot(expected) failed

  336815 pixels (ratio 0.10 of all image pixels) are different.

  Snapshot: jst-list-desktop.png

Call log:
  - Expect "toHaveScreenshot(jst-list-desktop.png)" with timeout 5000ms
    - verifying given screenshot expectation
  - taking page screenshot
    - disabled all CSS animations
  - waiting for fonts to load...
  - fonts loaded
  - 336815 pixels (ratio 0.10 of all image pixels) are different.
  - waiting 100ms before taking screenshot
  - taking page screenshot
    - disabled all CSS animations
  - waiting for fonts to load...
  - fonts loaded
  - captured a stable screenshot
  - 336815 pixels (ratio 0.10 of all image pixels) are different.

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
        - heading "Województwa" [level=2] [ref=e62]
        - paragraph [ref=e63]: Wybierz województwo
        - list [ref=e64]:
          - listitem [ref=e65]:
            - link "Dolnośląskie" [ref=e66] [cursor=pointer]:
              - /url: /jst/dolnoslaskie
          - listitem [ref=e67]:
            - link "Kujawsko-Pomorskie" [ref=e68] [cursor=pointer]:
              - /url: /jst/kujawsko-pomorskie
          - listitem [ref=e69]:
            - link "Lubelskie" [ref=e70] [cursor=pointer]:
              - /url: /jst/lubelskie
          - listitem [ref=e71]:
            - link "Lubuskie" [ref=e72] [cursor=pointer]:
              - /url: /jst/lubuskie
          - listitem [ref=e73]:
            - link "Łódzkie" [ref=e74] [cursor=pointer]:
              - /url: /jst/lodzkie
          - listitem [ref=e75]:
            - link "Małopolskie" [ref=e76] [cursor=pointer]:
              - /url: /jst/malopolskie
          - listitem [ref=e77]:
            - link "Mazowieckie" [ref=e78] [cursor=pointer]:
              - /url: /jst/mazowieckie
          - listitem [ref=e79]:
            - link "Opolskie" [ref=e80] [cursor=pointer]:
              - /url: /jst/opolskie
          - listitem [ref=e81]:
            - link "Podkarpackie" [ref=e82] [cursor=pointer]:
              - /url: /jst/podkarpackie
          - listitem [ref=e83]:
            - link "Podlaskie" [ref=e84] [cursor=pointer]:
              - /url: /jst/podlaskie
          - listitem [ref=e85]:
            - link "Pomorskie" [ref=e86] [cursor=pointer]:
              - /url: /jst/pomorskie
          - listitem [ref=e87]:
            - link "Śląskie" [ref=e88] [cursor=pointer]:
              - /url: /jst/slaskie
          - listitem [ref=e89]:
            - link "Świętokrzyskie" [ref=e90] [cursor=pointer]:
              - /url: /jst/swietokrzyskie
          - listitem [ref=e91]:
            - link "Warmińsko-Mazurskie" [ref=e92] [cursor=pointer]:
              - /url: /jst/warminsko-mazurskie
          - listitem [ref=e93]:
            - link "Wielkopolskie" [ref=e94] [cursor=pointer]:
              - /url: /jst/wielkopolskie
          - listitem [ref=e95]:
            - link "Zachodniopomorskie" [ref=e96] [cursor=pointer]:
              - /url: /jst/zachodniopomorskie
      - generic [ref=e97]:
        - generic [ref=e98]:
          - text: Sieć Obywatelska - Watchdog Polska ul. Szpitalna 5/5 00-031 Warszawa
          - paragraph [ref=e99]:
            - link "Klauzula RODO" [ref=e100] [cursor=pointer]:
              - /url: https://fedrowanie.siecobywatelska.pl/media_internal/tinycontent/uploads/11_KLAUZULA.-.dane.z.udip.na.stronach.www.i.w.mediach.spol.pdf
        - generic [ref=e101]:
          - text: "silnik:"
          - link "jawne.info.pl" [ref=e102] [cursor=pointer]:
            - /url: http://jawne.info.pl
          - text: "| v1.5.77.deps |"
          - link "GitHub" [ref=e103] [cursor=pointer]:
            - /url: https://github.com/watchdogpolska/feder
          - text: –
          - link "efd8a3b" [ref=e105] [cursor=pointer]:
            - /url: https://github.com/watchdogpolska/feder/compare/efd8a3b2...master
          - text: "|"
          - link "API" [ref=e106] [cursor=pointer]:
            - /url: /api/
        - generic [ref=e108]: Ta strona wykorzystuje cookies.
  - list [ref=e110]:
    - listitem [ref=e111]:
      - link "Ukryj »" [ref=e112] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e113]:
      - link "Toggle Theme" [ref=e114] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e117]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e118]
      - link "Historia /jst/" [ref=e119] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e120]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e121]
      - link "Wersje Django 5.2.17" [ref=e122] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e123]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e124]
      - 'link "Czas CPU: 63.25ms (65.24ms)" [ref=e125] [cursor=pointer]':
        - /url: "#"
    - listitem [ref=e126]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e127]
      - link "Ustawienia" [ref=e128] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e129]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e130]
      - link "Nagłówki" [ref=e131] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e132]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e133]
      - link "Zapytania JSTListView" [ref=e134] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e135]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e136]
      - link "SQL 5 queries in 1.78ms" [ref=e137] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e138]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e139]
      - link "Pliki statyczne 3 użyte plików" [ref=e140] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e141]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e142]
      - link "Templatki teryt/jst_list.html" [ref=e143] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e144]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e145]
      - link "Alerty" [ref=e146] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e147]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e148]
      - link "Cache 2 wywołania w 0.17ms" [ref=e149] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e150]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e151]
      - link "Sygnały 88 odbiorców 15 sygnałów" [ref=e152] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e153]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e154]
      - link "Gmina" [ref=e155] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e156]:
      - checkbox "Enable for next and successive requests" [ref=e157]
      - generic [ref=e158]: Przechwycone przekierowania
    - listitem [ref=e159]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e160]
      - link "Profilowanie" [ref=e161] [cursor=pointer]:
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