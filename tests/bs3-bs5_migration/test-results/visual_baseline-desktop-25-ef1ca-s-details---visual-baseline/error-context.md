# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: visual_baseline.spec.ts >> desktop (2560px) >> institutions-details - visual baseline
- Location: tests/bs3-bs5_migration/visual_baseline.spec.ts:10:11

# Error details

```
Error: expect(page).toHaveScreenshot(expected) failed

  377002 pixels (ratio 0.11 of all image pixels) are different.

  Snapshot: institutions-details-desktop.png

Call log:
  - Expect "toHaveScreenshot(institutions-details-desktop.png)" with timeout 5000ms
    - verifying given screenshot expectation
  - taking page screenshot
    - disabled all CSS animations
  - waiting for fonts to load...
  - fonts loaded
  - 377002 pixels (ratio 0.11 of all image pixels) are different.
  - waiting 100ms before taking screenshot
  - taking page screenshot
    - disabled all CSS animations
  - waiting for fonts to load...
  - fonts loaded
  - captured a stable screenshot
  - 377002 pixels (ratio 0.11 of all image pixels) are different.

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
          - link "Wielkopolskie" [ref=e63] [cursor=pointer]:
            - /url: /jst/wielkopolskie
        - listitem [ref=e64]:
          - text: /
          - link "Czarnkowsko-Trzcianecki" [ref=e65] [cursor=pointer]:
            - /url: /jst/czarnkowsko-trzcianecki
        - listitem [ref=e66]:
          - text: /
          - link "Trzcianka" [ref=e67] [cursor=pointer]:
            - /url: /jst/trzcianka-2
        - listitem [ref=e68]: / ,,KOMBUD” Sp. z o. o.
      - generic [ref=e70]:
        - link "Edytuj" [ref=e71] [cursor=pointer]:
          - /url: /instytucje/kombud-sp-z-o-o/~edytuj
        - link "Usuń" [ref=e72] [cursor=pointer]:
          - /url: /instytucje/kombud-sp-z-o-o/~usun
      - heading ",,KOMBUD” Sp. z o. o." [level=2] [ref=e74]
      - generic [ref=e76]:
        - generic [ref=e78]:
          - paragraph [ref=e79]: "E-mail: kombud@trzcianka.com.pl"
          - paragraph [ref=e80]: "Archiwalne:"
          - paragraph [ref=e82]:
            - text: "Tagi:"
            - link "spółki komunalne" [ref=e83] [cursor=pointer]:
              - /url: /instytucje/?tags=158
        - generic [ref=e84]:
          - heading "Sprawy" [level=2] [ref=e85]
          - generic [ref=e86]:
            - heading [level=3] [ref=e87]:
              - link "Jawność w spółkach komunalnych" [ref=e89] [cursor=pointer]:
                - /url: /monitoringi/jawnosc-w-spolkach-komunalnych
            - heading [level=5] [ref=e92]:
              - 'link "Jawność w spółkach komunalnych #31" [ref=e94] [cursor=pointer]':
                - /url: /sprawy/jawnosc-w-spolkach-komunalnych-31
      - generic [ref=e95]:
        - generic [ref=e96]:
          - text: Sieć Obywatelska - Watchdog Polska ul. Szpitalna 5/5 00-031 Warszawa
          - paragraph [ref=e97]:
            - link "Klauzula RODO" [ref=e98] [cursor=pointer]:
              - /url: https://fedrowanie.siecobywatelska.pl/media_internal/tinycontent/uploads/11_KLAUZULA.-.dane.z.udip.na.stronach.www.i.w.mediach.spol.pdf
        - generic [ref=e99]:
          - text: "silnik:"
          - link "jawne.info.pl" [ref=e100] [cursor=pointer]:
            - /url: http://jawne.info.pl
          - text: "| v1.5.77.deps |"
          - link "GitHub" [ref=e101] [cursor=pointer]:
            - /url: https://github.com/watchdogpolska/feder
          - text: –
          - link "efd8a3b" [ref=e103] [cursor=pointer]:
            - /url: https://github.com/watchdogpolska/feder/compare/efd8a3b2...master
          - text: "|"
          - link "API" [ref=e104] [cursor=pointer]:
            - /url: /api/
        - generic [ref=e106]: Ta strona wykorzystuje cookies.
  - list [ref=e108]:
    - listitem [ref=e109]:
      - link "Ukryj »" [ref=e110] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e111]:
      - link "Toggle Theme" [ref=e112] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e115]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e116]
      - link "Historia /instytucje/kombud-sp-z-o-o" [ref=e117] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e118]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e119]
      - link "Wersje Django 5.2.17" [ref=e120] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e121]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e122]
      - 'link "Czas CPU: 101.11ms (106.68ms)" [ref=e123] [cursor=pointer]':
        - /url: "#"
    - listitem [ref=e124]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e125]
      - link "Ustawienia" [ref=e126] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e127]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e128]
      - link "Nagłówki" [ref=e129] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e130]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e131]
      - link "Zapytania InstitutionDetailView" [ref=e132] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e133]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e134]
      - link "SQL 10 queries in 5.85ms" [ref=e135] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e136]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e137]
      - link "Pliki statyczne 3 użyte plików" [ref=e138] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e139]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e140]
      - link "Templatki institutions/institution_detail.html" [ref=e141] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e142]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e143]
      - link "Alerty" [ref=e144] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e145]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e146]
      - link "Cache 2 wywołania w 0.14ms" [ref=e147] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e148]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e149]
      - link "Sygnały 88 odbiorców 15 sygnałów" [ref=e150] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e151]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e152]
      - link "Gmina" [ref=e153] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e154]:
      - checkbox "Enable for next and successive requests" [ref=e155]
      - generic [ref=e156]: Przechwycone przekierowania
    - listitem [ref=e157]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e158]
      - link "Profilowanie" [ref=e159] [cursor=pointer]:
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