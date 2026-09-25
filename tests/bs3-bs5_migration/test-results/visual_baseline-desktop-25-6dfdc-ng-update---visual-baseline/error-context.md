# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: visual_baseline.spec.ts >> desktop (2560px) >> parcels-outgoing-update - visual baseline
- Location: tests/bs3-bs5_migration/visual_baseline.spec.ts:10:11

# Error details

```
Error: expect(page).toHaveScreenshot(expected) failed

  365209 pixels (ratio 0.10 of all image pixels) are different.

  Snapshot: parcels-outgoing-update-desktop.png

Call log:
  - Expect "toHaveScreenshot(parcels-outgoing-update-desktop.png)" with timeout 5000ms
    - verifying given screenshot expectation
  - taking page screenshot
    - disabled all CSS animations
  - waiting for fonts to load...
  - fonts loaded
  - 365209 pixels (ratio 0.10 of all image pixels) are different.
  - waiting 100ms before taking screenshot
  - taking page screenshot
    - disabled all CSS animations
  - waiting for fonts to load...
  - fonts loaded
  - captured a stable screenshot
  - 365209 pixels (ratio 0.10 of all image pixels) are different.

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
          - link "Żywienie w szpitalach" [ref=e63] [cursor=pointer]:
            - /url: /monitoringi/zywienie-w-szpitalach
        - listitem [ref=e64]:
          - text: /
          - 'link "Żywienie w szpitalach #576" [ref=e65] [cursor=pointer]':
            - /url: /sprawy/zywienie-w-szpitalach-576
        - listitem [ref=e66]:
          - text: /
          - link "Centrum Medyczne Ujastek w Krakowie (żywienie w szpitalach) - wniosek epuap30.10.2019" [ref=e67] [cursor=pointer]:
            - /url: /przesylki/outgoing-1
        - listitem [ref=e68]: / Zaktualizujs wychodzącą przesyłkę pocztową
      - generic [ref=e70]:
        - link "Edytuj" [ref=e71] [cursor=pointer]:
          - /url: /przesylki/outgoing-1/~update
        - link "Usuń" [ref=e72] [cursor=pointer]:
          - /url: /przesylki/outgoing-1/~delete
      - heading [level=2] [ref=e74]:
        - link "Centrum Medyczne Ujastek w Krakowie (żywienie w szpitalach) - wniosek epuap30.10.2019" [ref=e76] [cursor=pointer]:
          - /url: /przesylki/outgoing-1
      - generic [ref=e78]:
        - generic [ref=e79]:
          - generic [ref=e80]: Tytuł*
          - textbox "Tytuł*" [ref=e81]: Centrum Medyczne Ujastek w Krakowie (żywienie w szpitalach) - wniosek epuap30.10.2019
        - generic [ref=e82]:
          - generic [ref=e83]: Treść*
          - generic [ref=e84]:
            - generic [ref=e85]: Teraz
            - link "Centrum_Medyczne_Ujastek_w_Krakowie_żywienie_w_szpitalach__-_wniosek_epuap30.10.2019.pdf" [ref=e88] [cursor=pointer]:
              - /url: /media/Centrum_Medyczne_Ujastek_w_Krakowie_%C5%BCywienie_w_szpitalach__-_wniosek_epuap30.10.2019.pdf
          - button "Treść*" [ref=e90] [cursor=pointer]
        - generic [ref=e91]:
          - generic [ref=e92]: Adresat*
          - combobox [aria-hidden] [ref=e93]
          - combobox [ref=e96] [cursor=pointer]:
            - textbox "CENTRUM MEDYCZNE UJASTEK SPÓŁKA Z OGRANICZONĄ ODPOWIEDZIALNOŚCIĄ" [ref=e97]
        - generic [ref=e98]:
          - generic [ref=e99]: Data wysłania*
          - textbox "Data wysłania*" [ref=e100]: 30.10.2019
        - button "Aktualizuj" [ref=e103] [cursor=pointer]
      - generic [ref=e104]:
        - generic [ref=e105]:
          - text: Sieć Obywatelska - Watchdog Polska ul. Szpitalna 5/5 00-031 Warszawa
          - paragraph [ref=e106]:
            - link "Klauzula RODO" [ref=e107] [cursor=pointer]:
              - /url: https://fedrowanie.siecobywatelska.pl/media_internal/tinycontent/uploads/11_KLAUZULA.-.dane.z.udip.na.stronach.www.i.w.mediach.spol.pdf
        - generic [ref=e108]:
          - text: "silnik:"
          - link "jawne.info.pl" [ref=e109] [cursor=pointer]:
            - /url: http://jawne.info.pl
          - text: "| v1.5.77.deps |"
          - link "GitHub" [ref=e110] [cursor=pointer]:
            - /url: https://github.com/watchdogpolska/feder
          - text: –
          - link "efd8a3b" [ref=e112] [cursor=pointer]:
            - /url: https://github.com/watchdogpolska/feder/compare/efd8a3b2...master
          - text: "|"
          - link "API" [ref=e113] [cursor=pointer]:
            - /url: /api/
        - generic [ref=e115]: Ta strona wykorzystuje cookies.
  - list [ref=e117]:
    - listitem [ref=e118]:
      - link "Ukryj »" [ref=e119] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e120]:
      - link "Toggle Theme" [ref=e121] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e124]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e125]
      - link "Historia /przesylki/outgoing-1/~update" [ref=e126] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e127]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e128]
      - link "Wersje Django 5.2.17" [ref=e129] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e130]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e131]
      - 'link "Czas CPU: 124.59ms (127.20ms)" [ref=e132] [cursor=pointer]':
        - /url: "#"
    - listitem [ref=e133]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e134]
      - link "Ustawienia" [ref=e135] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e136]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e137]
      - link "Nagłówki" [ref=e138] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e139]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e140]
      - link "Zapytania OutgoingParcelPostUpdateView" [ref=e141] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e142]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e143]
      - link "SQL 11 queries in 3.57ms" [ref=e144] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e145]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e146]
      - link "Pliki statyczne 10 użytych plików" [ref=e147] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e148]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e149]
      - link "Templatki parcels/outgoingparcelpost_form.html" [ref=e150] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e151]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e152]
      - link "Alerty" [ref=e153] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e154]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e155]
      - link "Cache 2 wywołania w 0.20ms" [ref=e156] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e157]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e158]
      - link "Sygnały 88 odbiorców 15 sygnałów" [ref=e159] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e160]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e161]
      - link "Gmina" [ref=e162] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e163]:
      - checkbox "Enable for next and successive requests" [ref=e164]
      - generic [ref=e165]: Przechwycone przekierowania
    - listitem [ref=e166]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e167]
      - link "Profilowanie" [ref=e168] [cursor=pointer]:
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