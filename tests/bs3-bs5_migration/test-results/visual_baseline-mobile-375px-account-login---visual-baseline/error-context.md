# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: visual_baseline.spec.ts >> mobile (375px) >> account-login - visual baseline
- Location: tests/bs3-bs5_migration/visual_baseline.spec.ts:10:11

# Error details

```
Error: expect(page).toHaveScreenshot(expected) failed

  171689 pixels (ratio 0.57 of all image pixels) are different.

  Snapshot: account-login-mobile.png

Call log:
  - Expect "toHaveScreenshot(account-login-mobile.png)" with timeout 5000ms
    - verifying given screenshot expectation
  - taking page screenshot
    - disabled all CSS animations
  - waiting for fonts to load...
  - fonts loaded
  - 171689 pixels (ratio 0.57 of all image pixels) are different.
  - waiting 100ms before taking screenshot
  - taking page screenshot
    - disabled all CSS animations
  - waiting for fonts to load...
  - fonts loaded
  - captured a stable screenshot
  - 171689 pixels (ratio 0.57 of all image pixels) are different.

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
      - generic [ref=e69]:
        - heading "claude_ai" [level=2] [ref=e72]
        - generic [ref=e74]:
          - link "O mnie" [ref=e75] [cursor=pointer]:
            - /url: /uzytkownik/~aktualizuj/
          - link "E-mail" [ref=e76] [cursor=pointer]:
            - /url: /accounts/email/
          - link "Zmiana hasła" [ref=e77] [cursor=pointer]:
            - /url: /accounts/password/change/
          - link "Połącz konto Google" [ref=e78] [cursor=pointer]:
            - /url: /accounts/google/login/?process=connect
      - generic [ref=e80]:
        - generic [ref=e81]:
          - text: Sieć Obywatelska - Watchdog Polska ul. Szpitalna 5/5 00-031 Warszawa
          - paragraph [ref=e82]:
            - link "Klauzula RODO" [ref=e83] [cursor=pointer]:
              - /url: https://fedrowanie.siecobywatelska.pl/media_internal/tinycontent/uploads/11_KLAUZULA.-.dane.z.udip.na.stronach.www.i.w.mediach.spol.pdf
        - generic [ref=e84]:
          - text: "silnik:"
          - link "jawne.info.pl" [ref=e85] [cursor=pointer]:
            - /url: http://jawne.info.pl
          - text: "| v1.5.77.deps |"
          - link "GitHub" [ref=e86] [cursor=pointer]:
            - /url: https://github.com/watchdogpolska/feder
          - text: –
          - link "efd8a3b" [ref=e88] [cursor=pointer]:
            - /url: https://github.com/watchdogpolska/feder/compare/efd8a3b2...master
          - text: "|"
          - link "API" [ref=e89] [cursor=pointer]:
            - /url: /api/
        - generic [ref=e91]: Ta strona wykorzystuje cookies.
  - list [ref=e93]:
    - listitem [ref=e94]:
      - link "Ukryj »" [ref=e95] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e96]:
      - link "Toggle Theme" [ref=e97] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e100]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e101]
      - link "Historia /uzytkownik/claude_ai/" [ref=e102] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e103]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e104]
      - link "Wersje Django 5.2.17" [ref=e105] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e106]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e107]
      - 'link "Czas CPU: 76.88ms (83.06ms)" [ref=e108] [cursor=pointer]':
        - /url: "#"
    - listitem [ref=e109]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e110]
      - link "Ustawienia" [ref=e111] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e112]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e113]
      - link "Nagłówki" [ref=e114] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e115]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e116]
      - link "Zapytania UserDetailView" [ref=e117] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e118]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e119]
      - link "SQL 7 queries in 3.01ms" [ref=e120] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e121]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e122]
      - link "Pliki statyczne 3 użyte plików" [ref=e123] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e124]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e125]
      - link "Templatki users/user_detail.html" [ref=e126] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e127]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e128]
      - link "Alerty" [ref=e129] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e130]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e131]
      - link "Cache 2 wywołania w 0.19ms" [ref=e132] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e133]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e134]
      - link "Sygnały 88 odbiorców 15 sygnałów" [ref=e135] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e136]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e137]
      - link "Gmina" [ref=e138] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e139]:
      - checkbox "Enable for next and successive requests" [ref=e140]
      - generic [ref=e141]: Przechwycone przekierowania
    - listitem [ref=e142]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e143]
      - link "Profilowanie" [ref=e144] [cursor=pointer]:
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