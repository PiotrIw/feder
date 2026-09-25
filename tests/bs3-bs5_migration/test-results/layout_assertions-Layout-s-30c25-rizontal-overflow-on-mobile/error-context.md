# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: layout_assertions.spec.ts >> Layout sanity - mobile >> monitorings-details - no horizontal overflow on mobile
- Location: tests/bs3-bs5_migration/layout_assertions.spec.ts:30:9

# Error details

```
Error: Mobile layout has horizontal scroll

expect(received).toBeLessThanOrEqual(expected)

Expected: <= 20
Received:    105
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
        - listitem [ref=e70]: Monitoring sądów apelacyjnych
      - generic [ref=e72]:
        - link "Edytuj" [ref=e73] [cursor=pointer]:
          - /url: /monitoringi/monitoring-sadow-apelacyjnych/~edytuj
        - link "Aktualizuj wyniki" [ref=e74] [cursor=pointer]:
          - /url: /monitoringi/monitoring-sadow-apelacyjnych/~results-update
        - link "Przypisz" [ref=e75] [cursor=pointer]:
          - /url: /monitoringi/monitoring-sadow-apelacyjnych/~przypisz
        - link "Usuń" [ref=e76] [cursor=pointer]:
          - /url: /monitoringi/monitoring-sadow-apelacyjnych/~usun
        - link "Utwórz sprawę" [ref=e77] [cursor=pointer]:
          - /url: /sprawy/~utworz-5
        - link "Wiadomość masowa" [ref=e78] [cursor=pointer]:
          - /url: /monitoringi/monitoring-sadow-apelacyjnych/~wiadomosc-masowa
        - link "Uprawnienia" [ref=e79] [cursor=pointer]:
          - /url: /monitoringi/monitoring-sadow-apelacyjnych/~uprawnienia
        - link "Lista alertów" [ref=e80] [cursor=pointer]:
          - /url: /alerty/monitoring-5
        - link "Zobacz dzienniki" [ref=e81] [cursor=pointer]:
          - /url: /listy/logi/monitoring-5
        - link "Zobacz tagi" [ref=e83] [cursor=pointer]:
          - /url: /sprawy/tagi/monitoring-5
        - link "Zobacz raport" [ref=e85] [cursor=pointer]:
          - /url: /monitoringi/monitoring-sadow-apelacyjnych/raport
        - link "Zobacz tabelę spraw" [ref=e87] [cursor=pointer]:
          - /url: /monitoringi/monitoring-sadow-apelacyjnych/monitoring_cases_table
      - heading [level=2] [ref=e90]:
        - text: Monitoring sądów apelacyjnych
        - generic [ref=e92]:
          - text: przez
          - link "adobrawy" [ref=e93] [cursor=pointer]:
            - /url: /uzytkownik/adobrawy/
          - time [ref=e94]: 11 sierpnia 2017 02:47
      - generic [ref=e95]:
        - table [ref=e98]:
          - rowgroup [ref=e99]:
            - row [ref=e100]:
              - columnheader "Województwo" [ref=e101]
              - columnheader "Liczba spraw" [ref=e102]
              - columnheader "Liczba spraw z potw. odbioru" [ref=e103]
              - columnheader "Liczba spraw z odpowiedzią" [ref=e104]
            - row [ref=e105]:
              - cell "Dolnośląskie" [ref=e106]
              - cell "1" [ref=e107]
              - cell "0" [ref=e108]
              - cell "1" [ref=e109]
            - row [ref=e110]:
              - cell "Kujawsko-Pomorskie" [ref=e111]
              - cell "0" [ref=e112]
              - cell "0" [ref=e113]
              - cell "0" [ref=e114]
            - row [ref=e115]:
              - cell "Lubelskie" [ref=e116]
              - cell "1" [ref=e117]
              - cell "0" [ref=e118]
              - cell "1" [ref=e119]
            - row [ref=e120]:
              - cell "Lubuskie" [ref=e121]
              - cell "0" [ref=e122]
              - cell "0" [ref=e123]
              - cell "0" [ref=e124]
            - row [ref=e125]:
              - cell "Łódzkie" [ref=e126]
              - cell "1" [ref=e127]
              - cell "0" [ref=e128]
              - cell "1" [ref=e129]
            - row [ref=e130]:
              - cell "Małopolskie" [ref=e131]
              - cell "1" [ref=e132]
              - cell "0" [ref=e133]
              - cell "1" [ref=e134]
            - row [ref=e135]:
              - cell "Mazowieckie" [ref=e136]
              - cell "1" [ref=e137]
              - cell "0" [ref=e138]
              - cell "1" [ref=e139]
            - row [ref=e140]:
              - cell "Opolskie" [ref=e141]
              - cell "0" [ref=e142]
              - cell "0" [ref=e143]
              - cell "0" [ref=e144]
            - row [ref=e145]:
              - cell "Podkarpackie" [ref=e146]
              - cell "1" [ref=e147]
              - cell "0" [ref=e148]
              - cell "1" [ref=e149]
            - row [ref=e150]:
              - cell "Podlaskie" [ref=e151]
              - cell "1" [ref=e152]
              - cell "0" [ref=e153]
              - cell "1" [ref=e154]
            - row [ref=e155]:
              - cell "Pomorskie" [ref=e156]
              - cell "1" [ref=e157]
              - cell "0" [ref=e158]
              - cell "1" [ref=e159]
            - row [ref=e160]:
              - cell "Śląskie" [ref=e161]
              - cell "1" [ref=e162]
              - cell "0" [ref=e163]
              - cell "1" [ref=e164]
            - row [ref=e165]:
              - cell "Świętokrzyskie" [ref=e166]
              - cell "0" [ref=e167]
              - cell "0" [ref=e168]
              - cell "0" [ref=e169]
            - row [ref=e170]:
              - cell "Warmińsko-Mazurskie" [ref=e171]
              - cell "0" [ref=e172]
              - cell "0" [ref=e173]
              - cell "0" [ref=e174]
            - row [ref=e175]:
              - cell "Wielkopolskie" [ref=e176]
              - cell "1" [ref=e177]
              - cell "0" [ref=e178]
              - cell "1" [ref=e179]
            - row [ref=e180]:
              - cell "Zachodniopomorskie" [ref=e181]
              - cell "1" [ref=e182]
              - cell "0" [ref=e183]
              - cell "1" [ref=e184]
            - row [ref=e185]:
              - cell "Wszystkie" [ref=e186]
              - cell "11" [ref=e187]
              - cell "0" [ref=e188]
              - cell "11" [ref=e189]
        - generic [ref=e190]:
          - list [ref=e191]:
            - listitem [ref=e192]:
              - generic [ref=e193]: Instytucje i sprawy
            - listitem [ref=e194]:
              - link "Listy" [ref=e195] [cursor=pointer]:
                - /url: /monitoringi/monitoring-sadow-apelacyjnych/listy
            - listitem [ref=e196]:
              - link "Projekty" [ref=e197] [cursor=pointer]:
                - /url: /monitoringi/monitoring-sadow-apelacyjnych/projekty
            - listitem [ref=e198]:
              - link "Szablon" [ref=e199] [cursor=pointer]:
                - /url: /monitoringi/monitoring-sadow-apelacyjnych/template
            - listitem [ref=e200]:
              - link "Wyniki" [ref=e201] [cursor=pointer]:
                - /url: /monitoringi/monitoring-sadow-apelacyjnych/results
          - heading "Instytucje i sprawy" [level=3] [ref=e202]
          - generic [ref=e203]:
            - heading [level=4] [ref=e204]:
              - link "Sąd Apelacyjny w Katowicach" [ref=e206] [cursor=pointer]:
                - /url: /sprawy/monitoring-sadow-apelacyjnych-3
              - text: "Status ostatniego wniosku: nieznany"
            - generic [ref=e207]:
              - generic [ref=e208]: Zawartość
              - table [ref=e209]:
                - rowgroup [ref=e210]:
                  - row [ref=e211]:
                    - cell [ref=e212]:
                      - link "Wniosek o udostępnienie informacji publicznej" [ref=e214] [cursor=pointer]:
                        - /url: /listy/7304
                    - cell [ref=e215]:
                      - link "adobrawy" [ref=e217] [cursor=pointer]:
                        - /url: /uzytkownik/adobrawy/
                      - time [ref=e218]: 11 sierpnia 2017 02:48
                  - row [ref=e219]:
                    - cell [ref=e220]:
                      - link "pismo O.Adm-010-117/17" [ref=e222] [cursor=pointer]:
                        - /url: /listy/7940
                    - cell [ref=e223]:
                      - link "Sąd Apelacyjny w Katowicach" [ref=e225] [cursor=pointer]:
                        - /url: /instytucje/sad-apelacyjny-w-katowicach
                      - time [ref=e226]: 18 sierpnia 2017 15:30
                      - paragraph [ref=e227]: ": 1"
                  - row [ref=e229]:
                    - cell [ref=e230]:
                      - 'link "Not read: Wniosek o udostępnienie informacji publicznej" [ref=e232] [cursor=pointer]':
                        - /url: /listy/16862
                    - cell [ref=e233]:
                      - link "Sąd Apelacyjny w Katowicach" [ref=e235] [cursor=pointer]:
                        - /url: /instytucje/sad-apelacyjny-w-katowicach
                      - time [ref=e236]: 16 lipca 2018 08:58
          - generic [ref=e237]:
            - heading [level=4] [ref=e238]:
              - link "Sąd Apelacyjny w Białymstoku" [ref=e240] [cursor=pointer]:
                - /url: /sprawy/monitoring-sadow-apelacyjnych-1
              - text: "Status ostatniego wniosku: nieznany"
            - generic [ref=e241]:
              - generic [ref=e242]: Zawartość
              - table [ref=e243]:
                - rowgroup [ref=e244]:
                  - row [ref=e245]:
                    - cell [ref=e246]:
                      - link "Wniosek o udostępnienie informacji publicznej" [ref=e248] [cursor=pointer]:
                        - /url: /listy/7302
                    - cell [ref=e249]:
                      - link "adobrawy" [ref=e251] [cursor=pointer]:
                        - /url: /uzytkownik/adobrawy/
                      - time [ref=e252]: 11 sierpnia 2017 02:48
                  - row [ref=e253]:
                    - cell [ref=e254]:
                      - 'link "Read: Wniosek o udostępnienie informacji publicznej" [ref=e256] [cursor=pointer]':
                        - /url: /listy/7680
                    - cell [ref=e257]:
                      - link "Sąd Apelacyjny w Białymstoku" [ref=e259] [cursor=pointer]:
                        - /url: /instytucje/sad-apelacyjny-w-bialymstoku
                      - time [ref=e260]: 11 sierpnia 2017 07:15
                  - row [ref=e261]:
                    - cell [ref=e262]:
                      - link "A-061-79/17 dot. wniosku o udostępnienie informacji publicznej" [ref=e264] [cursor=pointer]:
                        - /url: /listy/8927
                    - cell [ref=e265]:
                      - link "Sąd Apelacyjny w Białymstoku" [ref=e267] [cursor=pointer]:
                        - /url: /instytucje/sad-apelacyjny-w-bialymstoku
                      - time [ref=e268]: 25 sierpnia 2017 12:15
                      - paragraph [ref=e269]: ": 2"
                  - row [ref=e271]:
                    - cell [ref=e272]:
                      - 'link "Nieprzeczytane: Wniosek o udostępnienie informacji publicznej" [ref=e274] [cursor=pointer]':
                        - /url: /listy/16858
                    - cell [ref=e275]:
                      - link "Sąd Apelacyjny w Białymstoku" [ref=e277] [cursor=pointer]:
                        - /url: /instytucje/sad-apelacyjny-w-bialymstoku
                      - time [ref=e278]: 16 lipca 2018 08:58
          - generic [ref=e279]:
            - heading [level=4] [ref=e280]:
              - link "Sąd Apelacyjny w Krakowie" [ref=e282] [cursor=pointer]:
                - /url: /sprawy/monitoring-sadow-apelacyjnych-4
              - text: "Status ostatniego wniosku: nieznany"
            - generic [ref=e283]:
              - generic [ref=e284]: Zawartość
              - table [ref=e285]:
                - rowgroup [ref=e286]:
                  - row [ref=e287]:
                    - cell [ref=e288]:
                      - link "Wniosek o udostępnienie informacji publicznej" [ref=e290] [cursor=pointer]:
                        - /url: /listy/7305
                    - cell [ref=e291]:
                      - link "adobrawy" [ref=e293] [cursor=pointer]:
                        - /url: /uzytkownik/adobrawy/
                      - time [ref=e294]: 11 sierpnia 2017 02:48
                  - row [ref=e295]:
                    - cell [ref=e296]:
                      - 'link "Przeczytano: Wniosek o udostępnienie informacji publicznej" [ref=e298] [cursor=pointer]':
                        - /url: /listy/7687
                    - cell [ref=e299]:
                      - link "Sąd Apelacyjny w Krakowie" [ref=e301] [cursor=pointer]:
                        - /url: /instytucje/sad-apelacyjny-w-krakowie
                      - time [ref=e302]: 11 sierpnia 2017 07:30
                  - row [ref=e303]:
                    - cell [ref=e304]:
                      - 'link "Przeczytano: Wniosek o udostępnienie informacji publicznej" [ref=e306] [cursor=pointer]':
                        - /url: /listy/7739
                    - cell [ref=e307]:
                      - link "Sąd Apelacyjny w Krakowie" [ref=e309] [cursor=pointer]:
                        - /url: /instytucje/sad-apelacyjny-w-krakowie
                      - time [ref=e310]: 11 sierpnia 2017 07:45
                  - row [ref=e311]:
                    - cell [ref=e312]:
                      - 'link "Przeczytano: Wniosek o udostępnienie informacji publicznej" [ref=e314] [cursor=pointer]':
                        - /url: /listy/7780
                    - cell [ref=e315]:
                      - link "Sąd Apelacyjny w Krakowie" [ref=e317] [cursor=pointer]:
                        - /url: /instytucje/sad-apelacyjny-w-krakowie
                      - time [ref=e318]: 11 sierpnia 2017 08:15
                  - row [ref=e319]:
                    - cell [ref=e320]:
                      - link "informacja publiczna (Adm.-0143-176/17)" [ref=e322] [cursor=pointer]:
                        - /url: /listy/8014
                    - cell [ref=e323]:
                      - link "Sąd Apelacyjny w Krakowie" [ref=e325] [cursor=pointer]:
                        - /url: /instytucje/sad-apelacyjny-w-krakowie
                      - time [ref=e326]: 22 sierpnia 2017 15:15
                      - paragraph [ref=e327]: ": 6"
                  - row [ref=e329]:
                    - cell [ref=e330]:
                      - 'link "Przeczytano: Wniosek o udostępnienie informacji publicznej" [ref=e332] [cursor=pointer]':
                        - /url: /listy/16559
                    - cell [ref=e333]:
                      - link "Sąd Apelacyjny w Krakowie" [ref=e335] [cursor=pointer]:
                        - /url: /instytucje/sad-apelacyjny-w-krakowie
                      - time [ref=e336]: 21 lutego 2018 10:00
          - generic [ref=e337]:
            - heading [level=4] [ref=e338]:
              - link "Sąd Apelacyjny w Warszawie" [ref=e340] [cursor=pointer]:
                - /url: /sprawy/monitoring-sadow-apelacyjnych-10
              - text: "Status ostatniego wniosku: nieznany"
            - generic [ref=e341]:
              - generic [ref=e342]: Zawartość
              - table [ref=e343]:
                - rowgroup [ref=e344]:
                  - row [ref=e345]:
                    - cell [ref=e346]:
                      - link "Wniosek o udostępnienie informacji publicznej" [ref=e348] [cursor=pointer]:
                        - /url: /listy/7311
                    - cell [ref=e349]:
                      - link "adobrawy" [ref=e351] [cursor=pointer]:
                        - /url: /uzytkownik/adobrawy/
                      - time [ref=e352]: 11 sierpnia 2017 02:48
                  - row [ref=e353]:
                    - cell [ref=e354]:
                      - 'link "Przeczytane: Wniosek o udostępnienie informacji publicznej" [ref=e356] [cursor=pointer]':
                        - /url: /listy/7756
                    - cell [ref=e357]:
                      - link "Sąd Apelacyjny w Warszawie" [ref=e359] [cursor=pointer]:
                        - /url: /instytucje/sad-apelacyjny-w-warszawie
                      - time [ref=e360]: 11 sierpnia 2017 08:00
                  - row [ref=e361]:
                    - cell [ref=e362]:
                      - 'link "RE: Wniosek o udostępnienie informacji publicznej" [ref=e364] [cursor=pointer]':
                        - /url: /listy/7759
                    - cell [ref=e365]:
                      - link "Sąd Apelacyjny w Warszawie" [ref=e367] [cursor=pointer]:
                        - /url: /instytucje/sad-apelacyjny-w-warszawie
                      - time [ref=e368]: 11 sierpnia 2017 08:00
                  - row [ref=e369]:
                    - cell [ref=e370]:
                      - link "przedłużenie terminu na rozpatrzenie wniosku" [ref=e372] [cursor=pointer]:
                        - /url: /listy/8990
                    - cell [ref=e373]:
                      - link "Sąd Apelacyjny w Warszawie" [ref=e375] [cursor=pointer]:
                        - /url: /instytucje/sad-apelacyjny-w-warszawie
                      - time [ref=e376]: 25 sierpnia 2017 16:00
                  - row [ref=e377]:
                    - cell [ref=e378]:
                      - link "odpowiedź na wniosek o udostepnienie inf.publicznej" [ref=e380] [cursor=pointer]:
                        - /url: /listy/9490
                    - cell [ref=e381]:
                      - link "Sąd Apelacyjny w Warszawie" [ref=e383] [cursor=pointer]:
                        - /url: /instytucje/sad-apelacyjny-w-warszawie
                      - time [ref=e384]: 14 września 2017 12:45
                      - paragraph [ref=e385]: ": 1"
          - generic [ref=e387]:
            - heading [level=4] [ref=e388]:
              - link "Sąd Apelacyjny w Poznaniu" [ref=e390] [cursor=pointer]:
                - /url: /sprawy/monitoring-sadow-apelacyjnych-7
              - text: "Status ostatniego wniosku: nieznany"
            - generic [ref=e391]:
              - generic [ref=e392]: Zawartość
              - table [ref=e393]:
                - rowgroup [ref=e394]:
                  - row [ref=e395]:
                    - cell [ref=e396]:
                      - link "Wniosek o udostępnienie informacji publicznej" [ref=e398] [cursor=pointer]:
                        - /url: /listy/7308
                    - cell [ref=e399]:
                      - link "adobrawy" [ref=e401] [cursor=pointer]:
                        - /url: /uzytkownik/adobrawy/
                      - time [ref=e402]: 11 sierpnia 2017 02:48
                  - row [ref=e403]:
                    - cell [ref=e404]:
                      - 'link "Re: Wniosek o udostępnienie informacji publicznej" [ref=e406] [cursor=pointer]':
                        - /url: /listy/7784
                    - cell [ref=e407]:
                      - link "Sąd Apelacyjny w Poznaniu" [ref=e409] [cursor=pointer]:
                        - /url: /instytucje/sad-apelacyjny-w-poznaniu
                      - time [ref=e410]: 11 sierpnia 2017 08:15
                  - row [ref=e411]:
                    - cell [ref=e412]:
                      - link "dot. informacji publicznej" [ref=e414] [cursor=pointer]:
                        - /url: /listy/7984
                    - cell [ref=e415]:
                      - link "Sąd Apelacyjny w Poznaniu" [ref=e417] [cursor=pointer]:
                        - /url: /instytucje/sad-apelacyjny-w-poznaniu
                      - time [ref=e418]: 22 sierpnia 2017 10:45
                  - row [ref=e419]:
                    - cell [ref=e420]:
                      - link "Częściowa odpowiedź na wniosek" [ref=e422] [cursor=pointer]:
                        - /url: /listy/8030
                    - cell [ref=e423]:
                      - link "Sąd Apelacyjny w Poznaniu" [ref=e425] [cursor=pointer]:
                        - /url: /instytucje/sad-apelacyjny-w-poznaniu
                      - time [ref=e426]: 23 sierpnia 2017 13:15
                  - row [ref=e427]:
                    - cell [ref=e428]:
                      - link "odpowiedź na wniosek o informację publiczną" [ref=e430] [cursor=pointer]:
                        - /url: /listy/8903
                    - cell [ref=e431]:
                      - link "Sąd Apelacyjny w Poznaniu" [ref=e433] [cursor=pointer]:
                        - /url: /instytucje/sad-apelacyjny-w-poznaniu
                      - time [ref=e434]: 25 sierpnia 2017 10:00
                      - paragraph [ref=e435]: ": 1"
                  - row [ref=e437]:
                    - cell [ref=e438]:
                      - link "[Brak tematu]" [ref=e440] [cursor=pointer]:
                        - /url: /listy/8979
                    - cell [ref=e441]:
                      - link "Sąd Apelacyjny w Poznaniu" [ref=e443] [cursor=pointer]:
                        - /url: /instytucje/sad-apelacyjny-w-poznaniu
                      - time [ref=e444]: 25 sierpnia 2017 15:00
                  - row [ref=e445]:
                    - cell [ref=e446]:
                      - link "[Brak tematu]" [ref=e448] [cursor=pointer]:
                        - /url: /listy/8981
                    - cell [ref=e449]:
                      - link "Sąd Apelacyjny w Poznaniu" [ref=e451] [cursor=pointer]:
                        - /url: /instytucje/sad-apelacyjny-w-poznaniu
                      - time [ref=e452]: 25 sierpnia 2017 15:15
                  - row [ref=e453]:
                    - cell [ref=e454]:
                      - link "odp.na wniosek dot. Informacji publicznej" [ref=e456] [cursor=pointer]:
                        - /url: /listy/9488
                    - cell [ref=e457]:
                      - link "Sąd Apelacyjny w Poznaniu" [ref=e459] [cursor=pointer]:
                        - /url: /instytucje/sad-apelacyjny-w-poznaniu
                      - time [ref=e460]: 14 września 2017 12:15
                      - paragraph [ref=e461]: ": 1"
          - generic [ref=e463]:
            - heading [level=4] [ref=e464]:
              - link "Sąd Apelacyjny w Gdańsku" [ref=e466] [cursor=pointer]:
                - /url: /sprawy/monitoring-sadow-apelacyjnych-2
              - text: "Status ostatniego wniosku: nieznany"
            - generic [ref=e467]:
              - generic [ref=e468]: Zawartość
              - table [ref=e469]:
                - rowgroup [ref=e470]:
                  - row [ref=e471]:
                    - cell [ref=e472]:
                      - link "Wniosek o udostępnienie informacji publicznej" [ref=e474] [cursor=pointer]:
                        - /url: /listy/7303
                    - cell [ref=e475]:
                      - link "adobrawy" [ref=e477] [cursor=pointer]:
                        - /url: /uzytkownik/adobrawy/
                      - time [ref=e478]: 11 sierpnia 2017 02:48
                  - row [ref=e479]:
                    - cell [ref=e480]:
                      - 'link "Wysyłanie wiadomości e-mail: Adm.105.154.2017.pdf" [ref=e482] [cursor=pointer]':
                        - /url: /listy/8983
                    - cell [ref=e483]:
                      - link "Sąd Apelacyjny w Gdańsku" [ref=e485] [cursor=pointer]:
                        - /url: /instytucje/sad-apelacyjny-w-gdansku
                      - time [ref=e486]: 25 sierpnia 2017 15:15
                      - paragraph [ref=e487]: ": 1"
                  - row [ref=e489]:
                    - cell [ref=e490]:
                      - 'link "Re: Wysyłanie wiadomości e-mail: Adm.105.154.2017.pdf" [ref=e492] [cursor=pointer]':
                        - /url: /listy/8995
                    - cell [ref=e493]:
                      - link "Szymon_Osowski" [ref=e495] [cursor=pointer]:
                        - /url: /uzytkownik/Szymon_Osowski/
                      - time [ref=e496]: 26 sierpnia 2017 09:46
                  - row [ref=e497]:
                    - cell [ref=e498]:
                      - 'link "Przeczytane: Wniosek o udostępnienie informacji publicznej" [ref=e500] [cursor=pointer]':
                        - /url: /listy/9013
                    - cell [ref=e501]:
                      - link "Sąd Apelacyjny w Gdańsku" [ref=e503] [cursor=pointer]:
                        - /url: /instytucje/sad-apelacyjny-w-gdansku
                      - time [ref=e504]: 28 sierpnia 2017 07:45
                  - row [ref=e505]:
                    - cell [ref=e506]:
                      - link "rejestr umów SA Gdańsk" [ref=e508] [cursor=pointer]:
                        - /url: /listy/9049
                    - cell [ref=e509]:
                      - link "Sąd Apelacyjny w Gdańsku" [ref=e511] [cursor=pointer]:
                        - /url: /instytucje/sad-apelacyjny-w-gdansku
                      - time [ref=e512]: 30 sierpnia 2017 08:45
                      - paragraph [ref=e513]: ": 1"
                  - row [ref=e515]:
                    - cell [ref=e516]:
                      - link "decyzja odmowa SA Gdańsk" [ref=e518] [cursor=pointer]:
                        - /url: /listy/9444
                    - cell [ref=e519]:
                      - link "Sąd Apelacyjny w Gdańsku" [ref=e521] [cursor=pointer]:
                        - /url: /instytucje/sad-apelacyjny-w-gdansku
                      - time [ref=e522]: 7 września 2017 11:45
                      - paragraph [ref=e523]: ": 1"
          - generic [ref=e525]:
            - heading [level=4] [ref=e526]:
              - link "Sąd Apelacyjny w Łodzi" [ref=e528] [cursor=pointer]:
                - /url: /sprawy/monitoring-sadow-apelacyjnych-6
              - text: "Status ostatniego wniosku: nieznany"
            - generic [ref=e529]:
              - generic [ref=e530]: Zawartość
              - table [ref=e531]:
                - rowgroup [ref=e532]:
                  - row [ref=e533]:
                    - cell [ref=e534]:
                      - link "Wniosek o udostępnienie informacji publicznej" [ref=e536] [cursor=pointer]:
                        - /url: /listy/7307
                    - cell [ref=e537]:
                      - link "adobrawy" [ref=e539] [cursor=pointer]:
                        - /url: /uzytkownik/adobrawy/
                      - time [ref=e540]: 11 sierpnia 2017 02:48
                  - row [ref=e541]:
                    - cell [ref=e542]:
                      - 'link "Przeczytane: Wniosek o udostępnienie informacji publicznej" [ref=e544] [cursor=pointer]':
                        - /url: /listy/7737
                    - cell [ref=e545]:
                      - link "Sąd Apelacyjny w Łodzi" [ref=e547] [cursor=pointer]:
                        - /url: /instytucje/sad-apelacyjny-w-lodzi
                      - time [ref=e548]: 11 sierpnia 2017 07:45
                  - row [ref=e549]:
                    - cell [ref=e550]:
                      - 'link "Przeczytane: Wniosek o udostępnienie informacji publicznej" [ref=e552] [cursor=pointer]':
                        - /url: /listy/7766
                    - cell [ref=e553]:
                      - link "Sąd Apelacyjny w Łodzi" [ref=e555] [cursor=pointer]:
                        - /url: /instytucje/sad-apelacyjny-w-lodzi
                      - time [ref=e556]: 11 sierpnia 2017 08:00
                  - row [ref=e557]:
                    - cell [ref=e558]:
                      - 'link "Przeczytane: Wniosek o udostępnienie informacji publicznej" [ref=e560] [cursor=pointer]':
                        - /url: /listy/7839
                    - cell [ref=e561]:
                      - link "Sąd Apelacyjny w Łodzi" [ref=e563] [cursor=pointer]:
                        - /url: /instytucje/sad-apelacyjny-w-lodzi
                      - time [ref=e564]: 11 sierpnia 2017 09:30
                  - row [ref=e565]:
                    - cell [ref=e566]:
                      - link "Pismo Prezesa SA w Łodzi AV-0164-105/17 dotyczące wniosku z dnia 11.08.17 r. o udostępnienie informacji publicznej" [ref=e568] [cursor=pointer]:
                        - /url: /listy/8102
                    - cell [ref=e569]:
                      - link "Sąd Apelacyjny w Łodzi" [ref=e571] [cursor=pointer]:
                        - /url: /instytucje/sad-apelacyjny-w-lodzi
                      - time [ref=e572]: 24 sierpnia 2017 15:00
                      - paragraph [ref=e573]: ": 4"
                  - row [ref=e575]:
                    - cell [ref=e576]:
                      - 'link "Nieprzeczytane: Wniosek o udostępnienie informacji publicznej" [ref=e578] [cursor=pointer]':
                        - /url: /listy/9210
                    - cell [ref=e579]:
                      - link "Sąd Apelacyjny w Łodzi" [ref=e581] [cursor=pointer]:
                        - /url: /instytucje/sad-apelacyjny-w-lodzi
                      - time [ref=e582]: 1 września 2017 13:45
          - generic [ref=e583]:
            - heading [level=4] [ref=e584]:
              - link "Sąd Apelacyjny w Lublinie" [ref=e586] [cursor=pointer]:
                - /url: /sprawy/monitoring-sadow-apelacyjnych-5
              - text: "Status ostatniego wniosku: nieznany"
            - generic [ref=e587]:
              - generic [ref=e588]: Zawartość
              - table [ref=e589]:
                - rowgroup [ref=e590]:
                  - row [ref=e591]:
                    - cell [ref=e592]:
                      - link "Wniosek o udostępnienie informacji publicznej" [ref=e594] [cursor=pointer]:
                        - /url: /listy/7306
                    - cell [ref=e595]:
                      - link "adobrawy" [ref=e597] [cursor=pointer]:
                        - /url: /uzytkownik/adobrawy/
                      - time [ref=e598]: 11 sierpnia 2017 02:48
                  - row [ref=e599]:
                    - cell [ref=e600]:
                      - 'link "Read: Wniosek o udostępnienie informacji publicznej" [ref=e602] [cursor=pointer]':
                        - /url: /listy/7888
                    - cell [ref=e603]:
                      - link "Sąd Apelacyjny w Lublinie" [ref=e605] [cursor=pointer]:
                        - /url: /instytucje/sad-apelacyjny-w-lublinie
                      - time [ref=e606]: 16 sierpnia 2017 08:00
                  - row [ref=e607]:
                    - cell [ref=e608]:
                      - link "Adm-063-85/17 informacja publiczna" [ref=e610] [cursor=pointer]:
                        - /url: /listy/8105
                    - cell [ref=e611]:
                      - link "Sąd Apelacyjny w Lublinie" [ref=e613] [cursor=pointer]:
                        - /url: /instytucje/sad-apelacyjny-w-lublinie
                      - time [ref=e614]: 24 sierpnia 2017 15:00
                      - paragraph [ref=e615]: ": 1"
          - generic [ref=e617]:
            - heading [level=4] [ref=e618]:
              - link "Sąd Apelacyjny w Szczecinie" [ref=e620] [cursor=pointer]:
                - /url: /sprawy/monitoring-sadow-apelacyjnych-9
              - text: "Status ostatniego wniosku: nieznany"
            - generic [ref=e621]:
              - generic [ref=e622]: Zawartość
              - table [ref=e623]:
                - rowgroup [ref=e624]:
                  - row [ref=e625]:
                    - cell [ref=e626]:
                      - link "Wniosek o udostępnienie informacji publicznej" [ref=e628] [cursor=pointer]:
                        - /url: /listy/7310
                    - cell [ref=e629]:
                      - link "adobrawy" [ref=e631] [cursor=pointer]:
                        - /url: /uzytkownik/adobrawy/
                      - time [ref=e632]: 11 sierpnia 2017 02:48
                  - row [ref=e633]:
                    - cell [ref=e634]:
                      - link "informacja publiczna" [ref=e636] [cursor=pointer]:
                        - /url: /listy/8043
                    - cell [ref=e637]:
                      - link "Sąd Apelacyjny w Szczecinie" [ref=e639] [cursor=pointer]:
                        - /url: /instytucje/sad-apelacyjny-w-szczecinie
                      - time [ref=e640]: 23 sierpnia 2017 15:00
                      - paragraph [ref=e641]: ": 1"
          - generic [ref=e643]:
            - heading [level=4] [ref=e644]:
              - link "Sąd Apelacyjny we Wrocławiu" [ref=e646] [cursor=pointer]:
                - /url: /sprawy/monitoring-sadow-apelacyjnych-11
              - text: "Status ostatniego wniosku: nieznany"
            - generic [ref=e647]:
              - generic [ref=e648]: Zawartość
              - table [ref=e649]:
                - rowgroup [ref=e650]:
                  - row [ref=e651]:
                    - cell [ref=e652]:
                      - link "Wniosek o udostępnienie informacji publicznej" [ref=e654] [cursor=pointer]:
                        - /url: /listy/7312
                    - cell [ref=e655]:
                      - link "adobrawy" [ref=e657] [cursor=pointer]:
                        - /url: /uzytkownik/adobrawy/
                      - time [ref=e658]: 11 sierpnia 2017 02:48
                  - row [ref=e659]:
                    - cell [ref=e660]:
                      - 'link "Read: Wniosek o udostępnienie informacji publicznej" [ref=e662] [cursor=pointer]':
                        - /url: /listy/7716
                    - cell [ref=e663]:
                      - link "Sąd Apelacyjny we Wrocławiu" [ref=e665] [cursor=pointer]:
                        - /url: /instytucje/sad-apelacyjny-we-wroclawiu
                      - time [ref=e666]: 11 sierpnia 2017 07:45
                  - row [ref=e667]:
                    - cell [ref=e668]:
                      - link "dot. wniosku o udostępnienie informacji publicznej" [ref=e670] [cursor=pointer]:
                        - /url: /listy/8002
                    - cell [ref=e671]:
                      - link "Sąd Apelacyjny we Wrocławiu" [ref=e673] [cursor=pointer]:
                        - /url: /instytucje/sad-apelacyjny-we-wroclawiu
                      - time [ref=e674]: 22 sierpnia 2017 14:30
                      - paragraph [ref=e675]: ": 2"
          - generic [ref=e677]:
            - heading [level=4] [ref=e678]:
              - link "Sąd Apelacyjny w Rzeszowie" [ref=e680] [cursor=pointer]:
                - /url: /sprawy/monitoring-sadow-apelacyjnych-8
              - text: "Status ostatniego wniosku: nieznany"
            - generic [ref=e681]:
              - generic [ref=e682]: Zawartość
              - table [ref=e683]:
                - rowgroup [ref=e684]:
                  - row [ref=e685]:
                    - cell [ref=e686]:
                      - link "Wniosek o udostępnienie informacji publicznej" [ref=e688] [cursor=pointer]:
                        - /url: /listy/7309
                    - cell [ref=e689]:
                      - link "adobrawy" [ref=e691] [cursor=pointer]:
                        - /url: /uzytkownik/adobrawy/
                      - time [ref=e692]: 11 sierpnia 2017 02:48
                  - row [ref=e693]:
                    - cell [ref=e694]:
                      - 'link "Przeczytane: Wniosek o udostępnienie informacji publicznej" [ref=e696] [cursor=pointer]':
                        - /url: /listy/7946
                    - cell [ref=e697]:
                      - link "Sąd Apelacyjny w Rzeszowie" [ref=e699] [cursor=pointer]:
                        - /url: /instytucje/sad-apelacyjny-w-rzeszowie
                      - time [ref=e700]: 21 sierpnia 2017 09:15
          - list [ref=e702]:
            - listitem [ref=e703]:
              - generic [aria-hidden]: ←
            - listitem [ref=e704]:
              - generic "Current Page" [ref=e705]: "1"
            - listitem [ref=e706]:
              - generic [aria-hidden]: →
      - generic [ref=e707]:
        - generic [ref=e708]:
          - text: Sieć Obywatelska - Watchdog Polska ul. Szpitalna 5/5 00-031 Warszawa
          - paragraph [ref=e709]:
            - link "Klauzula RODO" [ref=e710] [cursor=pointer]:
              - /url: https://fedrowanie.siecobywatelska.pl/media_internal/tinycontent/uploads/11_KLAUZULA.-.dane.z.udip.na.stronach.www.i.w.mediach.spol.pdf
        - generic [ref=e711]:
          - text: "silnik:"
          - link "jawne.info.pl" [ref=e712] [cursor=pointer]:
            - /url: http://jawne.info.pl
          - text: "| v1.5.77.deps |"
          - link "GitHub" [ref=e713] [cursor=pointer]:
            - /url: https://github.com/watchdogpolska/feder
          - text: –
          - link "efd8a3b" [ref=e715] [cursor=pointer]:
            - /url: https://github.com/watchdogpolska/feder/compare/efd8a3b2...master
          - text: "|"
          - link "API" [ref=e716] [cursor=pointer]:
            - /url: /api/
        - generic [ref=e718]: Ta strona wykorzystuje cookies.
  - list [ref=e720]:
    - listitem [ref=e721]:
      - link "Ukryj »" [ref=e722] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e723]:
      - link "Toggle Theme" [ref=e724] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e727]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e728]
      - link "Historia /monitoringi/monitoring-sadow-apelacyjnych" [ref=e729] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e730]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e731]
      - link "Wersje Django 5.2.17" [ref=e732] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e733]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e734]
      - 'link "Czas CPU: 435.39ms (471.81ms)" [ref=e735] [cursor=pointer]':
        - /url: "#"
    - listitem [ref=e736]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e737]
      - link "Ustawienia" [ref=e738] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e739]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e740]
      - link "Nagłówki" [ref=e741] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e742]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e743]
      - link "Zapytania MonitoringDetailView" [ref=e744] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e745]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e746]
      - link "SQL 92 queries in 52.87ms" [ref=e747] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e748]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e749]
      - link "Pliki statyczne 3 użyte plików" [ref=e750] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e751]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e752]
      - link "Templatki monitorings/monitoring_detail.html" [ref=e753] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e754]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e755]
      - link "Alerty" [ref=e756] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e757]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e758]
      - link "Cache 2 wywołania w 0.20ms" [ref=e759] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e760]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e761]
      - link "Sygnały 88 odbiorców 15 sygnałów" [ref=e762] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e763]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e764]
      - link "Gmina" [ref=e765] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e766]:
      - checkbox "Enable for next and successive requests" [ref=e767]
      - generic [ref=e768]: Przechwycone przekierowania
    - listitem [ref=e769]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e770]
      - link "Profilowanie" [ref=e771] [cursor=pointer]:
        - /url: "#"
```

# Test source

```ts
  1  | import { test, expect } from '@playwright/test';
  2  | import { PAGES } from './pages';
  3  | import { VIEWPORTS } from './viewports';
  4  | 
  5  | // Bootstrap 3's `.row` negative margins (-15px) routinely push scrollWidth ~4px past
  6  | // clientWidth on every page even with no visible scrollbar - that's cosmetic BS3 grid
  7  | // noise, not real overflow. A genuinely overflowing wide table measured ~278px over.
  8  | // This tolerance separates the two instead of flagging every single page.
  9  | const OVERFLOW_TOLERANCE_PX = 20;
  10 | 
  11 | test.describe('Layout sanity - desktop', () => {
  12 |   test.use({ viewport: VIEWPORTS.desktop });
  13 | 
  14 |   for (const page of PAGES) {
  15 |     test(`${page.name} - no horizontal overflow`, async ({ page: pw }) => {
  16 |       await pw.goto(page.path);
  17 |       await pw.waitForLoadState('networkidle');
  18 |       const overflowPx = await pw.evaluate(() =>
  19 |         document.documentElement.scrollWidth - document.documentElement.clientWidth
  20 |       );
  21 |       expect(overflowPx, 'Page has horizontal scroll').toBeLessThanOrEqual(OVERFLOW_TOLERANCE_PX);
  22 |     });
  23 |   }
  24 | });
  25 | 
  26 | test.describe('Layout sanity - mobile', () => {
  27 |   test.use({ viewport: VIEWPORTS.mobile });
  28 | 
  29 |   for (const page of PAGES) {
  30 |     test(`${page.name} - no horizontal overflow on mobile`, async ({ page: pw }) => {
  31 |       await pw.goto(page.path);
  32 |       await pw.waitForLoadState('networkidle');
  33 |       const overflowPx = await pw.evaluate(() =>
  34 |         document.documentElement.scrollWidth - document.documentElement.clientWidth
  35 |       );
> 36 |       expect(overflowPx, 'Mobile layout has horizontal scroll').toBeLessThanOrEqual(OVERFLOW_TOLERANCE_PX);
     |                                                                 ^ Error: Mobile layout has horizontal scroll
  37 |     });
  38 |   }
  39 | });
  40 | 
  41 | // This app's desktop layout (feder/main/templates/base.html) is a permanent left
  42 | // `.sidebar` next to `.content`, not a top navbar - `.navbar` is `display: none` above
  43 | // the mobile breakpoint (it only reappears, with `.navbar-toggle`, on small screens).
  44 | // So "nav above content" doesn't apply on desktop; the real desktop invariant is
  45 | // "sidebar sits to the left of content", checked below instead.
  46 | test.describe('Navigation structure', () => {
  47 |   test.use({ viewport: VIEWPORTS.desktop });
  48 | 
  49 |   test('sidebar is left of main content on desktop', async ({ page: pw }) => {
  50 |     await pw.goto('/');
  51 |     await pw.waitForLoadState('networkidle');
  52 |     const sidebarBox = await pw.locator('.sidebar').first().boundingBox();
  53 |     const contentBox = await pw.locator('.content').first().boundingBox();
  54 |     expect(sidebarBox).toBeTruthy();
  55 |     expect(contentBox).toBeTruthy();
  56 |     expect(sidebarBox!.x + sidebarBox!.width).toBeLessThanOrEqual(contentBox!.x + 5);
  57 |   });
  58 | 
  59 |   test('navbar collapses on mobile', async ({ page: pw }) => {
  60 |     await pw.setViewportSize(VIEWPORTS.mobile);
  61 |     await pw.goto('/');
  62 |     await pw.waitForLoadState('networkidle');
  63 |     const toggle = pw.locator('.navbar-toggle, .navbar-toggler');
  64 |     await expect(toggle).toBeVisible();
  65 |   });
  66 | });
  67 | 
```