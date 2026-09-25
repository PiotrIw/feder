# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: layout_assertions.spec.ts >> Layout sanity - mobile >> monitorings-letters - no horizontal overflow on mobile
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
              - link "Instytucje i sprawy" [ref=e193] [cursor=pointer]:
                - /url: /monitoringi/monitoring-sadow-apelacyjnych
            - listitem [ref=e194]:
              - generic [ref=e195]: Listy
            - listitem [ref=e196]:
              - link "Projekty" [ref=e197] [cursor=pointer]:
                - /url: /monitoringi/monitoring-sadow-apelacyjnych/projekty
            - listitem [ref=e198]:
              - link "Szablon" [ref=e199] [cursor=pointer]:
                - /url: /monitoringi/monitoring-sadow-apelacyjnych/template
            - listitem [ref=e200]:
              - link "Wyniki" [ref=e201] [cursor=pointer]:
                - /url: /monitoringi/monitoring-sadow-apelacyjnych/results
          - heading "Listy" [level=3] [ref=e202]
          - generic [ref=e203]:
            - heading [level=3] [ref=e204]:
              - 'link "Not read: Wniosek o udostępnienie informacji publicznej" [ref=e206] [cursor=pointer]':
                - /url: /listy/16862
              - generic [ref=e207]:
                - text: przez
                - link "Sąd Apelacyjny w Katowicach" [ref=e209] [cursor=pointer]:
                  - /url: /instytucje/sad-apelacyjny-w-katowicach
                - text: w sprawie
                - 'link "Monitoring sądów apelacyjnych #3" [ref=e211] [cursor=pointer]':
                  - /url: /sprawy/monitoring-sadow-apelacyjnych-3
                - time [ref=e212]: 16 lipca 2018 08:58
            - generic [ref=e213]:
              - paragraph [ref=e214]: Twoja wiadomość
              - paragraph [ref=e215]: "Do: 2003-informacja Temat: Wniosek o udostępnienie informacji publicznej Wysłano: 11 sierpnia 2017 02:48:11 (UTC+01:00) Sarajewo, Skopie, Warszawa, Zagrzeb"
              - paragraph [ref=e216]: "została usunięta nieprzeczytana: 5 czerwca 2018 07:52:37 (UTC+01:00) Sarajewo, Skopie, Warszawa, Zagrzeb."
          - generic [ref=e217]:
            - heading [level=3] [ref=e218]:
              - 'link "Nieprzeczytane: Wniosek o udostępnienie informacji publicznej" [ref=e220] [cursor=pointer]':
                - /url: /listy/16858
              - generic [ref=e221]:
                - text: przez
                - link "Sąd Apelacyjny w Białymstoku" [ref=e223] [cursor=pointer]:
                  - /url: /instytucje/sad-apelacyjny-w-bialymstoku
                - text: w sprawie
                - 'link "Monitoring sądów apelacyjnych #1" [ref=e225] [cursor=pointer]':
                  - /url: /sprawy/monitoring-sadow-apelacyjnych-1
                - time [ref=e226]: 16 lipca 2018 08:58
            - generic [ref=e227]:
              - paragraph [ref=e228]: Twoja wiadomość
              - paragraph [ref=e229]: "Do: Jarosławska Marzanna Temat: Wniosek o udostępnienie informacji publicznej Wysłano: 11 sierpnia 2017 02:48:11 (UTC+01:00) Sarajewo, Skopie, Warszawa, Zagrzeb"
              - paragraph [ref=e230]: "została usunięta nieprzeczytana: 28 maja 2018 16:53:54 (UTC+01:00) Sarajewo, Skopie, Warszawa, Zagrzeb."
          - generic [ref=e231]:
            - heading [level=3] [ref=e232]:
              - 'link "Przeczytano: Wniosek o udostępnienie informacji publicznej" [ref=e234] [cursor=pointer]':
                - /url: /listy/16559
              - generic [ref=e235]:
                - text: przez
                - link "Sąd Apelacyjny w Krakowie" [ref=e237] [cursor=pointer]:
                  - /url: /instytucje/sad-apelacyjny-w-krakowie
                - text: w sprawie
                - 'link "Monitoring sądów apelacyjnych #4" [ref=e239] [cursor=pointer]':
                  - /url: /sprawy/monitoring-sadow-apelacyjnych-4
                - time [ref=e240]: 21 lutego 2018 10:00
            - generic [ref=e241]:
              - paragraph [ref=e242]: Twoja wiadomość
              - paragraph [ref=e243]: "Do: Kluza, Agnieszka Temat: Wniosek o udostępnienie informacji publicznej Wysłano: 11 sierpnia 2017 02:48:11 (UTC+01:00) Sarajewo, Skopie, Warszawa, Zagrzeb"
              - paragraph [ref=e244]: została przeczytana o godzinie 21 lutego 2018 09:46:49 (UTC+01:00) Sarajewo, Skopie, Warszawa, Zagrzeb.
          - generic [ref=e245]:
            - heading [level=3] [ref=e246]:
              - link "odpowiedź na wniosek o udostepnienie inf.publicznej" [ref=e248] [cursor=pointer]:
                - /url: /listy/9490
              - generic [ref=e249]:
                - text: przez
                - link "Sąd Apelacyjny w Warszawie" [ref=e251] [cursor=pointer]:
                  - /url: /instytucje/sad-apelacyjny-w-warszawie
                - text: w sprawie
                - 'link "Monitoring sądów apelacyjnych #10" [ref=e253] [cursor=pointer]':
                  - /url: /sprawy/monitoring-sadow-apelacyjnych-10
                - time [ref=e254]: 14 września 2017 12:45
            - generic [ref=e255]:
              - paragraph [ref=e256]: Stowarzyszenie Sieć Obywatelska Watchdog PolskaStosownie do wniosku z 11 sierpnia 2017 r. o udostępnienie informacji publicznej - w wykonaniu zarządzenia Wiceprezesa Sądu Apelacyjnego w Warszawie, uprzejmie informuję, że:1. Orzeczenia dyscyplinarne sędziów Sądu Apelacyjnego w Warszawie nie są publikowane na żadnej stronie internetowej ,2. Rejestr umów od 1 stycznia 2017 do 31 lipca 2017 r. - w załączeniu,3. W Sądzie Apelacyjnym w Warszawie była przeprowadzona kontrola NIK- raport został zamieszczony na stronie tut. sądu w zakładce Kontrole,
              - paragraph [ref=e257]: 4. Właściwymi rzeczowo dla spraw z art. 212 Kodeku Karnego są Sądy Rejonowe, a instancją odwoławczą są Sądy Okręgowe,
              - paragraph [ref=e258]: "5. W 2017 r. z art. 231 Kodeksu Karnego w tut. sądzie zapadły 3 wyroki w sprawach: II AKa 203/16, II AKa 29/17, II Aka 120/17. Wyroki wraz z uzasadnieniem tych spraw zostały opublikowane na stronie Portalu Orzeczeń Sądu Apelacyjnego w Warszawie pod adresem http://orzeczenia.waw.sa.gov.pl ,"
              - paragraph [ref=e259]: 6. Kalendarz spotkań prezes sądu -nie jest dokumentem urzędowym - gdyż nie stanowi ani oświadczenia woli, ani oświadczenia wiedzy zgodnie z art.6 ust. 2 ustawy o dostępie do informacji publicznej. Należy go zakwalifikować jako dokumentację wewnętrzną. W związku z powyższym nie jest informacją publiczną, o której mowa w art. 1 ust. 1 ustawy o dostępie do informacji publicznej.
              - paragraph [ref=e260]: SekretarkaRenata KosOddział AdministracyjnySąd Apelacyjny w WarszawiePl.Krasińskich 2/4/600-207 Warszawa
            - generic [ref=e261]: "1"
          - generic [ref=e263]:
            - heading [level=3] [ref=e264]:
              - link "odp.na wniosek dot. Informacji publicznej" [ref=e266] [cursor=pointer]:
                - /url: /listy/9488
              - generic [ref=e267]:
                - text: przez
                - link "Sąd Apelacyjny w Poznaniu" [ref=e269] [cursor=pointer]:
                  - /url: /instytucje/sad-apelacyjny-w-poznaniu
                - text: w sprawie
                - 'link "Monitoring sądów apelacyjnych #7" [ref=e271] [cursor=pointer]':
                  - /url: /sprawy/monitoring-sadow-apelacyjnych-7
                - time [ref=e272]: 14 września 2017 12:15
            - generic [ref=e273]:
              - paragraph
            - generic [ref=e274]: "1"
          - generic [ref=e276]:
            - heading [level=3] [ref=e277]:
              - link "decyzja odmowa SA Gdańsk" [ref=e279] [cursor=pointer]:
                - /url: /listy/9444
              - generic [ref=e280]:
                - text: przez
                - link "Sąd Apelacyjny w Gdańsku" [ref=e282] [cursor=pointer]:
                  - /url: /instytucje/sad-apelacyjny-w-gdansku
                - text: w sprawie
                - 'link "Monitoring sądów apelacyjnych #2" [ref=e284] [cursor=pointer]':
                  - /url: /sprawy/monitoring-sadow-apelacyjnych-2
                - time [ref=e285]: 7 września 2017 11:45
            - paragraph [ref=e287]: Pismo wpłynęło do biura SOWP 5.09.2017 r.
            - generic [ref=e288]: "1"
          - generic [ref=e290]:
            - heading [level=3] [ref=e291]:
              - 'link "Nieprzeczytane: Wniosek o udostępnienie informacji publicznej" [ref=e293] [cursor=pointer]':
                - /url: /listy/9210
              - generic [ref=e294]:
                - text: przez
                - link "Sąd Apelacyjny w Łodzi" [ref=e296] [cursor=pointer]:
                  - /url: /instytucje/sad-apelacyjny-w-lodzi
                - text: w sprawie
                - 'link "Monitoring sądów apelacyjnych #6" [ref=e298] [cursor=pointer]':
                  - /url: /sprawy/monitoring-sadow-apelacyjnych-6
                - time [ref=e299]: 1 września 2017 13:45
            - generic [ref=e300]:
              - paragraph [ref=e301]: Twoja wiadomość
              - paragraph [ref=e302]: "Do: Grabia Dagmara Temat: Wniosek o udostępnienie informacji publicznej Wysłano: 11 sierpnia 2017 02:48:11 (UTC+01:00) Sarajewo, Skopie, Warszawa, Zagrzeb"
              - paragraph [ref=e303]: "została usunięta nieprzeczytana: 1 września 2017 08:21:13 (UTC+01:00) Sarajewo, Skopie, Warszawa, Zagrzeb."
          - generic [ref=e304]:
            - heading [level=3] [ref=e305]:
              - link "rejestr umów SA Gdańsk" [ref=e307] [cursor=pointer]:
                - /url: /listy/9049
              - generic [ref=e308]:
                - text: przez
                - link "Sąd Apelacyjny w Gdańsku" [ref=e310] [cursor=pointer]:
                  - /url: /instytucje/sad-apelacyjny-w-gdansku
                - text: w sprawie
                - 'link "Monitoring sądów apelacyjnych #2" [ref=e312] [cursor=pointer]':
                  - /url: /sprawy/monitoring-sadow-apelacyjnych-2
                - time [ref=e313]: 30 sierpnia 2017 08:45
            - generic [ref=e314]:
              - paragraph [ref=e315]: Dzień dobry.W załączeniu przesyłam pismo Pana SSA Jacka Pietrzaka Wiceprezesa Sądu Apelacyjnego w Gdańsku z dnia 29 sierpnia 2017r. wraz z rejestrem umów.Jednocześnie informuję, że decyzja, o której mowa w ww. piśmie została przesłana na wskazany adres pocztowy.
              - paragraph [ref=e316]: Z poważaniemKarolina Śreniawa-PisarskaZ-ca Kierownika Oddziału Administracyjnegow Sądzie Apelacyjnym w Gdańsku
            - generic [ref=e317]: "1"
          - generic [ref=e319]:
            - heading [level=3] [ref=e320]:
              - 'link "Przeczytane: Wniosek o udostępnienie informacji publicznej" [ref=e322] [cursor=pointer]':
                - /url: /listy/9013
              - generic [ref=e323]:
                - text: przez
                - link "Sąd Apelacyjny w Gdańsku" [ref=e325] [cursor=pointer]:
                  - /url: /instytucje/sad-apelacyjny-w-gdansku
                - text: w sprawie
                - 'link "Monitoring sądów apelacyjnych #2" [ref=e327] [cursor=pointer]':
                  - /url: /sprawy/monitoring-sadow-apelacyjnych-2
                - time [ref=e328]: 28 sierpnia 2017 07:45
            - generic [ref=e329]:
              - paragraph [ref=e330]: To jest potwierdzenie dla wiadomości e-mail wysłanej przez Ciebie do<boi@gdansk.sa.gov.pl> o 2017-08-26 09:46
              - paragraph [ref=e331]: To potwierdzenie mówi, że wiadomość została wyświetlona na komputerze adresata o2017-08-28 07:42
          - generic [ref=e332]:
            - heading [level=3] [ref=e333]:
              - 'link "Re: Wysyłanie wiadomości e-mail: Adm.105.154.2017.pdf" [ref=e335] [cursor=pointer]':
                - /url: /listy/8995
              - generic [ref=e336]:
                - text: przez
                - link "Szymon_Osowski" [ref=e338] [cursor=pointer]:
                  - /url: /uzytkownik/Szymon_Osowski/
                - text: w sprawie
                - 'link "Monitoring sądów apelacyjnych #2" [ref=e340] [cursor=pointer]':
                  - /url: /sprawy/monitoring-sadow-apelacyjnych-2
                - time [ref=e341]: 26 sierpnia 2017 09:46
            - generic [ref=e342]:
              - paragraph [ref=e343]: Szanowni Państwo,
              - paragraph [ref=e344]: "W zakresie wyłączenia jawności wnosimy o wydanie i doręczenie decyzji odmownej. Nadto wskazujemy, że każde wyłączenie jawności rodzi obowiązek wydania decyzji z urzędu. Nie można pozostawić wniosku bez rozpoznania - a przynajmniej nie doszukujemy się takiego rozwiązania w ustawie o dostępie do informacji publicznej. Adres stowarzyszenia: ul. Ursynowska 22/2, 02-605 Warszawa."
              - paragraph [ref=e345]: "Odnośnie żądania podpisania wniosku wskazujemy, że nie ma podstaw do zobowiązania nas w tym zakresie - zgadzamy się ze stanowiskiem wyrażonym na naszej stronie internetowej: http://informacjapubliczna.org/aktualnosci/mozna-zmuszac-podpisania-wniosku-o-informacje/"
              - paragraph [ref=e346]: Bartosz Wilk, Szymon Osowski, członkowie zarządu zgodnie z zasadami reprezentacji
              - paragraph [ref=e347]: "Sieć Obywatelska - Watchdog Polskaul. Ursynowska 22/2 | 02-605 Warszawatel: 22 844 73 55 | fax: 22 207 24 09www.siecobywatelska.plwww.watchdogportal.plwww.funduszesoleckie.plwww.informacjapubliczna.org.plNIP 526282872KRS 0000181348 Sąd Rejonowy dla m. st. Warszawy w Warszawie, XII Wydział Gospodarczy Krajowego Rejestru Sądowego"
            - button "Pokaż cytat" [ref=e348] [cursor=pointer]
          - generic [ref=e350]:
            - heading [level=3] [ref=e351]:
              - link "przedłużenie terminu na rozpatrzenie wniosku" [ref=e353] [cursor=pointer]:
                - /url: /listy/8990
              - generic [ref=e354]:
                - text: przez
                - link "Sąd Apelacyjny w Warszawie" [ref=e356] [cursor=pointer]:
                  - /url: /instytucje/sad-apelacyjny-w-warszawie
                - text: w sprawie
                - 'link "Monitoring sądów apelacyjnych #10" [ref=e358] [cursor=pointer]':
                  - /url: /sprawy/monitoring-sadow-apelacyjnych-10
                - time [ref=e359]: 25 sierpnia 2017 16:00
            - generic [ref=e360]:
              - paragraph [ref=e361]: Stowarzyszenie Sieć Obywatelska Watchdog Polska Stosownie do wniosku z 11 sierpnia 2017 r. o udostępnienie informacji publicznej, zgodnie z art.13 ust.2 ustawy o dostępie do informacji publicznej, wyznaczono nowy termin na załatwienie wniosku o udostepnienie informacji publicznej. tj. dzień 22 września 2017 r. ze względu na duży zakres wniosku.
              - paragraph [ref=e362]: Renata KosSekretarkaOddział AdministracyjnySąd Apelacyjny w WarszawiePl.Krasińskich 2/4/600-207 Warszawa
          - generic [ref=e363]:
            - heading [level=3] [ref=e364]:
              - 'link "Wysyłanie wiadomości e-mail: Adm.105.154.2017.pdf" [ref=e366] [cursor=pointer]':
                - /url: /listy/8983
              - generic [ref=e367]:
                - text: przez
                - link "Sąd Apelacyjny w Gdańsku" [ref=e369] [cursor=pointer]:
                  - /url: /instytucje/sad-apelacyjny-w-gdansku
                - text: w sprawie
                - 'link "Monitoring sądów apelacyjnych #2" [ref=e371] [cursor=pointer]':
                  - /url: /sprawy/monitoring-sadow-apelacyjnych-2
                - time [ref=e372]: 25 sierpnia 2017 15:15
            - generic [ref=e373]:
              - paragraph [ref=e374]: (dot. Adm.105.154.2017)
              - paragraph [ref=e375]: W załączeniu uprzejmie przesyłam odpowiedź z dnia 25 sierpnia 2017r. Pana SSA Jacka Pietrzaka wykonującego funkcję Prezesa Sądu Apelacyjnego w Gdańsku na Państwa wniosek z dnia 11 sierpnia 2017r. wraz ze stosownym zobowiązaniem.Uprzejmie proszę o potwierdzenie otrzymania nin. meila.
              - paragraph [ref=e376]: Z poważaniemInsp. Mariola BracaOddział AdministracyjnySądu Apelacyjnegow Gdańsku
            - generic [ref=e377]: "1"
          - generic [ref=e379]:
            - heading [level=3] [ref=e380]:
              - link "[Brak tematu]" [ref=e382] [cursor=pointer]:
                - /url: /listy/8981
              - generic [ref=e383]:
                - text: przez
                - link "Sąd Apelacyjny w Poznaniu" [ref=e385] [cursor=pointer]:
                  - /url: /instytucje/sad-apelacyjny-w-poznaniu
                - text: w sprawie
                - 'link "Monitoring sądów apelacyjnych #7" [ref=e387] [cursor=pointer]':
                  - /url: /sprawy/monitoring-sadow-apelacyjnych-7
                - time [ref=e388]: 25 sierpnia 2017 15:15
            - generic [ref=e389]:
              - paragraph [ref=e390]: W związku z pismem z dnia 11.8.2017. przekazanym na adres mailowy Sądu Apelacyjnego w Poznaniu - wpływ do II Wydziału Karnego SA 14 sierpnia 2017r. , uprzejmie informuję, iż wobec obszerności informacji, o jakie się Państwo zwracają, zostaną one udzielone w terminie późniejszym.
              - paragraph [ref=e391]: Z poważaniemZ-ca Kierownika II Wydziału Karnego Sądu Apelacyjnego w PoznaniuMilenia Brdęk
          - generic [ref=e392]:
            - heading [level=3] [ref=e393]:
              - link "[Brak tematu]" [ref=e395] [cursor=pointer]:
                - /url: /listy/8979
              - generic [ref=e396]:
                - text: przez
                - link "Sąd Apelacyjny w Poznaniu" [ref=e398] [cursor=pointer]:
                  - /url: /instytucje/sad-apelacyjny-w-poznaniu
                - text: w sprawie
                - 'link "Monitoring sądów apelacyjnych #7" [ref=e400] [cursor=pointer]':
                  - /url: /sprawy/monitoring-sadow-apelacyjnych-7
                - time [ref=e401]: 25 sierpnia 2017 15:00
            - paragraph [ref=e403]: W związki z pismem z dnia 11.8.2017. przekazanym na adres mailowy Sadu Apelacyjnego w Poznaniu, uprzejmie informuę, iz wobec obszerwnosci informajci, o jakie się Państwo zwracaja , zostanąone udzielone w terminiw późniejszym.
          - generic [ref=e404]:
            - heading [level=3] [ref=e405]:
              - link "A-061-79/17 dot. wniosku o udostępnienie informacji publicznej" [ref=e407] [cursor=pointer]:
                - /url: /listy/8927
              - generic [ref=e408]:
                - text: przez
                - link "Sąd Apelacyjny w Białymstoku" [ref=e410] [cursor=pointer]:
                  - /url: /instytucje/sad-apelacyjny-w-bialymstoku
                - text: w sprawie
                - 'link "Monitoring sądów apelacyjnych #1" [ref=e412] [cursor=pointer]':
                  - /url: /sprawy/monitoring-sadow-apelacyjnych-1
                - time [ref=e413]: 25 sierpnia 2017 12:15
            - paragraph [ref=e415]: "<html xmlns:v=\"urn:schemas-microsoft-com:vml\" xmlns:o=\"urn:schemas-microsoft-com:office:office\" xmlns:w=\"urn:schemas-microsoft-com:office:word\" xmlns:m=\"http://schemas.microsoft.com/office/2004/12/omml\" xmlns=\"http://www.w3.org/TR/REC-html40\"><head><meta http-equiv=\"Content-Type\" content=\"text/html; charset=iso-8859-2\"><meta name=\"Generator\" content=\"Microsoft Word 14 (filtered medium)\"><style><!--/* Font Definitions */@font-face {font-family:Calibri; panose-1:2 15 5 2 2 2 4 3 2 4;}/* Style Definitions */p.MsoNormal, li.MsoNormal, div.MsoNormal {margin:0cm; margin-bottom:.0001pt; font-size:11.0pt; font-family:\"Calibri\",\"sans-serif\"; mso-fareast-language:EN-US;}a:link, span.MsoHyperlink {mso-style-priority:99; color:blue; text-decoration:underline;}a:visited, span.MsoHyperlinkFollowed {mso-style-priority:99; color:purple; text-decoration:underline;}span.Stylwiadomocie-mail17 {mso-style-type:personal-compose; font-family:\"Calibri\",\"sans-serif\"; color:windowtext;}.MsoChpDefault {mso-style-type:export-only; font-family:\"Calibri\",\"sans-serif\"; mso-fareast-language:EN-US;}@page WordSection1 {size:612.0pt 792.0pt; margin:70.85pt 70.85pt 70.85pt 70.85pt;}div.WordSection1 {page:WordSection1;}--></style><!--[if gte mso 9]><xml><o:shapedefaults v:ext=\"edit\" spidmax=\"1026\" /></xml><![endif]--><!--[if gte mso 9]><xml><o:shapelayout v:ext=\"edit\"><o:idmap v:ext=\"edit\" data=\"1\" /></o:shapelayout></xml><![endif]--></head><body lang=\"PL\" link=\"blue\" vlink=\"purple\"><div class=\"WordSection1\"><p class=\"MsoNormal\"><o:p>&nbsp;</o:p></p></div></body></html>"
            - generic [ref=e416]: "2"
          - generic [ref=e418]:
            - heading [level=3] [ref=e419]:
              - link "odpowiedź na wniosek o informację publiczną" [ref=e421] [cursor=pointer]:
                - /url: /listy/8903
              - generic [ref=e422]:
                - text: przez
                - link "Sąd Apelacyjny w Poznaniu" [ref=e424] [cursor=pointer]:
                  - /url: /instytucje/sad-apelacyjny-w-poznaniu
                - text: w sprawie
                - 'link "Monitoring sądów apelacyjnych #7" [ref=e426] [cursor=pointer]':
                  - /url: /sprawy/monitoring-sadow-apelacyjnych-7
                - time [ref=e427]: 25 sierpnia 2017 10:00
            - paragraph [ref=e429]: <!DOCTYPE html><html><head> <meta charset="UTF-8"></head><body><p>Witam, w załączeniu uprzejmie przekazuję odpowiedź na wniosek&#160; o informacje publiczną przekazany nam przez Oddział Kadr naszego sądu.<br></p><p><br></p><p>Beata Becker<br></p><p>Z-ca Kierownika Oddziału Administracyjnego<br></p><p>Sąd Apelacyjny w Poznaniu<br></p><p>tel. 61 8 27 45 72<br></p></body></html>
            - generic [ref=e430]: "1"
          - generic [ref=e432]:
            - heading [level=3] [ref=e433]:
              - link "Adm-063-85/17 informacja publiczna" [ref=e435] [cursor=pointer]:
                - /url: /listy/8105
              - generic [ref=e436]:
                - text: przez
                - link "Sąd Apelacyjny w Lublinie" [ref=e438] [cursor=pointer]:
                  - /url: /instytucje/sad-apelacyjny-w-lublinie
                - text: w sprawie
                - 'link "Monitoring sądów apelacyjnych #5" [ref=e440] [cursor=pointer]':
                  - /url: /sprawy/monitoring-sadow-apelacyjnych-5
                - time [ref=e441]: 24 sierpnia 2017 15:00
            - paragraph [ref=e443]: "<html xmlns:v=\"urn:schemas-microsoft-com:vml\" xmlns:o=\"urn:schemas-microsoft-com:office:office\" xmlns:w=\"urn:schemas-microsoft-com:office:word\" xmlns:m=\"http://schemas.microsoft.com/office/2004/12/omml\" xmlns=\"http://www.w3.org/TR/REC-html40\"><head><meta http-equiv=\"Content-Type\" content=\"text/html; charset=us-ascii\"><meta name=\"Generator\" content=\"Microsoft Word 15 (filtered medium)\"><style><!--/* Font Definitions */@font-face {font-family:\"Cambria Math\"; panose-1:2 4 5 3 5 4 6 3 2 4;}@font-face {font-family:Calibri; panose-1:2 15 5 2 2 2 4 3 2 4;}/* Style Definitions */p.MsoNormal, li.MsoNormal, div.MsoNormal {margin:0cm; margin-bottom:.0001pt; font-size:11.0pt; font-family:\"Calibri\",sans-serif; mso-fareast-language:EN-US;}a:link, span.MsoHyperlink {mso-style-priority:99; color:#0563C1; text-decoration:underline;}a:visited, span.MsoHyperlinkFollowed {mso-style-priority:99; color:#954F72; text-decoration:underline;}span.Stylwiadomocie-mail17 {mso-style-type:personal-compose; font-family:\"Calibri\",sans-serif; color:windowtext;}.MsoChpDefault {mso-style-type:export-only; font-family:\"Calibri\",sans-serif; mso-fareast-language:EN-US;}@page WordSection1 {size:612.0pt 792.0pt; margin:70.85pt 70.85pt 70.85pt 70.85pt;}div.WordSection1 {page:WordSection1;}--></style><!--[if gte mso 9]><xml><o:shapedefaults v:ext=\"edit\" spidmax=\"1026\" /></xml><![endif]--><!--[if gte mso 9]><xml><o:shapelayout v:ext=\"edit\"><o:idmap v:ext=\"edit\" data=\"1\" /></o:shapelayout></xml><![endif]--></head><body lang=\"PL\" link=\"#0563C1\" vlink=\"#954F72\"><div class=\"WordSection1\"><p class=\"MsoNormal\"><o:p>&nbsp;</o:p></p></div></body></html>"
            - generic [ref=e444]: "1"
          - generic [ref=e446]:
            - heading [level=3] [ref=e447]:
              - link "Pismo Prezesa SA w Łodzi AV-0164-105/17 dotyczące wniosku z dnia 11.08.17 r. o udostępnienie informacji publicznej" [ref=e449] [cursor=pointer]:
                - /url: /listy/8102
              - generic [ref=e450]:
                - text: przez
                - link "Sąd Apelacyjny w Łodzi" [ref=e452] [cursor=pointer]:
                  - /url: /instytucje/sad-apelacyjny-w-lodzi
                - text: w sprawie
                - 'link "Monitoring sądów apelacyjnych #6" [ref=e454] [cursor=pointer]':
                  - /url: /sprawy/monitoring-sadow-apelacyjnych-6
                - time [ref=e455]: 24 sierpnia 2017 15:00
            - generic [ref=e456]:
              - paragraph [ref=e457]: Dzień dobry
              - paragraph [ref=e458]: W załączeniu przesyłam pismo Prezesa Sądu Apelacyjnego w Łodzi z dnia 24 sierpnia 2017 roku o nr AV-0164-105/17 stanowiące odpowiedź na wniosek z dnia 11 sierpnia 2017 r. o udostępnienie informacji publicznej.
              - paragraph [ref=e459]: Pozdrawiam
              - paragraph [ref=e460]: Proszę o potwierdzenie przeczytania tej wiadomości.
              - paragraph [ref=e461]: "Przemysław StalskiZastępca KierownikaOddziału AdministracyjnegoSądu Apelacyjnego w Łodzitel.: (42) 68 50 642fax: (42) 20 91 172e-mail: przemyslaw.stalski@lodz.sa.gov.pl<mailto:przemyslaw.stalski@lodz.sa.gov.pl>www.lodz.sa.gov.pl<http://www.lodz.sa.gov.pl>"
            - generic [ref=e462]: "4"
          - generic [ref=e464]:
            - heading [level=3] [ref=e465]:
              - link "informacja publiczna" [ref=e467] [cursor=pointer]:
                - /url: /listy/8043
              - generic [ref=e468]:
                - text: przez
                - link "Sąd Apelacyjny w Szczecinie" [ref=e470] [cursor=pointer]:
                  - /url: /instytucje/sad-apelacyjny-w-szczecinie
                - text: w sprawie
                - 'link "Monitoring sądów apelacyjnych #9" [ref=e472] [cursor=pointer]':
                  - /url: /sprawy/monitoring-sadow-apelacyjnych-9
                - time [ref=e473]: 23 sierpnia 2017 15:00
            - generic [ref=e474]:
              - paragraph [ref=e475]: Dzień dobry,
              - paragraph
              - paragraph [ref=e476]: w odpowiedzi na pismo z dnia 11 sierpnia 2017 r., w załączeniu uprzejmieprzesyłam zawiadomienie.
              - paragraph
              - paragraph
              - paragraph [ref=e477]: Z poważaniem,
              - paragraph [ref=e478]: Inspektor Emilia Biegańska | Oddział Administracyjny
              - paragraph [ref=e479]: Sąd Apelacyjny w Szczecinie | ul. Mickiewicza 163 | 71-165 Szczecin
              - paragraph [ref=e480]: "tel./ fax: (91) 48 49 481/(91) 48 49 482 | e-mail:<mailto:ebieganska@szczecin.sa.gov.pl> ebieganska@szczecin.sa.gov.pl"
            - generic [ref=e481]: "1"
          - generic [ref=e483]:
            - heading [level=3] [ref=e484]:
              - link "Częściowa odpowiedź na wniosek" [ref=e486] [cursor=pointer]:
                - /url: /listy/8030
              - generic [ref=e487]:
                - text: przez
                - link "Sąd Apelacyjny w Poznaniu" [ref=e489] [cursor=pointer]:
                  - /url: /instytucje/sad-apelacyjny-w-poznaniu
                - text: w sprawie
                - 'link "Monitoring sądów apelacyjnych #7" [ref=e491] [cursor=pointer]':
                  - /url: /sprawy/monitoring-sadow-apelacyjnych-7
                - time [ref=e492]: 23 sierpnia 2017 13:15
            - paragraph [ref=e494]: <!DOCTYPE html><html><head> <meta charset="UTF-8"></head><body><p>Dotyczy wniosku o udostępnienie informacji publicznej - pkt. 3.<br></p><p>Uprzejmie informuję, iż dokumentacja przebiegu i efekt&#243;w kontroli przeprowadzanych w sądach oraz wystąpienia nie są umieszczane na stronie internetowej. Osoby przeprowadzające kontrole nie wydają opinii podmiot&#243;w.<br></p><p>Kierownik Oddziału Kontroli<br></p><p>Lidia Eder<br></p></body></html>
          - generic [ref=e495]:
            - heading [level=3] [ref=e496]:
              - link "informacja publiczna (Adm.-0143-176/17)" [ref=e498] [cursor=pointer]:
                - /url: /listy/8014
              - generic [ref=e499]:
                - text: przez
                - link "Sąd Apelacyjny w Krakowie" [ref=e501] [cursor=pointer]:
                  - /url: /instytucje/sad-apelacyjny-w-krakowie
                - text: w sprawie
                - 'link "Monitoring sądów apelacyjnych #4" [ref=e503] [cursor=pointer]':
                  - /url: /sprawy/monitoring-sadow-apelacyjnych-4
                - time [ref=e504]: 22 sierpnia 2017 15:15
            - generic [ref=e505]:
              - paragraph [ref=e506]: Dzień dobry
              - paragraph [ref=e507]: W wykonaniu polecenia uprzejmie przesyłam pismo Pana Prezesa Sądu Apelacyjnego w Krakowie z dnia 22 sierpnia 2017r., znak Adm.-0143-176/17.Jednocześnie proszę o potwierdzenie otrzymania niniejszej korespondencji.
              - paragraph [ref=e508]: Z poważaniem
              - paragraph [ref=e509]: "Stanisław TronowOddział AdministracyjnySądu Apelacyjnego w Krakowieul. Przy Rondzie 331-547 KrakówTel. 12 417-55-34Fax: 12 417-54-29Wszelkie odpowiedzi proszę kierować na adres oddzial.administracyjny@krakow.sa.gov.pl"
            - generic [ref=e510]: "6"
          - generic [ref=e512]:
            - heading [level=3] [ref=e513]:
              - link "dot. wniosku o udostępnienie informacji publicznej" [ref=e515] [cursor=pointer]:
                - /url: /listy/8002
              - generic [ref=e516]:
                - text: przez
                - link "Sąd Apelacyjny we Wrocławiu" [ref=e518] [cursor=pointer]:
                  - /url: /instytucje/sad-apelacyjny-we-wroclawiu
                - text: w sprawie
                - 'link "Monitoring sądów apelacyjnych #11" [ref=e520] [cursor=pointer]':
                  - /url: /sprawy/monitoring-sadow-apelacyjnych-11
                - time [ref=e521]: 22 sierpnia 2017 14:30
            - generic [ref=e522]:
              - paragraph [ref=e523]: ABI-061-173/17
              - paragraph [ref=e524]: Szanowni Państwo,
              - paragraph [ref=e525]: w odpowiedzi na wniosek o udzielenie informacji publicznej w załączeniu przesyłam pismo Prezesa Sądu Apelacyjnego we Wrocławiu wraz z załącznikiem.
              - paragraph [ref=e526]: Z poważaniem,
              - paragraph [ref=e527]: "Grażyna SochaAdministrator Bezpieczeństwa Informacji/Pełnomocnik ds. ochrony informacji niejawnychw Sądzie Apelacyjnym we Wrocławiutel. (71) 798-77-43faks: (71) 798-77-52"
              - paragraph [ref=e528]: "Uwaga: Niniejsza wiadomość, w szczególności jej treść oraz załączniki, może być poufna. W przypadku gdy nie jest Pan/Pani zamierzonym jej adresatem informujemy, że wszelkie rozpowszechnianie, dystrybucja lub powielanie powyższej wiadomości jest zabronione. Jednocześnie prosimy o powiadomienie nadawcy oraz niezwłoczne usunięcie powyższej wiadomości wraz z załącznikami.Dziękujemy. Sąd Apelacyjny we Wrocławiu."
              - paragraph [ref=e529]: "Confidentiality Notice: This email, particularly its content and any attached files, may be confidential. If you are not an intended recipient, any disclosure, distribution and reproduction of this message is prohibited. In this case please notify the sender immediately and then delete this message and any attachments.Thank you. Court of Appeal in Wrocław."
            - generic [ref=e530]: "2"
          - generic [ref=e532]:
            - heading [level=3] [ref=e533]:
              - link "dot. informacji publicznej" [ref=e535] [cursor=pointer]:
                - /url: /listy/7984
              - generic [ref=e536]:
                - text: przez
                - link "Sąd Apelacyjny w Poznaniu" [ref=e538] [cursor=pointer]:
                  - /url: /instytucje/sad-apelacyjny-w-poznaniu
                - text: w sprawie
                - 'link "Monitoring sądów apelacyjnych #7" [ref=e540] [cursor=pointer]':
                  - /url: /sprawy/monitoring-sadow-apelacyjnych-7
                - time [ref=e541]: 22 sierpnia 2017 10:45
            - generic [ref=e542]:
              - paragraph [ref=e543]: Dzień dobry,
              - paragraph
              - paragraph [ref=e544]: W związku z Państwa wnioskiem o informację publiczną dotyczącą kalendarzaspotkań
              - paragraph [ref=e545]: Pana Prezesa Sądu Apelacyjnego w Poznaniu, informuje , że w miesiącu lipcuPan Prezes
              - paragraph [ref=e546]: nie odbył żadnych spotkań z uwagi na okres urlopowy.
              - paragraph
              - paragraph [ref=e547]: Paulina Winiarska
              - paragraph [ref=e548]: Sekretariat Prezesa Sądu Apelacyjnego
              - paragraph [ref=e549]: w Poznaniu
          - generic [ref=e550]:
            - heading [level=3] [ref=e551]:
              - 'link "Przeczytane: Wniosek o udostępnienie informacji publicznej" [ref=e553] [cursor=pointer]':
                - /url: /listy/7946
              - generic [ref=e554]:
                - text: przez
                - link "Sąd Apelacyjny w Rzeszowie" [ref=e556] [cursor=pointer]:
                  - /url: /instytucje/sad-apelacyjny-w-rzeszowie
                - text: w sprawie
                - 'link "Monitoring sądów apelacyjnych #8" [ref=e558] [cursor=pointer]':
                  - /url: /sprawy/monitoring-sadow-apelacyjnych-8
                - time [ref=e559]: 21 sierpnia 2017 09:15
            - generic [ref=e560]:
              - paragraph [ref=e561]: Twoja wiadomość
              - paragraph [ref=e562]: "Do: Serafin-Kurowiecka Elżbieta Temat: Wniosek o udostępnienie informacji publicznej Wysłano: 11 sierpnia 2017 02:48:11 (UTC+01:00) Sarajewo, Skopie, Warszawa, Zagrzeb"
              - paragraph [ref=e563]: "została przeczytana: 21 sierpnia 2017 09:08:50 (UTC+01:00) Sarajewo, Skopie, Warszawa, Zagrzeb."
          - generic [ref=e564]:
            - heading [level=3] [ref=e565]:
              - link "pismo O.Adm-010-117/17" [ref=e567] [cursor=pointer]:
                - /url: /listy/7940
              - generic [ref=e568]:
                - text: przez
                - link "Sąd Apelacyjny w Katowicach" [ref=e570] [cursor=pointer]:
                  - /url: /instytucje/sad-apelacyjny-w-katowicach
                - text: w sprawie
                - 'link "Monitoring sądów apelacyjnych #3" [ref=e572] [cursor=pointer]':
                  - /url: /sprawy/monitoring-sadow-apelacyjnych-3
                - time [ref=e573]: 18 sierpnia 2017 15:30
            - generic [ref=e574]:
              - paragraph [ref=e575]: W załączeniu przesyłam pismo Prezesa Sądu Apelacyjnego w Katowicach z dnia 18 sierpnia 2017 r. O.Adm-010-117/17 z załącznikiem.
              - paragraph [ref=e576]: specjalista ds. administracyjnychBarbara Gawor
            - generic [ref=e577]: "1"
          - list [ref=e580]:
            - listitem [ref=e581]:
              - generic [aria-hidden]: ←
            - listitem [ref=e582]:
              - generic "Current Page" [ref=e583]: "1"
            - listitem [ref=e584]:
              - link "Page 2 of 2" [ref=e585] [cursor=pointer]:
                - /url: /monitoringi/monitoring-sadow-apelacyjnych/listy/strona-2
                - text: "2"
            - listitem [ref=e586]:
              - link "Next Page" [ref=e587] [cursor=pointer]:
                - /url: /monitoringi/monitoring-sadow-apelacyjnych/listy/strona-2
                - text: →
      - generic [ref=e588]:
        - generic [ref=e589]:
          - text: Sieć Obywatelska - Watchdog Polska ul. Szpitalna 5/5 00-031 Warszawa
          - paragraph [ref=e590]:
            - link "Klauzula RODO" [ref=e591] [cursor=pointer]:
              - /url: https://fedrowanie.siecobywatelska.pl/media_internal/tinycontent/uploads/11_KLAUZULA.-.dane.z.udip.na.stronach.www.i.w.mediach.spol.pdf
        - generic [ref=e592]:
          - text: "silnik:"
          - link "jawne.info.pl" [ref=e593] [cursor=pointer]:
            - /url: http://jawne.info.pl
          - text: "| v1.5.77.deps |"
          - link "GitHub" [ref=e594] [cursor=pointer]:
            - /url: https://github.com/watchdogpolska/feder
          - text: –
          - link "efd8a3b" [ref=e596] [cursor=pointer]:
            - /url: https://github.com/watchdogpolska/feder/compare/efd8a3b2...master
          - text: "|"
          - link "API" [ref=e597] [cursor=pointer]:
            - /url: /api/
        - generic [ref=e599]: Ta strona wykorzystuje cookies.
  - list [ref=e601]:
    - listitem [ref=e602]:
      - link "Ukryj »" [ref=e603] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e604]:
      - link "Toggle Theme" [ref=e605] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e608]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e609]
      - link "Historia /monitoringi/monitoring-sadow-apelacyjnych/listy" [ref=e610] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e611]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e612]
      - link "Wersje Django 5.2.17" [ref=e613] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e614]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e615]
      - 'link "Czas CPU: 280.96ms (301.90ms)" [ref=e616] [cursor=pointer]':
        - /url: "#"
    - listitem [ref=e617]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e618]
      - link "Ustawienia" [ref=e619] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e620]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e621]
      - link "Nagłówki" [ref=e622] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e623]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e624]
      - link "Zapytania LetterListMonitoringView" [ref=e625] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e626]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e627]
      - link "SQL 61 queries in 29.55ms" [ref=e628] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e629]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e630]
      - link "Pliki statyczne 3 użyte plików" [ref=e631] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e632]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e633]
      - link "Templatki monitorings/monitoring_letter_list.html" [ref=e634] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e635]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e636]
      - link "Alerty" [ref=e637] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e638]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e639]
      - link "Cache 2 wywołania w 0.13ms" [ref=e640] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e641]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e642]
      - link "Sygnały 88 odbiorców 15 sygnałów" [ref=e643] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e644]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e645]
      - link "Gmina" [ref=e646] [cursor=pointer]:
        - /url: "#"
    - listitem [ref=e647]:
      - checkbox "Enable for next and successive requests" [ref=e648]
      - generic [ref=e649]: Przechwycone przekierowania
    - listitem [ref=e650]:
      - checkbox "Disable for next and successive requests" [checked] [ref=e651]
      - link "Profilowanie" [ref=e652] [cursor=pointer]:
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