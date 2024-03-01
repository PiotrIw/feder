from langchain.prompts import PromptTemplate

letter_evaluation_intro = PromptTemplate.from_template(
    """
    Do {institution} wysłano wniosek o informację publiczną  o treści:
    ```{monitoring_question}```.
    """,
    template_format="f-string",
)

letter_categorization = PromptTemplate.from_template(
    """{intro}
    Oceń odpowiedź z {institution} podaną na końcu, przypisując ją do jednej z kategorii
    z listy poniżej wybierając odpowiednią literę wraz z opisem, nie dodając żadnych
    innych komentarzy. Przy ocenie weź pod uwagę tylko zrozumiały tekst w języku
    Polskim.
    Lista kategorii:
    ```
    A) email jest odpowiedzią z {institution} i zawiera odpowiedzi na pytania z
        wniosku o informację publiczną.
    B) email jest odpowiedzią z {institution} i zawiera odmowę odpowiedzi na
        pytania z wniosku o informację publiczną.
    C) email jest odpowiedzią z {institution} i zawiera informację o
        przedłużeniu terminu na odpowiedź.
    D) email jest potwierdzeniem dostarczenia lub otwarcia maila z {institution}
        i nie zawiera odpowiedzi na pytania z wniosku o informację publiczną.
    E) email jest odpowiedzią z innej instytucji lub na inny wniosek.
    F) email nie jest odpowiedzią z {institution} i jest spamem.
    G) nie można ustalić kategorii odpowiedzi.
    ```
    Odpowiedź z {institution}, jako połączony tekst maila i załączników:
    ```{monitoring_response}```.
    """,
    template_format="f-string",
)

letter_response_normalization = PromptTemplate.from_template(
    """
    Przeanalizuj odpowiedź na wniosek o informację publiczną z {institution} podaną
    na końcu jako tekst do analizy, i uzupełnij wartości odpowiedzi w pliku json z
    pytaniami i miejscami na wartość odpowiedzi:
    ```
    {normalized_questions}
    ```
    Jako wynik podaj poprawnie sformatowany json, bez żadnych dodatkowych wyjaśnień.
    Nie zmieniaj kluczy ani wartości pytań. Jeśli w pliku json z pytaniami istnieje już
    wartość odpowiedzi a tekst zawiera również informację dla odpowiedź na pytanie to
    uzupełnij wartość odpowiedzi w pliku json o brakujące informacje.

    Tekst do analizy:
    ```{monitoring_response}```.
    """,
    template_format="f-string",
)

monitoring_response_normalized_template = PromptTemplate.from_template(
    """
    Przeanalizuj podny poniżej tekst i sprawdź czy jest wnioskiem o udostępnienie
    informacji publicznej. Jeśli tak to zbuduj template do ujednolicenia treści
    odpowiedzi w postaci json:
    {{
        "1": {{"Pytanie":"treść pytania 1", "Odpowiedź":""}},
        "2": {{"Pytanie":"treść pytania 2", "Odpowiedź":""}}
    }}
    Jako klucze w jsonie użyj numerów pytań z zapytania o informację publiczną, a jako
    wartości pytania, treść pytań wraz z oznaczeniem w zapytaniu. W przypadku braku
    oznaczeń pytań w zapytaniu ponumeruj je po kolei od 1.

    Na przykład, rezultatem z treści wniosku o informację publiczną:
    ```
    <br />
    3.&nbsp;&nbsp;Czy dane w poszczeg&oacute;lnych wydziałach urzędu wprowadzane
       są do państwowych system&oacute;w? <br /><br />
    a) Jeśli tak, to jakich? <br />
    b) Jeśli tak, to z kt&oacute;rych system&oacute;w urząd może pobrać dane dla
        swoich cel&oacute;w? <br /><br />
    4. Czy systemy urzędu są zintegrowane z innymi systemami samorządowymi lub
       rządowymi? Jeżeli tak, prosimy o podanie nazw system&oacute;w. <br />
    ```
    powinnien być json:
    {{
        "3": {{"Pytanie":"3. Czy dane w poszczególnych wydziałach urzędu wprowadzane
                         są do państwowych systemów?",
              "Odpowiedź":""}},
        "3a": {{"Pytanie":"a) Jeśli tak, to jakich?",
               "Odpowiedź":""}},
        "4": {{"Pytanie":"4. Czy systemy urzędu są zintegrowane z innymi systemami
                         samorządowymi lub rządowymi? Jeżeli tak, prosimy o podanie
                         nazw systemów.",
              "Odpowiedź":""}}
    }}
    W odpowiedzi podaj tylko poprawnie sformatowany json, bez żadnych dodatkowych
    wyjaśnień.
    Jeśli tekst nie jest wnioskiem o udostępnienime informacji publicznej, lecz jest
    informacją wysłaną do instytucji publicznej to odpowiedzią powinien być pusty
    json: {{}}.
    Tekst do analizy:
    ```{monitoring_template}```.
    """,
    template_format="f-string",
)

monitoring_chat_prompt_template = PromptTemplate.from_template(
    """
    Skorzystaj z poniższych fragmentów kontekstu, aby odpowiedzieć na pytanie na końcu.
    Jeśli nie znasz odpowiedzi, po prostu powiedz, że nie wiesz, nie próbuj wymyślać
    odpowiedzi. Kontekst zawiera zebrane odpowiedzi w formacie JSON na ankietę
    wysłaną do instytucji. Struktura JSONa z odpowiedziami jest następująca:
    ```
    {{
        {{ Nazwa_instytucji_1: {{
            "1": {{"Pytanie":"treść pytania 1", "Odpowiedź":"treść odpowiedzi 1"}},
            "2": {{"Pytanie":"treść pytania 2", "Odpowiedź":"treść odpowiedzi 2"}}
        }},
        {{ Nazwa_instytucji_2: {{
            "1": {{"Pytanie":"treść pytania 1", "Odpowiedź":"treść odpowiedzi 1"}},
            "2": {{"Pytanie":"treść pytania 2", "Odpowiedź":"treść odpowiedzi 2"}}
        }}
    }}
    ```
    Jeśli instytucja nie udzieliła odpowiedzi to JSON dla tej instytucji będzie
    zawierał pusty obiekt: {{}}. Udziel odpowiedzi po polsku.

    Kontekst:
    ---
    {text}
    ---
    Pytanie: {question}
    """,
    template_format="f-string",
)

monitoring_chat_refine_template = PromptTemplate.from_template(
    """
    Uzupełnij odpowiedź na pytanie z poprzedniego zadania, korzystając z poniższych
    fragmentów kontekstu, aby otrzymać wyczerpującą odpowiedzieć na pytanie na końcu.
    Jeśli nie znasz odpowiedzi, po prostu powiedz, że nie wiesz, nie próbuj wymyślać
    odpowiedzi. Kontekst zawiera zebrane odpowiedzi w formacie JSON na ankietę
    wysłaną do instytucji. Struktura JSONa z odpowiedziami jest następująca:
    ```
    {{
        {{ Nazwa_instytucji_1: {{
            "1": {{"Pytanie":"treść pytania 1", "Odpowiedź":"treść odpowiedzi 1"}},
            "2": {{"Pytanie":"treść pytania 2", "Odpowiedź":"treść odpowiedzi 2"}}
        }},
        {{ Nazwa_instytucji_2: {{
            "1": {{"Pytanie":"treść pytania 1", "Odpowiedź":"treść odpowiedzi 1"}},
            "2": {{"Pytanie":"treść pytania 2", "Odpowiedź":"treść odpowiedzi 2"}}
        }}
    }}
    ```
    Jeśli instytucja nie udzieliła odpowiedzi to JSON dla tej instytucji będzie
    zawierał pusty obiekt: {{}}. Udziel odpowiedzi po polsku.

    Odpowiedź z poprzedniego zadania: {existing_answer}
    Kontekst:
    ---
    {text}
    ---
    Pytanie: {question}
    """,
    template_format="f-string",
)
