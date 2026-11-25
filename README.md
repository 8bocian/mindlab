# MindLab: wstrzykiwanie myśli i podgląd „myślenia” modelu

Jedna komenda wynajmuje GPU na Hyperstacku, instaluje wszystko, ładuje model i otwiera w przeglądarce dashboard, w którym:

- **suwakami wstrzykujesz pojęcia** (np. „ocean”, „fear”) do wybranych warstw modelu, na żywo, także w trakcie pisania odpowiedzi,
- **widzisz, o czym model „myśli”**: mierniki obecności każdego pojęcia przy każdym tokenie, wykres w czasie, top 5 kandydatów na następny token,
- **logit lens**: tabela pokazująca, co przewidziałaby każda warstwa (a przy modelach z pętlą, jak Inkling 2 Mini, każde przejście pętli osobno),
- dla modeli z pętlą wybierasz też, **w które przejście** wstrzykujesz.

Po Ctrl+C maszyna jest hibernowana albo usuwana. Strażnik na samej VM pilnuje tego także wtedy, gdy laptop zaśnie.

---

## 1. Jednorazowe przygotowanie (ok. 10 min)

1. **Konto na [Hyperstack](https://www.hyperstack.cloud/)**: załóż konto i doładuj saldo, np. $20.
2. **Klucz API**: w panelu *API Keys* wygeneruj klucz.
3. **Environment**: w panelu utwórz environment w wybranym regionie, np. `default-CANADA-1` albo europejskim. Region decyduje, jakie GPU są dostępne.
4. **Klucz SSH**: jeśli nie masz, w terminalu wpisz:
   ```bash
   ssh-keygen -t ed25519
   ```
   (Windows: PowerShell, ta sama komenda). Nie musisz go wgrywać, `lab.py` zrobi to sam.
5. **Pakiety lokalnie** (Python 3.9+):
   ```bash
   pip install requests paramiko
   ```
   Potrzebny jest też klient `ssh`. Na macOS i Linuksie jest zawsze, na Windows 10/11 jest wbudowany.
6. **Konfiguracja**:
   ```bash
   cp lab.env.example lab.env
   ```
   W `lab.env` uzupełnij przynajmniej `HYPERSTACK_API_KEY` i `ENVIRONMENT`.

## 2. Sprawdzenie

```bash
python lab.py check
```
Wypisze regiony, klucze, dostępne GPU (strzałka ← pokazuje, co zostanie wybrane) i obrazy systemu. Nic nie wynajmuje, nic nie kosztuje.

## 3. Start

```bash
python lab.py up
```
Co się dzieje po kolei:

1. Tworzy się VM z RTX PRO 6000 (96 GB). Jeśli jest zahibernowana z poprzedniej sesji, zostaje wybudzona.
2. Skrypt wgrywa kod i uruchamia strażnika bezczynności.
3. Instaluje Pythona i PyTorcha dopasowanego do sterownika, pobiera model. Za pierwszym razem trwa to ok. 5–15 min, po hibernacji ok. 1 min.
4. Startuje serwer, otwiera tunel SSH, a przeglądarka otwiera `http://localhost:8000`.

Dashboard **nie jest wystawiony do internetu**. Działa tylko przez szyfrowany tunel SSH z twojego komputera.

**Ctrl+C w terminalu kończy sesję** i wykonuje `END_ACTION` z `lab.env`, domyślnie hibernację.

Pozostałe komendy:

| komenda | co robi |
|---|---|
| `python lab.py status` | stan VM (ostrzega, jeśli nalicza pełną stawkę) |
| `python lab.py hibernate` | hibernacja: płacisz tylko za dysk |
| `python lab.py down` | usunięcie: nic nie płacisz, następny start od zera |
| `python lab.py logs` | log serwera i strażnika z VM |
| `python lab.py ssh` | wypisuje komendę SSH do VM |
| `python lab.py up --keep` | po Ctrl+C VM zostaje włączona (uważaj na koszty) |

## 4. Jak korzystać z dashboardu

**Pierwszy eksperyment: wstrzyknięcie myśli**

1. Kliknij gotowy eksperyment *„Co teraz myślisz?”* i wyślij go **bez wstrzykiwania**. To twój punkt odniesienia.
2. Ustaw suwak `ocean` na **0,3** i wyślij to samo jeszcze raz. Porównaj odpowiedzi.
3. Podnoś siłę: 0,5, 0,8… Gdzieś pojawi się punkt, w którym model zaczyna pisać o oceanie, a dalej się „sypie”, czyli powtarza słowa albo pisze bełkot. To okno działania steeringu.
4. Spróbuj **wartości ujemnych**: odejmowanie pojęcia, np. `fear` = −0,4 w pytaniu o horror.

**Test introspekcji** (jak w badaniach Anthropic): wybierz preset *„Test introspekcji”*, wstrzyknij pojęcie i sprawdź, czy model je „zauważy” i nazwie. Potem powtórz bez wstrzykiwania, bo kontrola jest kluczowa: model może „zgadywać” wstrzyknięcie także wtedy, gdy go nie ma.

**Gdzie wstrzykiwać**: suwaki *od/do warstwy*. Zwykle najlepiej działają warstwy w ok. ⅓–½ głębokości modelu. Im więcej warstw naraz, tym mniejszej siły potrzeba.

**Odczyt**:
- *Mierniki (z-score)*: ≈0 oznacza neutralne tło, >3 wyraźną obecność pojęcia. Odczyt z tej samej warstwy, w którą wstrzykujesz, zawsze pokaże wysoki wynik, bo sam go tam dodałeś. Ciekawsze jest ustawienie **warstwy odczytu później** i sprawdzenie, czy pojęcie „przeżywa” w głąb sieci.
- *Kolor tokenów → pojęcie: X*: podświetla, w których miejscach odpowiedzi pojęcie było „aktywne”. Działa też **bez wstrzykiwania**: napisz o morzu i zobacz, które słowa zapalają `ocean`.
- *Logit lens*: kliknij token w rozmowie, żeby go przypiąć. Tabela pokaże, jak decyzja o tym tokenie dojrzewała przez warstwy.
- *Nowe pojęcie*: wpisz słowo i kliknij „Dodaj”. Wektor policzy się w kilka sekund. Wartość `sep` mówi, jak dobrze pojęcie odróżnia się od tła, czyli największą separację w odchyleniach standardowych.

**Inkling 2 Mini (pętla)**: w `lab.env` ustaw `MODEL=thinkingmachines/Inkling-2-Mini` i zrób `python lab.py up`. W panelu „Gdzie wstrzykiwać” pojawią się checkboxy **przejść pętli**, a logit lens pokaże wiersze `L12·p1`, `L12·p2`, `L12·p3`. Ciekawe pytania:
- Czy wstrzyknięcie tylko w 1. przejście działa inaczej niż tylko w 3.?
- W którym przejściu odpowiedź „krzepnie” w logit lensie?

## 5. Tryb dla kilku osób: frontend na Vercelu, backend na Hyperstacku

```
znajomi ──► mindlab.vercel.app (index.html + config.js, zawsze online)
              │  fetch + WebSocket, hasło
              ▼
         https://mindlab-api.twojadomena.pl ──(Cloudflare Tunnel)──► VM Hyperstack :8000
```

- Każdy ma **własne suwaki**: wstrzykiwanie jednej osoby nie wpływa na odpowiedzi innych.
- Wektory pojęć są w **wspólnej bibliotece** (liczone raz), ale każdy widzi tylko swoją listę. „×” chowa pojęcie tylko u ciebie.
- GPU generuje jedną odpowiedź naraz, reszta czeka w **kolejce** i widzi swoją pozycję.
- Gdy VM jest wyłączona, strona pokazuje „serwer jest teraz wyłączony” i sama się połączy po starcie.
- Strażnik hibernuje VM po `IDLE_MINUTES` bez aktywności. Otwarta, ale nieużywana karta nie podtrzymuje VM.

Instrukcja konfiguracji krok po kroku jest w sekcji 6.

## 6. Konfiguracja trybu publicznego (jednorazowo)

**A. Domena w Cloudflare.** Załóż konto na cloudflare.com, dodaj swoją domenę (*Add a domain*) i zmień nameservery u rejestratora na te, które poda Cloudflare. Domenę możesz też kupić od razu w Cloudflare (*Domain Registration*).

**B. Tunel.** W panelu Cloudflare: *Zero Trust → Networks → Tunnels → Create a tunnel → Cloudflared*, nazwa np. `mindlab`. Na ekranie instalacji skopiuj sam **token** (długi ciąg po `--token`). Niczego nie instalujesz, zrobi to `lab.py`. W zakładce *Public Hostname* dodaj: subdomena `mindlab-api`, domena twoja, *Service* `HTTP` → `localhost:8000`.

**C. `lab.env`:**
```
CLOUDFLARE_TUNNEL_TOKEN=eyJh...
ACCESS_PASSWORD=jakies-dlugie-haslo-dla-znajomych
ALLOWED_ORIGINS=https://mindlab.vercel.app
PUBLIC_URL=https://mindlab.vercel.app
```

**D. `server/static/config.js`:** wpisz adres tunelu:
```js
window.MINDLAB_API = "https://mindlab-api.twojadomena.pl";
```

**E. Vercel.** Wypchnij repo na GitHuba (`lab.env` jest w `.gitignore`). Na vercel.com: *Add New → Project →* wybierz repo, **Root Directory: `server/static`**, Framework Preset: *Other*, *Deploy*. Dostaniesz adres typu `mindlab-xyz.vercel.app`. Jeśli różni się od tego w `ALLOWED_ORIGINS`, popraw go w `lab.env`.

**Pierwsza instalacja i każda zmiana kodu lub `lab.env`:**
```bash
python lab.py start       # instaluje VM: serwer, tunel i strażnik jako usługi systemd, potem kończy
```

**Na co dzień, bez `lab.py` i bez twojego komputera:** w panelu Hyperstack (*Virtual Machines*):
- **włączanie:** *Restore* (z hibernacji). Po ok. 2–3 min od stanu ACTIVE dashboard działa sam,
- **wyłączanie:** *Hibernate*. **Nie używaj *Stop*:** zatrzymana VM nalicza pełną stawkę,
- strażnik i tak zahibernuje VM po `IDLE_MINUTES` bez aktywności albo po `MAX_HOURS`, licząc od każdego startu.

Pomocniczo z terminala: `python lab.py status` (czy nalicza), `python lab.py hibernate`, `python lab.py logs` (logi serwera, strażnika i tunelu).

**Gdy *Restore* w panelu odmawia („Not Enough Stock”):** uruchom `python lab.py start`. Spróbuje wybudzić VM, a jeśli brak kart tego typu, utworzy nową VM na pierwszej dostępnej karcie z `GPU_FALLBACK` (domyślnie A100 → L40 → H100, bez wersji spot). To pełna instalacja, ok. 10–15 min. Starą VM usunie dopiero, gdy nowa działa (`DELETE_OLD_ON_FALLBACK=1`). `python lab.py check` pokazuje, które karty z listy są teraz dostępne.

## 7. Jak to działa (w skrócie)

- **Wektor pojęcia**: model czyta kilka zdań z danym słowem („Tell me about ocean.”, „I keep thinking about ocean.”…) i te same zdania z 48 neutralnymi słowami. Różnica średnich stanów wewnętrznych w każdej warstwie daje kierunek „ocean”.
- **Wstrzyknięcie**: do stanu każdego tokena w wybranych warstwach dodawany jest ten kierunek, przeskalowany przez typową długość stanu w danej warstwie. Stąd suwak „0,3” oznacza ok. 30% typowej długości.
- **Odczyt**: rzut stanu na kierunek pojęcia, porównany z rozkładem dla neutralnych zdań (z-score).
- **Logit lens**: stan z każdej warstwy przechodzi przez końcową normalizację i warstwę wyjściową modelu, czyli „co by powiedział, gdyby skończył myśleć tutaj”.
- **Pętle**: serwer liczy, ile razy każda warstwa uruchamia się na jeden token. Jeśli więcej niż raz, model ma pętlę i każde przejście jest osobnym miejscem.

Całość działa na zwykłym `transformers` z hookami PyTorcha. vLLM nie jest tu potrzebny: przy kilku osobach wystarcza kolejka, a takie podejście daje pełną kontrolę i działa z prawie każdym modelem.

## 6. Koszty i bezpieczeństwo

- RTX PRO 6000 kosztuje ok. **$1,85/h**, rozliczane co minutę. Weekend (10 h) to ok. $18,50.
- **Stan SHUTOFF (zatrzymana VM) też nalicza pełną stawkę.** Dlatego `lab.py` hibernuje albo usuwa, nigdy nie „wyłącza”.
- Hibernacja: dysk ok. $0,07/GB miesięcznie, czyli przy 100 GB ok. $7/mies.
- **Strażnik** na VM hibernuje ją (albo usuwa, zależnie od `END_ACTION`) po `IDLE_MINUTES` bez aktywności w dashboardzie lub po `MAX_HOURS`. Działa nawet wtedy, gdy twój komputer jest wyłączony. Żeby mógł to zrobić, na VM zapisany jest twój klucz API (plik z uprawnieniami 600). Jeśli ci to nie odpowiada, ustaw `WATCHDOG=0` i pilnuj sam.
- Gdy saldo spadnie do $0, Hyperstack sam hibernuje VM, ale saldo może zejść poniżej zera. Doładowuj z umiarem.
- Raz w tygodniu warto zajrzeć do panelu Hyperstack albo wpisać `python lab.py status`.

## 7. Gdy coś nie działa

| problem | co zrobić |
|---|---|
| `HTTP 401` | zły `HYPERSTACK_API_KEY` |
| brak GPU w regionie | `python lab.py check`, potem zmień `ENVIRONMENT` (region) albo `GPU_MATCH` (np. `H100`) |
| VM nie startuje / nie wybrano obrazu | ustaw ręcznie `FLAVOR_NAME` i `IMAGE_NAME` z listy w `check` |
| SSH nie łączy | sprawdź `SSH_USER` (zwykle `ubuntu`), `SSH_PRIVATE_KEY`, `SSH_ALLOW_CIDR` |
| błąd przy instalacji | przy pytaniu wybierz `z` (zostaw), potem `python lab.py ssh` i zajrzyj do `~/mindlab/setup.log` |
| **Inkling się nie ładuje** | Spróbuj kolejno: `EXTRA_PIP=git+https://github.com/huggingface/transformers` (najnowszy transformers), potem `TRUST_REMOTE_CODE=1`. Jeśli dashboard pokazuje dziwne warstwy, podaj ręcznie `LAYERS_PATH` i `NORM_PATH` (nazwy znajdziesz przez `print(model)`). Wagi NVFP4 wymagają GPU Blackwell, a RTX PRO 6000 nim jest. |
| model za duży | wybierz mniejszy albo GPU z większą pamięcią (`GPU_MATCH`) |

`lab.py` obsługuje tylko te polecenia API Hyperstacka, które są opisane w ich dokumentacji. Nie mogłem ich jednak przetestować na prawdziwym koncie, tylko na atrapie API. Jeśli przy pierwszym uruchomieniu coś się nie zgodzi, najczęściej będzie to nazwa obrazu lub flavora, które ustawisz ręcznie w `lab.env`.

## Bez chmury (własne GPU, Colab)

Na dowolnej maszynie z kartą NVIDIA:
```bash
cd server
pip install torch -r requirements.txt
MODEL=Qwen/Qwen3-4B python app.py      # potem otwórz http://localhost:8000
```

## Struktura

```
lab.py              # orkiestracja: check / up / down / hibernate / status / logs / ssh
lab.env.example     # konfiguracja (skopiuj do lab.env)
server/engine.py    # hooki, wektory pojęć, wstrzykiwanie, logit lens, generowanie
server/app.py       # FastAPI + WebSocket
server/static/index.html   # dashboard
vm/setup_vm.sh      # instalacja na VM (idempotentna)
vm/start_server.sh  # start serwera i strażnika na VM
vm/watchdog.py      # strażnik bezczynności (hibernuje/usuwa VM przez API)
```
