# Projekt: Naiwny Klasyfikator Bayesowski

## Opis projektu
W ramach projektu zaimplementowano dwie klasy:
- **Mushroom Classification**: klasyfikacja grzybów jako jadalne lub trujące (cechy kategoryczne).
- **Iris Dataset**: klasyfikacja gatunków irysów (cechy ilościowe).

Szczegółowe opisy modeli znajdują się w notatniku Opis_i_benchmark.NBC.ipynb.

Projekt zakłada wsparcie dla obu typów cech — kategorycznych i ilościowych — poprzez implementację odpowiednich klas.

---

## Struktura projektu
Repozytorium zawiera:
1. **Implementację klasyfikatora:** Klasy MultinomialNaiveBayesClassifier i GaussianNaiveBayesClassifier.
2. **Analizę eksploracyjną danych:** Notatnik analizujący zbiory danych i przygotowujący je do klasyfikacji.
3. **Ocena modelu:** Notatnik + skrypt (benchmark) do ewaluacji jakości klasyfikatora.

---

## Zbiory danych
### 1. **Mushroom Classification**  
- Źródło: [Kaggle](https://www.kaggle.com/datasets/uciml/mushroom-classification).
- Opis: Zawiera cechy kategoryczne, np. kształt kapelusza, kolor, powierzchnia itp. Klasy: **jadalny** lub **trujący**.

### 2. **Iris Dataset**  
- Źródło: [Scikit-learn](https://scikit-learn.org/1.5/auto_examples/datasets/plot_iris_dataset.html).
- Opis: Zawiera cechy ilościowe, np. długość i szerokość kielicha oraz płatka. Klasy: **setosa**, **versicolor**, **virginica**.

## Instalacja
1. Sklonuj repozytorium:
   ```bash
   git clone <link_do_repozytorium>
   cd <nazwa_repozytorium>
    ```
2. Uruchom środowisko wirtualne
    ```bash
    python3 -m venv env
    ```
### Windows, w konsoli batch lub powershell:
    ```bash
    .\env\Scripts\activate.bat
    .\env\Scripts\Activate.ps1
    ```
### Linux 
    ```bash
    source ./env/bin/activate
    ```
3. Zainstaluj wymagane pakiety:
    ```bash
    pip install -r requirements.txt
    ```


---

## Struktura katalogów

├── data/                     # Zbiory danych (np. mushroom.csv)
├── notebooks/                # Notatniki z analizą danych i ewaluacją
├── naive_bayes/              # Implementacja klasyfikatora
│   ├── __init__.py
│   ├── naive_bayes.py        # Główna logika klasyfikatora
├── README.md                 # Dokumentacja projektu
├── requirements.txt          # Wymagane biblioteki
└── tests/                    # Testy jednostkowe dla klasyfikatora


---

## Autorzy

Hubert Miklas
Dominik Kozimor