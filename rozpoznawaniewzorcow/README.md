# Lokalne anomalie i maskowanie odstających obiektów w LOF

Projekt realizowany w ramach przedmiotu **Rozpoznawanie wzorców** (Studia II stopnia).  
**Publikacja bazowa:** M. M. Breunig, H.-P. Kriegel, R. T. Ng, J. Sander, *LOF: Identifying Density-Based Local Outliers*, SIGMOD 2000.

## Protokół badania (Etap 1 / Checkpoint 1)
* **Generator danych:** Dwa syntetyczne skupiska gaussowskie o różnej gęstości (`C1`: $n=250, \sigma=0.4$; `C2`: $n=200, \sigma=2.2$) oraz $20$ oznaczonych anomalii ($5$ lokalnych w sąsiedztwie `C1` oraz $15$ globalnych). Łącznie $N = 470$ obiektów ($\le 500$).
* **Stałe parametry:** Liczba sąsiadów $k = 20$, metryka odległości: euklidesowa (`euclidean`), główne ziarno generatora `SEED = 42`.
* **Metryka oceny rankingu:** Average Precision (AP) wyliczana na ciągłych wynikach `LOF = -negative_outlier_factor_`.

## Instrukcja uruchomienia
1. Instalacja zależności:
   ```bash
   pip install -r requirements.txt