
##### 0 Premessa.

  * Lavoro in corso d'opera.
  * Analisi di fattibilità non ancora completa 
  * Risultati preliminari solo su dati "fittzi" (cioe' generati dal SW proprietario Colibrì).
  * Prime impressioni: l'approccio potrebbe funzionare.

===================
##### 1. dati generati da Colibì.
    
  * Un insieme di ~ 2000 formule, corrispondenti ad altrettante chip della cartella NCS, generate dal SW proprietario Colibì
  sono usate come dati sperimentali, cioè come fossero dati misurati.

  * Per ogni punto di questo insieme, abbiamo quindi (oltre ad altre info di corredo):
  
    * le 3 coordinate colorimetriche
    * la formula corrispondente, generata da Colibrì, che consideriamo come dato misurato per la sperimentazione.

===================
##### 2. Suddivisione dei dati sperimentali in sottoinsiemi.

  l'insieme complessivo dei dati viene suddiviso in N sottoinsiemi indipendenti, ciascuno relativo ad una specifica 
  combinazione di tre pigmenti + il bianco.

  Ogni sottoinsieme copre una area ristretta dello sapzio colore, corrispondente ad una serie di chip NCS.

  Per ogni sottoinsieme, si ha un problema di intepolazione di una funzione a 4 valori (le 4 quantità 
  di pigmenti della formula) su uno spazio di input a 3 componenti (le 3 coordinate colorimetriche).

  Ogni sottoinsieme viene trattato a sè, in modo indipendente.

===================
##### 3. Campionamento a rotazione all'interno dei sottoinsiemi.

  Per ogni sottoinsieme, a rotazione, si estrae un punto e lo si usa come elemento di 
  test e, usando tutti gli altri come training-set, si procede alla interpolazione.
  Poi si calcola la differenza tra la quantità interploata e quella misurata, per ogni pigmento.

  La somma dei quadrati di queste differenze è usata per la stima della performance 
  dell'algoritmo di interpolazione al variare dei parametri

===================
##### 4. Blackpage.

  tool di esplorazione/sperimentazione dell'algoritmo e interfaccia di visualizzazione dei risultati 

===================
##### 5. Next Step.

  * verifica della correlazione evntuale tra errori della stima del punto e: 
    - parametri geometrici del cluster di punti nello sapzio colore (distanza media, distanmza dal baricentro, ...).
    - differenza tra le coordinate colorimetriche effettive (misure lab) e quelle generate da Colibrì (rumore di Colibrì)

  * misure lab per i punti con errori maggiori nella stima, per verificare la corrispondenza tra errore, DeltaE_00 e differenza percepita.

  * misure lab per una griglia di punti per sottoinsieme (i.e. quadrupletta di pigmenti) e analisi corrispondente.


===================
