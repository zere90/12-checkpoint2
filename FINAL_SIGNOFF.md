# Final report sign-off

|Block|Member|Status|Notes|
|-|-|-|-|
|1. Model|Tilesbay Yermukhamed|pending||
|2. Comparison|Rakhima Noken|pending||
| 3. Scaling | Tamerlan Aitzhanov | reviewed | Reviewed scaling results for M=100, 200, and 400. Table 3 shows SA and SA-solo runtimes as 0.0 s despite nonzero CSV values. Please correct runtime formatting and clarify LP-rounded-start handling for GA and random search. |
|4. Memo|Danial Kabdulla|reviewed|SA = transfer+swap specifically (the pair’s version, SA-I/SA-T), not SA-solo — it beats SA-solo at every size (0.037% vs 0.403% gap at M=200, 0.039% vs 0.490% at M=400). GA runtime: \~93s mean at M=100, \~107s at M=200, \~176s at M=400 (range 53–195s). Random search at M=400: 8/10 seeds feasible, 2 fail outright — the reported gap (91.7%) is only over the 8 that worked, so it’s actually worse than it looks.|



