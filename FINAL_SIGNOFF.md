# Final report sign-off

|Block|Member|Status|Notes|
|-|-|-|-|
| 1. Model | Tilesbay Yermukhamed | reviewed | Reviewed Table 1, the own-objective table and the Conclusion against our submitted CP1 PDF. Table 1 lists 325 free y variables, but CP1 has 290 (350 minus 60 pinned by zones) plus 10 t, so 360 in total; please correct this. The M/M/1 blow-up argument comes from Tilesbay's paragraph, while Tamerlan's paragraph rejects the linear f(u) = u, and CP1 writes M/M/1 as 1/(1-u) where the reference model uses u/(1-u). The SA seed 0 in the own-objective table is the run with the tuned T0 and Tend (CP1 objective 6.968945); please say so. The Conclusion should say optimal solutions under the reference model can score worse under CP1, because my ten SA seeds, all with cost z_ref, range from 6.73 to 7.82; please also add that the CP1 witness (5.075) costs 6.041 under the reference model. Everything else matches CP1. |
|2. Comparison|Rakhima Noken|pending||
| 3. Scaling | Tamerlan Aitzhanov | reviewed | Reviewed scaling results for M=100, 200, and 400. Table 3 shows SA and SA-solo runtimes as 0.0 s despite nonzero CSV values. Please correct runtime formatting and clarify LP-rounded-start handling for GA and random search. |
|4. Memo|Danial Kabdulla|reviewed|SA = transfer+swap specifically (the pair’s version, SA-I/SA-T), not SA-solo — it beats SA-solo at every size (0.037% vs 0.403% gap at M=200, 0.039% vs 0.490% at M=400). GA runtime: \~93s mean at M=100, \~107s at M=200, \~176s at M=400 (range 53–195s). Random search at M=400: 8/10 seeds feasible, 2 fail outright — the reported gap (91.7%) is only over the 8 that worked, so it’s actually worse than it looks.|



