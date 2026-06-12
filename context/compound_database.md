# DWSIM Compound Name Database

Reference for the Basis Agent. Contains exact DWSIM compound names, common aliases,
multi-component mixture expansions, and a list of unsupported compound classes.

Loaded at startup — **no code changes needed to add new compounds**.
The Basis Agent reads this file directly and rebuilds its lookup table on every run.

---

## How to Update This File

### Adding a single compound
Add one row to the **Single-Component Aliases** table:
```
| DWSIM Exact Name | CAS Number | alias1, alias2, abbreviation | Category | ? |
```
- Use the exact string DWSIM accepts in `sim.AddCompound()`.
- Aliases are comma-separated, case-insensitive.
- Set the Verified column to `✓` once you have confirmed DWSIM accepts the name,
  or `?` if unverified (the agent will attach a warning to unverified entries).

### Adding a mixture alias
Add one row to the **Multi-Component Mixture Aliases** table:
```
| alias | Component1, Component2, Component3 | Brief note |
```
Components must be valid DWSIM names from the single-component table.

### Adding an unsupported compound
Add one row to the **Unsupported Compounds** table:
```
| name or class | colloquial1, colloquial2, abbreviation | Category | Reason DWSIM cannot handle it |
```
The colloquial names are used by the Basis Agent to catch plain-English references (e.g. "brine" → NaCl).

---

## Single-Component Aliases

DWSIM names are case-sensitive. The Verified column (✓ = tested, ? = unverified) indicates
whether the name has been confirmed to work with `sim.AddCompound()`.

| DWSIM Name | CAS | Aliases | Category | Verified |
|---|---|---|---|---|
| Water | 7732-18-5 | water, H2O, dihydrogen monoxide, aqua, steam, liquid water | Polar | ✓ |
| Methanol | 67-56-1 | methyl alcohol, methyl hydroxide, wood alcohol, MeOH, CH3OH, carbinol | Alcohol | ✓ |
| Ethanol | 64-17-5 | ethyl alcohol, grain alcohol, EtOH, C2H5OH, absolute alcohol, ethyl hydroxide | Alcohol | ✓ |
| 1-propanol | 71-23-8 | 1-Propanol, n-propanol, propan-1-ol, propyl alcohol, n-propyl alcohol, propanol | Alcohol | ✓ |
| Isopropanol | 67-63-0 | 2-Propanol, 2-propanol, isopropyl alcohol, IPA, propan-2-ol, rubbing alcohol, iPrOH | Alcohol | ✓ |
| 1-butanol | 71-36-3 | 1-Butanol, n-butanol, n-butyl alcohol, butan-1-ol, butanol, 1-butyl alcohol | Alcohol | ✓ |
| 2-butanol | 78-92-2 | 2-Butanol, sec-butanol, sec-butyl alcohol, butan-2-ol, SBA | Alcohol | ✓ |
| 2-methyl-1-propanol | 78-83-1 | 2-Methyl-1-propanol, isobutanol, isobutyl alcohol, 2-methylpropan-1-ol | Alcohol | ✓ |
| 2-methyl-2-propanol | 75-65-0 | 2-Methyl-2-propanol, tert-butanol, t-butanol, t-BuOH, 2-methylpropan-2-ol | Alcohol | ✓ |
| 1-Pentanol | 71-41-0 | n-pentanol, amyl alcohol, pentan-1-ol | Alcohol | ? |
| Ethylene glycol | 107-21-1 | Ethylene Glycol, EG, ethane-1,2-diol, monoethylene glycol, MEG, glycol | Glycol | ✓ |
| Propylene Glycol | 57-55-6 | PG, propane-1,2-diol, monopropylene glycol, MPG | Glycol | ? |
| Diethylene Glycol | 111-46-6 | DEG, 2,2'-oxydiethanol, 2,2'-oxybisethanol, diethyleneglycol | Glycol | ? |
| Triethylene Glycol | 112-27-6 | TEG, triglycol, triethyleneglycol, 2,2'-ethylenedioxydiethanol | Glycol | ? |
| Glycerol | 56-81-5 | glycerin, glycerine, propane-1,2,3-triol, glycol alcohol | Glycol | ? |
| Methane | 74-82-8 | natural gas (main component), CH4, marsh gas, methyl hydride | Light HC | ✓ |
| Ethane | 74-84-0 | C2H6, ethyl hydride, dimethyl | Light HC | ✓ |
| Propane | 74-98-6 | C3H8, dimethylmethane, propyl hydride | Light HC | ✓ |
| n-Butane | 106-97-8 | butane, normal butane, C4H10 | Light HC | ✓ |
| Isobutane | 75-28-5 | 2-methylpropane, i-butane, methylpropane | Light HC | ✓ |
| n-Pentane | 109-66-0 | pentane, normal pentane, C5H12 | Light HC | ✓ |
| Isopentane | 78-78-4 | 2-methylbutane, i-pentane, methylbutane | Light HC | ? |
| n-Hexane | 110-54-3 | hexane, normal hexane, C6H14 | Light HC | ✓ |
| n-Heptane | 142-82-5 | heptane, normal heptane, C7H16 | HC | ✓ |
| n-Octane | 111-65-9 | octane, normal octane, C8H18 | HC | ✓ |
| n-Nonane | 111-84-2 | nonane, normal nonane, C9H20 | HC | ? |
| n-Decane | 124-18-5 | decane, normal decane, C10H22 | HC | ? |
| n-Dodecane | 112-40-3 | dodecane, laurane, C12H26 | HC | ? |
| n-Hexadecane | 544-76-3 | hexadecane, cetane, C16H34 | HC | ? |
| Cyclohexane | 110-82-7 | hexamethylene, cyclohexyl, hexahydrobenzene | Cyclic HC | ✓ |
| Methylcyclohexane | 108-87-2 | MCH, hexahydrotoluene, cyclohexylmethane | Cyclic HC | ? |
| Cyclopentane | 287-92-3 | pentamethylene, C5H10 | Cyclic HC | ? |
| Benzene | 71-43-2 | benzol, cyclohexatriene, C6H6, phene | Aromatic | ✓ |
| Toluene | 108-88-3 | methylbenzene, toluol, phenylmethane, C7H8 | Aromatic | ✓ |
| o-Xylene | 95-47-6 | 1,2-dimethylbenzene, ortho-xylene | Aromatic | ✓ |
| m-Xylene | 108-38-3 | 1,3-dimethylbenzene, meta-xylene, xylene | Aromatic | ✓ |
| p-Xylene | 106-42-3 | 1,4-dimethylbenzene, para-xylene | Aromatic | ✓ |
| Ethylbenzene | 100-41-4 | phenylethane, ethyl benzene, C8H10 | Aromatic | ✓ |
| Styrene | 100-42-5 | vinylbenzene, ethenylbenzene, styrol, C8H8 | Aromatic | ? |
| Cumene | 98-82-8 | isopropylbenzene, (1-methylethyl)benzene, 2-phenylpropane | Aromatic | ? |
| Naphthalene | 91-20-3 | naphthalin, tar camphor, C10H8 | Aromatic | ? |
| Phenol | 108-95-2 | carbolic acid, hydroxybenzene, benzenol, C6H5OH | Aromatic | ? |
| Hydrogen | 1333-74-0 | H2, dihydrogen, hydrogen gas | Light Gas | ✓ |
| Nitrogen | 7727-37-9 | N2, dinitrogen, nitrogen gas | Light Gas | ✓ |
| Oxygen | 7782-44-7 | O2, dioxygen, oxygen gas | Light Gas | ✓ |
| Argon | 7440-37-1 | Ar | Light Gas | ? |
| Helium | 7440-59-7 | He | Light Gas | ? |
| Carbon dioxide | 124-38-9 | Carbon Dioxide, CO2, carbonic anhydride, carbonic acid gas, dry ice (solid) | Light Gas | ✓ |
| Carbon Monoxide | 630-08-0 | CO, carbon oxide, carbonous oxide | Light Gas | ✓ |
| Hydrogen sulfide | 7783-06-4 | Hydrogen Sulfide, H2S, sulfuretted hydrogen, sewer gas | Light Gas | ✓ |
| Ammonia | 7664-41-7 | NH3, azane, ammonium hydride | Light Gas | ✓ |
| Sulfur Dioxide | 7446-09-5 | SO2, sulfurous anhydride, sulphur dioxide | Light Gas | ? |
| Nitrous Oxide | 10024-97-2 | N2O, laughing gas, dinitrogen monoxide | Light Gas | ? |
| Hydrogen chloride | 7647-01-0 | Hydrogen Chloride, HCl gas, hydrochloric acid gas, muriatic acid gas | Light Gas | ✓ |
| Hydrogen Fluoride | 7664-39-3 | HF, fluoric acid | Light Gas | ? |
| Ethylene | 74-85-1 | ethene, C2H4, elayl | Alkene | ✓ |
| Propylene | 115-07-1 | propene, 1-propene, C3H6, methylethylene | Alkene | ✓ |
| 1-Butene | 106-98-9 | but-1-ene, alpha-butylene, butene-1 | Alkene | ? |
| 2-Butene | 107-01-7 | but-2-ene, beta-butylene, butene-2 | Alkene | ? |
| Isobutene | 115-11-7 | Isobutylene, isobutylene, 2-methylpropene, isobutene, 2-methylpropylene, 2-methyl-1-propene | Alkene | ✓ |
| 1,3-Butadiene | 106-99-0 | butadiene, buta-1,3-diene, bivinyl, divinyl | Alkene | ? |
| Acetylene | 74-86-2 | ethyne, C2H2, narcylene | Alkyne | ? |
| Acetone | 67-64-1 | propanone, dimethyl ketone, 2-propanone, dimethylformaldehyde | Ketone | ✓ |
| Methyl ethyl ketone | 78-93-3 | Methyl Ethyl Ketone, MEK, 2-butanone, ethyl methyl ketone, butan-2-one, butanone | Ketone | ✓ |
| Methyl isobutyl ketone | 108-10-1 | Methyl Isobutyl Ketone, MIBK, 4-methylpentan-2-one, hexone | Ketone | ✓ |
| Cyclohexanone | 108-94-1 | pimelic ketone, ketohexamethylene, cyclohexan-1-one, oxocyclohexane | Ketone | ? |
| Acetaldehyde | 75-07-0 | ethanal, acetic aldehyde, ethyl aldehyde | Aldehyde | ? |
| Formaldehyde | 50-00-0 | methanal, methyl aldehyde, methylene oxide | Aldehyde | ? |
| Acetic acid | 64-19-7 | Acetic Acid, ethanoic acid, glacial acetic acid, AcOH, vinegar (dilute), CH3COOH | Acid | ✓ |
| Formic acid | 64-18-6 | Formic Acid, methanoic acid, hydrogen carboxylate, aminic acid | Acid | ✓ |
| Propionic Acid | 79-09-4 | propanoic acid, methylacetic acid, ethylformic acid | Acid | ? |
| Butyric Acid | 107-92-6 | n-butyric acid, butanoic acid, 1-propanecarboxylic acid | Acid | ? |
| Acrylic acid | 79-10-7 | Acrylic Acid, propenoic acid, acroleic acid | Acid | ✓ |
| Benzoic acid | 65-85-0 | Benzoic Acid, benzenecarboxylic acid, phenylformic acid | Acid | ✓ |
| Methyl acetate | 79-20-9 | Methyl Acetate, methyl ethanoate, acetic acid methyl ester | Ester | ✓ |
| Ethyl acetate | 141-78-6 | Ethyl Acetate, EtOAc, ethyl ethanoate, acetic ether, acetic acid ethyl ester | Ester | ✓ |
| n-Butyl Acetate | 123-86-4 | butyl acetate, n-butyl ethanoate | Ester | ? |
| Isopropyl Acetate | 108-21-4 | isopropyl ethanoate, 2-propyl acetate | Ester | ? |
| Diethyl ether | 60-29-7 | Diethyl Ether, ether, ethyl ether, ethoxyethane, diethyl oxide | Ether | ✓ |
| Dimethyl Ether | 115-10-6 | DME, methoxymethane, wood ether | Ether | ? |
| Tetrahydrofuran | 109-99-9 | THF, oxolane, tetramethylene oxide, oxacyclopentane | Ether | ✓ |
| 1,4-Dioxane | 123-91-1 | dioxane, diethylene dioxide, diethylene ether | Ether | ? |
| Methyl tert-butyl ether | 1634-04-4 | Methyl tert-Butyl Ether, MTBE, tert-butyl methyl ether, 2-methoxy-2-methylpropane | Ether | ✓ |
| Diisopropyl Ether | 108-20-3 | DIPE, isopropyl ether | Ether | ? |
| Ethylene oxide | 75-21-8 | Ethylene Oxide, EO, oxirane, 1,2-epoxyethane | Epoxide | ✓ |
| 1,2-propylene oxide | 75-56-9 | Propylene Oxide, propylene oxide, PO, methyloxirane, 1,2-epoxypropane | Epoxide | ✓ |
| Chloroform | 67-66-3 | trichloromethane, TCM, CHCl3, methane trichloride | Halide | ✓ |
| Dichloromethane | 75-09-2 | DCM, methylene chloride, CH2Cl2, methylene dichloride | Halide | ✗ NOT IN THIS BUILD |
| Carbon tetrachloride | 56-23-5 | Carbon Tetrachloride, tetrachloromethane, CCl4, perchloromethane | Halide | ✓ |
| Vinyl chloride | 75-01-4 | Vinyl Chloride, chloroethylene, chloroethene, VCM, VC | Halide | ✓ |
| Trichloroethylene | 79-01-6 | TCE, trichloroethene, trike | Halide | ? |
| 1,2-Dichloroethane | 107-06-2 | EDC, ethylene dichloride, ethylene chloride, DCE, glycol dichloride | Halide | ? |
| Chlorobenzene | 108-90-7 | monochlorobenzene, phenyl chloride, MCB | Halide | ? |
| Acetonitrile | 75-05-8 | MeCN, methyl cyanide, cyanomethane, ethanenitrile | Nitrile | ✓ |
| Acrylonitrile | 107-13-1 | propenenitrile, vinyl cyanide, AN | Nitrile | ? |
| Dimethyl sulfoxide | 67-68-5 | Dimethyl Sulfoxide, DMSO, methyl sulfoxide, dimethyl sulphoxide | Solvent | ✓ |
| N,n-dimethylformamide | 68-12-2 | Dimethylformamide, DMF, N,N-dimethylformamide, dimethyl formamide | Solvent | ✓ |
| N-Methylpyrrolidone | 872-50-4 | NMP, N-methyl-2-pyrrolidinone, m-pyrol | Solvent | ? |
| Furfural | 98-01-1 | furan-2-carbaldehyde, 2-furaldehyde, furfuraldehyde | Solvent | ? |
| Sulfolane | 126-33-0 | tetramethylene sulfone, thiolane-1,1-dioxide | Solvent | ? |
| Methylamine | 74-89-5 | monomethylamine, methanamine, MMA | Amine | ? |
| Ethylamine | 75-04-7 | monoethylamine, ethanamine | Amine | ? |
| Dimethylamine | 124-40-3 | DMA, N-methylmethanamine | Amine | ? |
| Trimethylamine | 75-50-3 | TMA, N,N-dimethylmethanamine | Amine | ? |
| Ethanolamine | 141-43-5 | MEA, monoethanolamine, 2-aminoethanol, colamine | Amine | ? |
| Diethanolamine | 111-42-2 | DEA, 2,2-iminodiethanamine, bis(2-hydroxyethyl)amine | Amine | ? |
| Triethanolamine | 102-71-6 | TEA, 2,2,2-nitrilotriethanol, tris(2-hydroxyethyl)amine | Amine | ? |
| Aniline | 62-53-3 | phenylamine, aminobenzene, benzenamine | Amine | ? |
| Pyridine | 110-86-1 | azine, azabenzene, 1-azabenzene, pyridin, azine (heterocycle) | Amine | ? |
| Hydrogen Peroxide | 7722-84-1 | H2O2 | Oxidiser | ? |
| Nitric Oxide | 10102-43-3 | NO, nitrogen oxide, nitrogen monoxide | Gas | ? |
| Nitrogen Dioxide | 10102-44-4 | NO2 | Gas | ? |
| 1,1,1,2-Tetrafluoroethane | 811-97-2 | R-134a, HFC-134a, tetrafluoroethane | Refrigerant | ? |
| Difluoromethane | 75-10-5 | R-32, HFC-32, methylene fluoride | Refrigerant | ? |
| Pentafluoroethane | 354-33-6 | R-125, HFC-125 | Refrigerant | ? |
| Chlorodifluoromethane | 75-45-6 | R-22, HCFC-22, Freon-22 | Refrigerant | ? |

---

## Multi-Component Mixture Aliases

When a user specifies one of these aliases, expand to individual DWSIM components.
Typical mole fractions are provided as a starting-point estimate — always verify with
the user for rigorous design. Compositions sum to 1.0.

| Alias | DWSIM Components | Typical Mole Fractions | Notes |
|---|---|---|---|
| natural gas | Methane, Ethane, Propane, n-Butane | 0.85, 0.10, 0.03, 0.02 | North Sea typical. Specify exact fractions for design work. |
| LPG | Propane, n-Butane | 0.60, 0.40 | Liquefied petroleum gas. Commercial blend varies. |
| LNG | Methane, Ethane, Propane | 0.90, 0.08, 0.02 | Methane-dominant. Specify origin for accurate composition. |
| NGL | Ethane, Propane, n-Butane, n-Pentane | 0.40, 0.30, 0.20, 0.10 | Natural gas liquids — field-dependent. |
| air | Nitrogen, Oxygen | 0.79, 0.21 | Dry basis. Argon (<1%) omitted for simplicity. |
| syngas | Carbon Monoxide, Hydrogen | 0.50, 0.50 | H2:CO ratio varies. Specify for your reforming route. |
| biogas | Methane, Carbon Dioxide | 0.60, 0.40 | Typical anaerobic digestion product. |
| flue gas | Carbon Dioxide, Nitrogen, Oxygen, Water | 0.12, 0.75, 0.04, 0.09 | Natural gas combustion, 10% excess air. |
| lean MEA solution | Ethanolamine, Water | 0.18, 0.82 | 30 wt% MEA (mole fraction basis). |
| BTX | Benzene, Toluene, o-Xylene | 0.33, 0.33, 0.34 | Equal-molar approximation — specify actual reformate fractions. |
| naphtha | n-Hexane, n-Heptane, n-Octane | 0.33, 0.34, 0.33 | Heavy naphtha approximation only — not for rigorous design. |
| C4 fraction | n-Butane, Isobutane, 1-Butene, Isobutene | 0.25, 0.25, 0.25, 0.25 | Refinery C4 — specify actual FCC or steam cracker composition. |
| refrigerant blend R-404A | 1,1,1,2-Tetrafluoroethane, Pentafluoroethane, Difluoromethane | 0.44, 0.40, 0.16 | Approximate — verify with refrigerant data sheet. |

---

## Unsupported Compounds

These compounds will fail during DWSIM compound addition or produce physically
meaningless results. The Basis Agent raises an ERROR before the Planner runs.

| Compound / Class | Colloquial / Alias Names | Category | Reason |
|---|---|---|---|
| Sodium chloride, NaCl | salt, table salt, rock salt, brine, common salt, halite, sea salt, nacl | Electrolyte | Electrolyte packages not implemented in wrapper |
| Hydrochloric acid (solution), HCl (aq) | muriatic acid, spirits of salt, acidum salis, hydrochloric acid | Electrolyte | Requires electrolyte NRTL — not supported |
| Sodium hydroxide, NaOH | caustic soda, lye, caustic, soda lye, white caustic, naoh | Electrolyte | Strong base — electrolyte model required |
| Potassium hydroxide, KOH | caustic potash, potash lye, potassium lye, koh | Electrolyte | Strong base — electrolyte model required |
| Sulfuric acid, H2SO4 | oil of vitriol, vitriol, battery acid, sulphuric acid, h2so4 | Electrolyte | Strong acid — electrolyte model required |
| Nitric acid, HNO3 | aqua fortis, spirit of niter, hno3 | Electrolyte | Strong acid — electrolyte model required |
| Phosphoric acid, H3PO4 | orthophosphoric acid, h3po4 | Electrolyte | Partially dissociates — electrolyte model required |
| Calcium chloride, CaCl2 | calcium salt, cacl2 | Electrolyte | Salt — electrolyte model required |
| Ammonium chloride, NH4Cl | sal ammoniac, salmiac, nh4cl | Electrolyte | Salt — electrolyte model required |
| Sodium sulfate, Na2SO4 | Glauber's salt, na2so4 | Electrolyte | Salt — electrolyte model required |
| Potassium chloride, KCl | potash muriate, sylvite, kcl | Electrolyte | Salt — electrolyte model required |
| Sodium carbonate, Na2CO3 | soda ash, washing soda, soda, natron, na2co3 | Electrolyte | Salt — electrolyte model required |
| Polyethylene, PE | polythene, HDPE, LDPE, LLDPE, polyethylene | Polymer | Polymer flash not supported in DWSIM wrapper |
| Polypropylene, PP | polypropene, PP plastic | Polymer | Polymer flash not supported in DWSIM wrapper |
| Polystyrene, PS | styrofoam, expanded polystyrene, EPS | Polymer | Polymer flash not supported in DWSIM wrapper |
| PET, polyethylene terephthalate | polyester, PET plastic, PETE | Polymer | Polymer flash not supported |
| Sugars, carbohydrates | glucose, sucrose, fructose, lactose, starch, sugar, dextrose | Biomolecule | Not in DWSIM standard compound database |
| Proteins, enzymes | albumin, casein, enzyme catalyst | Biomolecule | No thermodynamic model available in DWSIM |
| DNA, RNA | nucleic acid, genetic material | Biomolecule | No thermodynamic model available in DWSIM |
| Ionic liquids (general) | room-temperature ionic liquid, RTIL, molten salt | Ionic Liquid | Not in DWSIM standard compound database |
| Metals (iron, copper, aluminium) | steel, brass, bronze, metal alloy | Metal | Solid-phase components not supported |
| Crude oil, petroleum (unspecified) | black gold, crude, petroleum, fossil fuel | Complex mixture | Must be pseudo-componentised first |

---

## DWSIM Naming Conventions

Notes for the Basis Agent's LLM stage — quirks in DWSIM's compound database:

- **Capitalisation is INCONSISTENT in this build**: verified by `rag/dump_compounds.py` (1488 compounds). Most names are title-case but many follow `1-propanol` (lowercase) or `Methyl ethyl ketone` (sentence-case). Always use the exact name from the DWSIM Name column; `BasisAgent` normalises automatically.
- **n- prefix**: Use `n-Butane`, `n-Hexane` etc. — DWSIM distinguishes normal from branched.
- **Numbered alcohols are lowercase**: `1-propanol`, `2-butanol`, `2-methyl-2-propanol` — not title-case.
- **Acids are sentence-case**: `Acetic acid`, `Formic acid`, `Acrylic acid` — not `Acetic Acid`.
- **Esters/ethers are sentence-case**: `Ethyl acetate`, `Diethyl ether`, `Methyl tert-butyl ether`.
- **Xylenes**: DWSIM accepts `o-Xylene`, `m-Xylene`, `p-Xylene` (duplicates with lowercase variants also accepted).
- **Isobutylene → Isobutene**: This build uses `Isobutene`, not `Isobutylene`. The BasisAgent normalises this.
- **2-Propanol → Isopropanol**: This build uses `Isopropanol` as the canonical name.
- **CO2**: Formula `CO2` not accepted by `AddCompound()` — DWSIM name is `Carbon dioxide` (lowercase d).
- **H2S**: DWSIM name is `Hydrogen sulfide` (lowercase s).
- **DMF**: DWSIM name is `N,n-dimethylformamide` (unusual mixed case) — not `Dimethylformamide`.
- **DMSO**: DWSIM name is `Dimethyl sulfoxide` (lowercase s).
- **Propylene oxide**: DWSIM name is `1,2-propylene oxide` — not `Propylene Oxide` or `Propylene oxide`.
- **DCM (Dichloromethane)**: NOT present in this DWSIM installation. Benchmarks using DCM are unsolvable with this build.
- **Generating the validated compound list**: `rag/dump_compounds.py` was run inside the Docker container and produced the verified 1488-compound list. Re-run after DWSIM upgrades.
