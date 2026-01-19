# CMD-approach-to-PROXY-SVARs
REPLICATION PACKAGE: Proxy-SVAR Identification of two monetary policy shocks 
using minimum distance estimation procedure
Author: Luca Marchesi, Luca Orlando
Date: January 2026
Contact: luca.marchesi5@studio.unibo.it
	 luca.orlando8@studio.unibo.it
================================================================================

OVERVIEW
This package contains the code and data necessary to replicate the findings
presented in the project. The research focuses on:
Proxy-SVAR identification of Monetary Policy Shocks (Target and Path).

--------------------------------------------------------------------------------
DIRECTORY STRUCTURE
--------------------------------------------------------------------------------

/Root           - Main execution scripts (MainCode.m, MainCodeodyssean.m).
/Functions      - Helper functions for optimization and matrix algebra.
/Data           - Macroeconomic datasets (e.g., all_data.mat).
/Instruments    - External instruments (proxies) for identification.
/figures        - Storage for generated IRF plots and series.

--------------------------------------------------------------------------------
SOFTWARE REQUIREMENTS
--------------------------------------------------------------------------------

- Fully replicable in MATLAB.

--------------------------------------------------------------------------------
DATA DESCRIPTION
--------------------------------------------------------------------------------

The primary VAR dataset (1979-2011) includes: 
- FFR: Fed Funds Rate (Target Variable) 
- 1Y: 1-Year Treasury Rate (Path Variable) 
- log(CPI): Log Consumer Price Index 
- log(IP): Log Industrial Production

Instruments (TP1_instr.mat) consist of high-frequency monetary surprises
decomposed into Target and Path factors. 

--------------------------------------------------------------------------------
INSTRUCTIONS FOR REPLICATION
--------------------------------------------------------------------------------

1. PROXY-SVAR (MATLAB):
   - Set the /Root folder as your working directory.
   - Run 'MainCode.m' to estimate the baseline SVAR-IV model. 
   - The script performs Minimum Distance estimation using the Angelini-Fanelli
     (2019) framework and implements the Lakdawala (2019) zero restriction 
     (B[1,2] = 0).
   - Confidence bands are generated via Wild Bootstrap. 

2. ODYSSEAN VS DELPHIC ANALYSIS:
   - Run 'MainCodeodyssean.m' to compare standard proxy results with
     Odyssean-restricted instruments.

--------------------------------------------------------------------------------
KEY RESULTS REPLICATED
--------------------------------------------------------------------------------

- Table 1: Instrument relevance checks (F-statistics and R-squared). 
- Impulse Response Functions (IRFs): Dynamic impact of monetary shocks on 
  macroeconomic aggregates (48-month horizon). 
- Comparison of dynamic responses to a Path shock obtained with and without 
  the Delphic component.

--------------------------------------------------------------------------------
MAIN REFERENCES
--------------------------------------------------------------------------------

- Angelini, G., & Fanelli, L. (2019): 'Exogenous uncertainty and the 
  identification of structural vector autoregressions with external instruments'.
- Lakdawala, S. (2019): 'Decomposing the effects of monetary policy using an 
  external instruments SVAR'.
================================================================================
