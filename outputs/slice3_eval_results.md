# Slice 3 Evaluation Results

## Inputs
- Contracts: `contract_004, contract_005`
- Cases evaluated: `5`

## Metrics
- Precision@3: `1.000`
- Citation accuracy: `1.000`
- Answer contains expected text: `1.000`
- Answer faithfulness: `not run`; LLM-as-judge is still pending.

## Cases
### q001

- Expected contract: `contract_005`
- Expected page: `2`
- Top citation: `[Document, trang 1, contract_005]`
- Matched citation: `[SCHEDULE 1 TO EXHIBIT 'B', trang 2, contract_005]`
- Matched rank: `2`
- Precision@3 hit: `True`
- Citation correct: `True`
- Answer contains expected text: `True`
- Answer: `not generated`

### q002

- Expected contract: `contract_005`
- Expected page: `1`
- Top citation: `[NON-SOLICITATION, trang 1, contract_005]`
- Matched citation: `[NON-SOLICITATION, trang 1, contract_005]`
- Matched rank: `1`
- Precision@3 hit: `True`
- Citation correct: `True`
- Answer contains expected text: `True`
- Answer: `not generated`

### q003

- Expected contract: `contract_005`
- Expected page: `1`
- Top citation: `[Document, trang 1, contract_005]`
- Matched citation: `[Document, trang 1, contract_005]`
- Matched rank: `1`
- Precision@3 hit: `True`
- Citation correct: `True`
- Answer contains expected text: `True`
- Answer: `not generated`

### q004

- Expected contract: `contract_005`
- Expected page: `1`
- Top citation: `[Document, trang 1, contract_005]`
- Matched citation: `[Document, trang 1, contract_005]`
- Matched rank: `1`
- Precision@3 hit: `True`
- Citation correct: `True`
- Answer contains expected text: `True`
- Answer: `not generated`

### q005

- Expected contract: `contract_005`
- Expected page: `1`
- Top citation: `[NON-SOLICITATION, trang 1, contract_005]`
- Matched citation: `[NON-SOLICITATION, trang 1, contract_005]`
- Matched rank: `1`
- Precision@3 hit: `True`
- Citation correct: `True`
- Answer contains expected text: `True`
- Answer: `not generated`
