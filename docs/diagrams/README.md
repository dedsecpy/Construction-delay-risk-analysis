# Architecture diagrams

| Diagram | Mermaid source | Rendered SVG |
|---------|----------------|--------------|
| System architecture | `01_system_architecture.mmd` | `01_system_architecture.svg` |
| Complete Bayesian network (+ PDTs) | `02_bayesian_network.mmd` | `02_bayesian_network.svg` · `02_bayesian_network_complete.svg` · `02_bayesian_network_from_mermaid.svg` |
| Train / val / test | `03_train_val_test.mmd` | `03_train_val_test.svg` |
| Inspection loop | `04_inspection_loop.mmd` | `04_inspection_loop.svg` |
| Early-warning | `05_early_warning.mmd` | `05_early_warning.svg` |

Re-render all SVGs from Mermaid:

```powershell
npx --yes @mermaid-js/mermaid-cli@11 -i docs/diagrams/01_system_architecture.mmd -o docs/diagrams/01_system_architecture.svg -b transparent
npx --yes @mermaid-js/mermaid-cli@11 -i docs/diagrams/02_bayesian_network.mmd -o docs/diagrams/02_bayesian_network.svg -b transparent
npx --yes @mermaid-js/mermaid-cli@11 -i docs/diagrams/03_train_val_test.mmd -o docs/diagrams/03_train_val_test.svg -b transparent
npx --yes @mermaid-js/mermaid-cli@11 -i docs/diagrams/04_inspection_loop.mmd -o docs/diagrams/04_inspection_loop.svg -b transparent
npx --yes @mermaid-js/mermaid-cli@11 -i docs/diagrams/05_early_warning.mmd -o docs/diagrams/05_early_warning.svg -b transparent
```
