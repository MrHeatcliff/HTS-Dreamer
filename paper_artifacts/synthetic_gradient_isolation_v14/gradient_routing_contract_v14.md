# Gradient Routing Contract V14

Hierarchy reconstruction routing flags only affect hierarchy-loss gradients. Temporal, sparse-dynamics, VC, and sparse losses still update the shared trunk normally.

- `hier_recon_update_shared_trunk=false`: hierarchy loss sees detached trunk activations.
- `hier_recon_update_z1_head=false`: hierarchy loss sees detached z1.
- `hier_recon_include_level1_decoder_loss=false`: weighted level-1 hierarchy loss is exactly zero.
