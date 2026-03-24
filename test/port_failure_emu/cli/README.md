# CLI for Port Failure Emulation

This command-line interface (CLI) allows users to activate and manage port failure emulation on specified devices.

The CLI provides feedback on its operations, including errors and status updates.

To see TRACE prints, run with `PFE_DEBUG=1` in the environment.

For example:

```bash
PFE_DEBUG=1 ./pfe_cli --mode RDMA --devices mlx5_0,mlx5_1
```
