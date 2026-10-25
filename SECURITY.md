# Security policy

## Scope

The testbed is designed for an isolated laboratory network: the access point and the management network hand out no default route and no upstream resolver, the devices are reachable only from the edge server, and the NNPM web application is served on the management network. The software is not hardened for exposure to the public internet; do not expose NNPM or the devices to untrusted networks.

Points that deserve attention when deploying:

- **SSH credentials.** The edge server authenticates to the devices with a key (`ssh_public_key_path`) or a password taken from an environment variable; neither is stored in the repository or in `config.yaml`.
- **Privileges on the devices.** Only `tc` and the heterogeneity module run with `sudo -n`, through a sudoers rule that names exactly those commands. Nothing else needs root.
- **NNPM access.** Set `nnpm_basic_auth_env` in `config.yaml` to require HTTP basic authentication. Flask's debug mode is off by default and must never be enabled on the laboratory network.
- **Package integrity.** Every Python dependency is installed with `--require-hashes` from the lock files, and the local mirror has a manifest with the SHA-256 of every file (`python -m src.provisioning.mirror verify`).

## Supported versions

Security fixes are made on the `main` branch and included in the next release. The latest release is the supported one.

## Reporting a vulnerability

Please do not open a public issue for a vulnerability. Report it privately through GitHub's "Report a vulnerability" form on the Security tab of the repository, or by e-mail to the maintainer listed in [CITATION.cff](CITATION.cff) and `pyproject.toml`. Include the affected component, the steps to reproduce and, if possible, a suggested fix. You will receive an acknowledgement within a week, and a fix or a mitigation will be published as soon as it is available.
