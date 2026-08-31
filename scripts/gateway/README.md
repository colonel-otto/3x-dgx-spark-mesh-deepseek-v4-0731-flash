# LiteLLM LAN gateway (bigdog :4000)

The gateway that fronts every LAN model. `litellm.service` and
`cutover-to-systemd.sh` are the deployed copies, kept here so the unit is
reviewable and recoverable — the live files are on bigdog at
`/etc/systemd/system/litellm.service` and `$HOME/litellm/`.

## Why a unit exists

LiteLLM ran as a bare `nohup` process. It did not survive a reboot, and a
config edit needed a manual kill + relaunch — a gateway serving a **7-day-stale
config** is a failure this setup has already hit. The unit makes
`sudo systemctl restart litellm` the one obvious way to apply a config change.

## Applying a config change

```bash
# edit $HOME/litellm/config.yaml, then:
sudo systemctl restart litellm
systemctl is-active litellm            # unit blocks until :4000 actually serves
```

`ExecStartPost` runs `litellm-wait-ready`, which polls `/health/liveliness` for
up to 90s, so the unit does not report "started" until the port answers. A
service that starts and stays `running` while wedged is a trap this repo has hit
before; this closes it for the gateway.

That gate is a **script file, not an inline `bash -c`**, and deliberately so:
the first version was inline, passed `systemd-analyze verify`, and still died
with `syntax error near unexpected token`. `verify` checks unit syntax but does
**not** evaluate an embedded shell string — so an inline gate can look healthy
and never actually run. It was caught by executing the gate by hand against the
live gateway, which is the only check that proves it. Both directions are
tested: it exits 0 against a serving port and keeps polling against a dead one.

## Two things the unit deliberately does not do

- **It does not pick the Qwen backend.** `config.yaml` names the routes; the
  unit only runs the process.
- **It does not restart forever.** `StartLimitBurst=5` /
  `StartLimitIntervalSec=300` stop a broken config from spinning. Note these are
  `[Unit]` directives — systemd **silently ignores** them under `[Service]`, so
  verify placement with `systemd-analyze verify` after any edit.

## Verify

```bash
curl -s localhost:4000/v1/models | python3 -m json.tool
sudo systemctl status litellm --no-pager
journalctl -u litellm -n 50 --no-pager
```
