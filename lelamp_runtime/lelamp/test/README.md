# Manual hardware checks

These modules are operator-run hardware checks, not automated pytest tests.
They may play audio, record from a microphone, drive RGB LEDs, or move the
lamp's motors. Run only the named module from `lelamp_runtime/` with the
expected device connected; do not collect this directory with pytest.

- `check_audio`: speaker and microphone loopback
- `check_rgb`: LED colors and priority behavior
- `check_motors`: selected recording playback and safe torque release

Recording analysis that does not connect to hardware lives in
`lelamp.tools.analyze_recordings`.
