from pathlib import Path


AGENT_SOURCE = Path(__file__).resolve().parents[1] / "agent.py"


def test_inworld_realtime_gate_runs_before_legacy_provider_setup():
    source = AGENT_SOURCE.read_text(encoding="utf-8")

    gate_index = source.index("is_inworld_realtime = _is_inworld_realtime_voice_engine_from_env()")
    bridge_call_index = source.index("await _run_inworld_realtime_voice_engine(ctx)", gate_index)
    openrouter_index = source.index("openrouter_model = os.getenv")
    tts_source_inspection_index = source.index("_log_livekit_tts_source_inspection()", gate_index)
    stt_provider_log_index = source.index("Startup provider config: STT_PROVIDER")
    hume_selection_log_index = source.index("tts_runtime_selection hume_active")
    build_tts_index = source.index("session_tts = build_tts()")
    build_stt_index = source.index('"stt": build_stt()')
    build_vad_index = source.index('"vad": build_vad()')

    assert gate_index < bridge_call_index < openrouter_index
    assert gate_index < bridge_call_index < tts_source_inspection_index
    assert gate_index < bridge_call_index < stt_provider_log_index
    assert gate_index < bridge_call_index < hume_selection_log_index
    assert gate_index < bridge_call_index < build_tts_index
    assert gate_index < bridge_call_index < build_stt_index
    assert gate_index < bridge_call_index < build_vad_index


def test_inworld_realtime_gate_uses_exact_voice_engine_env_value():
    source = AGENT_SOURCE.read_text(encoding="utf-8")

    assert 'os.getenv("VOICE_ENGINE", "").strip().lower() == "inworld_realtime"' in source
