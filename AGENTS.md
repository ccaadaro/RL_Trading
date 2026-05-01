# Agent Instructions

## Repository workflow

- Never work directly on main or master.
- Always inspect `git status` before modifying files.
- Use one branch per task.
- Prefer small, reviewable changes.
- Do not reformat unrelated files.
- Do not force-push.
- Do not merge pull requests without human approval.

## Handoff protocol

Before finishing, summarize:
- Files changed.
- Tests run.
- Remaining risks.
- Next recommended step.

## Handoff: Shadow-Live Hardening (2026-04-30)

### Problemas Encontrados
1.  **Divergencia de Archivos**: El bot estaba cargando la estrategia desde `/home/nosferatu/freqtrade/user_data/strategies/` mientras el desarrollo ocurría en `RL_Trading/`. Se sincronizaron ambos archivos.
2.  **Crashes de Inicialización**: El motor de señales (`InstitutionalSignalEngine`) sufría de `AttributeError: _bar_buffer` y `NoneType` en `_executor` porque se inicializaban en el método `run()` (hilo) en lugar de en el constructor, causando condiciones de carrera.
3.  **Telemetría Silenciosa**: El socket ZMQ PUB descartaba mensajes por un `SNDHWM` muy bajo y la falta de suscriptores activos durante el arranque ("slow joiner").

### Soluciones Implementadas
-   **Constructor Robusto**: Se movió la inicialización de `_bar_buffer`, `_executor` y `_last_hb` al `__init__`.
-   **Sincronización de Buffers**: El motor ahora hereda el buffer del `provider` en el arranque para tener `n_bars=2000` inmediatamente.
-   **Heartbeat Constante**: Se implementó un bucle ZMQ no bloqueante que garantiza un pulso de diagnóstico cada 2 segundos, incluso si no llegan barras nuevas.
-   **Seguridad de Procesos**: Se refactorizó `launch_shadow_stack.sh` para usar `pkill` por ruta absoluta, evitando matar el servidor de Antigravity.

### Estado Actual (2026-04-30)
-   **Pipeline**: Activo y estable.
-   **Telemetría**: Verificada en puerto 5556 con mensajes `shadow_selector` fluyendo.
-   **Monitor**: `monitor_shadow_selector.py` ahora recibe datos correctamente.

### Siguiente Paso Recomendado
-   Verificar la precisión de las decisiones del selector de régimen en el dashboard conforme lleguen nuevas barras de dólar.

---

## Handoff: Pipeline Execution Fixes (2026-05-01)

### Contexto
El pipeline llevaba al menos varios días ejecutándose pero **nunca completándose** (cero logs "Pipeline OK" en producción). Tres bugs silenciosos en cadena lo bloqueaban completamente.

### Bugs Encontrados y Corregidos (commit `11d2a39`)

#### Bug 1 — `UnboundLocalError: prob_v1` en shadow selector
- **Archivo**: `InstitutionalDollarStrategy.py`, shadow selector (~línea 371)
- **Causa**: Cuando `is_stale=True` o `spread_bps>15`, el branch de stale asignaba `alpha_prob_final`, `selected_model`, `selector_reason` pero NO asignaba `prob_v1`/`prob_v2`. El dict de telemetría a continuación referenciaba `prob_v1` → `UnboundLocalError`.
- **Efecto**: Warning logueado → ejecución continúa → pero Bug 2 impedía llegar a "Pipeline OK".
- **Fix**: Inicializar `prob_v1=0.5`, `prob_v2=0.5` ANTES del bloque `if is_stale`.

#### Bug 2 — `available_risk` siempre vacío → return silencioso (el más crítico)
- **Archivo**: `InstitutionalDollarStrategy.py`, turbulence check (~línea 292 y 442)
- **Causa**: `_run_pipeline` llamaba `build_feature_matrix(df.copy(), ...)`. Esta función retorna `X` con solo las columnas de `SIGNAL_FEAT_COLS_V2` (features del modelo: `cvd_4h_sum_trade_feature`, etc.). Nunca contiene `log_return_feature`, `volatility_24_feature`, `intraday_range_feature`.
  La línea siguiente: `available_risk = [c for c in risk_vec if c in X.columns]` → siempre lista vacía → `len(available_risk) < 2` → **return silencioso** antes de llegar a "Pipeline OK".
  Esto significa que el pipeline NUNCA ha producido una señal de trading en producción.
- **Verificación**: Test manual confirmó que `build_feature_matrix(df)` (sin `.copy()`) muta `df` in-place con `compute_ohlcv_features`, añadiendo las features necesarias. Después `df.columns` SÍ contiene los 3 features de riesgo.
- **Fix**:
  1. `build_feature_matrix(df)` en lugar de `build_feature_matrix(df.copy())`
  2. `available_risk = [c for c in risk_vec if c in df.columns]` en lugar de `X.columns`
  3. Eliminar el loop redundante que copiaba features de `X` a `df`

#### Bug 3 — return silencioso sin log (gap de monitoreo)
- **Archivo**: `InstitutionalDollarStrategy.py`, línea `if len(available_risk) < 2: return`
- **Causa**: El return no tenía ningún `logger.warning()`, haciendo imposible detectar el problema.
- **Fix**: Añadir `logger.warning("Insufficient risk features (%s) — skipping pipeline", available_risk)` antes del return.

### Estado al finalizar la sesión

| Componente | Estado |
|---|---|
| Daemon (`market_data_daemon.py`) | Reiniciado, PID 724171 (WebSocket cayó durante validación) |
| Dashboard server | Corriendo, PID 132402, `http://localhost:8050` |
| Freqtrade | Corriendo, PID 711045, dry-run |
| Pipeline ejecución | **CONFIRMADA** — primer "Pipeline OK" a las 2026-05-01 08:44:55, bar=1005, HMM fitteado |
| Código | Commiteado en `fix/shadow-live-telemetry`, sincronizado a las 3 copias |

### Para el próximo agente

1. **Pipeline confirmado funcionando** — primer "Pipeline OK" a 2026-05-01 08:44:55. No hay que verificar más esto. Los siguientes logs confirmados:
   ```
   [HMM] Re-fitting model (interval=500 bars, window=1005)...
   [KILL-SWITCH] Regime unknown -> Position zeroed
   [HTF-HMM] Fitting on 730 1h bars...
   Pipeline OK | bar=1005 close=77408.29 alpha=0.500 regime=unknown target_pos=0.0000 (event=no)
   ```

2. **Doble carga (problema pre-existente)**: El bot carga la estrategia DOS veces porque la encuentra en `/home/nosferatu/freqtrade/user_data/strategies/` Y en `RL_Trading/`. Hay dos `InstitutionalSignalEngine` corriendo. Solo uno puede bindear puerto 5556; el segundo falla silenciosamente. La solución definitiva es borrar las copias extras en `strategies/` y `strategies/institutional/` y usar siempre `--strategy-path /home/nosferatu/freqtrade/user_data/strategies/RL_Trading`.

3. **`regime=unknown` y `target_pos=0.0000`** — ACTIVO, bloquea todos los trades:
   - El HMM se fittea correctamente (1005 barras) pero `predict_current` devuelve `"unknown"`.
   - Kill-Switch 1 (`hard_blackout = {"unknown", "panic_selloff", "bear_neutral"}`) pone `raw_target_pos=0.0`.
   - Probable causa: el assert `self._hmm.features[0] == "log_return_feature"` falla porque el HMM se fittó con un orden de features distinto al esperado, o `predict_current` lanza excepción que se captura silenciosamente → `df["hmm_semantic_regime"] = "unknown"`. Añadir `exc_info=True` al warning de HMM (línea ~518) para ver el traceback completo.
   - Alternativamente: `HMMRegimeModel.predict_current` puede devolver `"unknown"` por diseño cuando la confianza es baja o el estado no tiene etiqueta semántica asignada. Ver `utils/risk_directors.py`.

4. **`alpha=0.500`** — ACTIVO, el modelo no genera señal:
   - El stacking (línea ~334-343) anula la señal cuando `alpha_slow` es neutral (`abs(alpha_slow - 0.5) < 0.02`): `df["alpha_prob"] = 0.5 + (last_fast - 0.5) * 0.3`.
   - Si el modelo rápido también da ~0.5, el resultado es exactamente 0.500.
   - También puede ser efecto del shadow selector devolviendo `alpha_prob_final=0.5` por spread tóxico o stale data.
   - Con `alpha=0.500`, Kill-Switch 3 (`if last_oof < 0.50`) no dispara, pero Kill-Switch 1 (regime unknown) ya ha puesto la posición a cero.

4. **Decisión estratégica pendiente**: El modelo de $2M bars tiene AUC ~0.505 y no supera los costes. El EXPERIMENTAL_LOG documenta "Phase 8: FAILED". El código tiene un path de Phase 9 (1h candles vía `FreqtradeCandleProvider`). Discutir con el usuario antes de actuar.

For non-trivial changes:
- Read the related issue or pull request first.
- Create or use a task-specific branch.
- Open a pull request instead of committing directly to the main branch.
- Leave a handoff comment in the issue or pull request.