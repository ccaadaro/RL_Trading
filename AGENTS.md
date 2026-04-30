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

### Estado Actual
-   **Pipeline**: Activo y estable.
-   **Telemetría**: Verificada en puerto 5556 con mensajes `shadow_selector` fluyendo.
-   **Monitor**: `monitor_shadow_selector.py` ahora recibe datos correctamente.

### Siguiente Paso Recomendado
-   Verificar la precisión de las decisiones del selector de régimen en el dashboard conforme lleguen nuevas barras de dólar.

For non-trivial changes:
- Read the related issue or pull request first.
- Create or use a task-specific branch.
- Open a pull request instead of committing directly to the main branch.
- Leave a handoff comment in the issue or pull request.