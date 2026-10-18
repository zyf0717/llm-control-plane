"""Small controls shared by the single-slot and multi-slot views."""

from shiny import ui

REASONING_LEVELS = ["none", "low", "medium", "high", "xhigh", "max"]


def input_action_row(
    input_control,
    button_id,
    button_label,
    *,
    col_widths=None,
    button_class=None,
):
    """Render one input control with a trailing aligned action button."""
    return ui.layout_columns(
        input_control,
        ui.div(
            ui.input_action_button(
                button_id,
                button_label,
                class_=button_class,
            ),
            style="display: flex; align-items: flex-end; height: 100%;",
        ),
        col_widths=col_widths or [10, 2],
    )


def endpoint_selector_ui(input_id: str, models_id: str, *, refresh_id: str):
    return ui.div(
        ui.div(
            ui.input_select(input_id, "Machine Endpoint", choices={}, width="100%").add_style(
                "margin: 0; flex: 1; min-width: 0;"
            ),
            ui.input_action_button(
                refresh_id, "🔄", class_="dashboard-square-button",
                title="Refresh endpoints", **{"aria-label": "Refresh endpoints"},
            ),
            style="display: flex; align-items: flex-end; gap: 0.5rem;",
        ),
        ui.help_text(ui.output_text(models_id, inline=True),
                     style="margin: 0.25rem 0 0; overflow-wrap: anywhere;"),
    )


def endpoint_models_text(endpoint: dict, *, loading: bool = False) -> str:
    models = [model["id"] for model in endpoint.get("models", []) if model.get("id")]
    if models:
        return ", ".join(models)
    return "Loading models…" if loading else "No models reported."


def resolve_endpoint_selection(choices: dict, *, preferred_endpoint_key: str | None) -> str | None:
    preferred = str(preferred_endpoint_key or "").strip()
    return preferred if preferred in choices else next(iter(choices), None)
