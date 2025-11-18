import shinyswatch
from shiny import ui

app_ui = ui.page_fillable(
    ui.tags.style(
        """
        /* Make the sidebar independently scrollable */
        .sidebar {
            max-height: 100vh;
            overflow-y: auto;
            position: sticky;
            top: 0;
            background: inherit;
        }
        """
    ),
    ui.navset_bar(
        ui.nav_panel(
            "Single-Node",
            ui.page_sidebar(
                ui.sidebar(
                    ui.tags.script(
                        """
                Shiny.addCustomMessageHandler("logout", function(_) {
                    window.location.href = "https://llm-dashboard.paperclips.dev/cdn-cgi/access/logout";
                });
                """
                    ),
                    ui.layout_columns(
                        ui.input_select("endpoint", "", choices=[]),
                        ui.input_action_button("refreshEndpoints", "⟳"),
                        col_widths=[9, 3],
                    ),
                    ui.input_switch("stream", "Streaming", True),
                    ui.input_switch("autoScroll", "Auto-scroll", True),
                    ui.input_switch("outputReasoning", "Show reasoning", True),
                    ui.input_select(
                        "reasoningEffort",
                        "Reasoning",
                        choices={
                            "none": "None",
                            "low": "Low",
                            "medium": "Medium",
                            "high": "High",
                        },
                        selected="medium",
                    ),
                    ui.hr(),
                    ui.input_switch("outputJSON", "JSON output", False),
                    ui.hr(),
                    shinyswatch.theme_picker_ui(),
                    ui.input_action_button("logout", "Logout"),
                ),
                ui.layout_columns(
                    # Left column with chat UI
                    ui.chat_ui("chat"),
                    # Right column vertical flex with a dedicated scroller
                    ui.div(
                        # Top convo ID row (stays fixed)
                        ui.layout_columns(
                            ui.input_text(
                                "convoID",
                                "",
                                placeholder="Convo ID",
                                width="100%",
                            ),
                            ui.input_action_button("generateConvoID", "New"),
                            col_widths=[7, 5],
                        ),
                        # Bottom accordion with run info: scrollable
                        ui.div(
                            ui.output_ui("outputRunInfo"),
                            class_="flex-grow-1 overflow-auto mt-2",
                        ),
                        class_="h-100 d-flex flex-column",  # column stretches; child can scroll
                    ),
                    col_widths=[9, 3],
                ),
                fillable=True,
            ),
        ),
        ui.nav_panel("Multi-Node"),
        ui.nav_panel(
            "History",
            ui.layout_columns(
                ui.input_action_button("refreshHistory", "Refresh"),
                col_widths=[2],
            ),
            ui.output_ui("historyBox"),
        ),
        title="LLM Control Plane",
    ),
    theme=shinyswatch.theme.flatly,
)
