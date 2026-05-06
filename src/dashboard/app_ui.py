import shinyswatch
from shiny import ui

app_ui = ui.page_fluid(
    ui.tags.style("""
        :root {
            --dashboard-panel-height: calc(100dvh - 8rem);
            --dashboard-chat-height: var(--dashboard-panel-height);
        }

        /* Make the sidebar independently scrollable */
        .sidebar {
            max-height: 100vh;
            overflow-y: auto;
            position: sticky;
            top: 0;
            background: inherit;
        }

        .dashboard-pane {
            height: var(--dashboard-panel-height);
            min-height: 0;
        }

        .dashboard-side-panel {
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
        }

        .dashboard-run-info {
            flex: 1 1 auto;
            min-height: 0;
            overflow-y: auto;
        }

        @media (max-width: 991.98px) {
            :root {
                --dashboard-panel-height: auto;
                --dashboard-chat-height: min(60dvh, 32rem);
            }

            .dashboard-run-info {
                max-height: 20rem;
            }
        }
        """),
    ui.navset_bar(
        ui.nav_panel(
            "Single-Node",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.tags.script("""
                Shiny.addCustomMessageHandler("logout", function(_) {
                    window.location.href = "https://llm-dashboard.paperclips.dev/cdn-cgi/access/logout";
                });
                """),
                    ui.layout_columns(
                        ui.input_select("endpoint", "", choices=[]),
                        ui.input_action_button("refreshEndpoints", "🔄"),
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
                    ui.chat_ui(
                        "chat",
                        width="100%",
                        height="var(--dashboard-chat-height)",
                    ),
                    ui.div(
                        ui.layout_columns(
                            ui.input_text(
                                "convoID",
                                "",
                                placeholder="Convo ID",
                                width="100%",
                            ),
                            ui.input_action_button("generateConvoID", "🔄"),
                            col_widths=[10, 2],
                        ),
                        ui.input_text_area(
                            "systemPrompt",
                            "",
                            placeholder="System prompt (optional)",
                            width="100%",
                            rows=4,
                        ),
                        ui.output_ui("file_upload_ui"),
                        ui.div(
                            ui.output_ui("outputRunInfo"),
                            class_="dashboard-run-info",
                        ),
                        class_="dashboard-pane dashboard-side-panel",
                    ),
                    col_widths=[9, 3],
                ),
                fillable=True,
            ),
        ),
        ui.nav_panel(
            "Multi-Node",
            ui.div("Multi-node UI coming soon!"),
        ),
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
