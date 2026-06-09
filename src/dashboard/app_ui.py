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

        .dashboard-trace-json {
            max-height: 28rem;
            overflow: auto;
            white-space: pre-wrap;
            overflow-wrap: anywhere;
            margin-bottom: 0;
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
                        ui.input_select("endpoint", "Machine Endpoint", choices=[]),
                        ui.div(
                            ui.input_action_button("refreshEndpoints", "🔄"),
                            style="display: flex; align-items: flex-end; height: 100%;",
                        ),
                        col_widths=[9, 3],
                    ),
                    ui.input_switch("stream", "Streaming", True),
                    ui.input_switch("autoScroll", "Follow output", True),
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
                    ui.output_ui("system_prompt_ui"),
                    ui.hr(),
                    ui.input_switch("outputJSON", "JSON output", False),
                    ui.hr(),
                    shinyswatch.theme_picker_ui(),
                    ui.input_action_button("logout", "Logout"),
                    width="375px",
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
                        ui.layout_columns(
                            ui.input_select(
                                "ragEndpoint",
                                "RAG Endpoint",
                                choices=[],
                                width="100%",
                            ),
                            ui.div(
                                ui.input_action_button("refreshRagEndpoints", "🔄"),
                                style="display: flex; align-items: flex-end; height: 100%;",
                            ),
                            col_widths=[10, 2],
                        ),
                        ui.input_select(
                            "searchProvider",
                            "Search Provider",
                            choices=[],
                            width="100%",
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
                ui.input_select(
                    "historyConvoSelector",
                    "Conversation",
                    choices={},
                    width="100%",
                ),
                ui.div(
                    ui.input_action_button("refreshHistory", "Refresh"),
                    style="display: flex; align-items: flex-end; height: 100%;",
                ),
                col_widths=[3, 2],
            ),
            ui.output_ui("historyBox"),
        ),
        ui.nav_panel(
            "Traces",
            ui.layout_columns(
                ui.input_text(
                    "traceConvoFilter",
                    "Convo ID",
                    placeholder="Optional",
                    width="100%",
                ),
                ui.input_text(
                    "traceIDFilter",
                    "Trace ID",
                    placeholder="Optional",
                    width="100%",
                ),
                ui.input_text(
                    "traceEndpointFilter",
                    "Endpoint",
                    placeholder="Optional",
                    width="100%",
                ),
                ui.input_select(
                    "traceMaxRows",
                    "Rows",
                    choices={"50": "50", "100": "100", "200": "200", "500": "500"},
                    selected="200",
                    width="100%",
                ),
                ui.div(
                    ui.input_action_button("refreshTraces", "Refresh"),
                    style="display: flex; align-items: flex-end; height: 100%;",
                ),
                col_widths=[3, 3, 3, 1, 2],
            ),
            ui.output_ui("traceBox"),
        ),
        title="LLM Control Plane",
    ),
    theme=shinyswatch.theme.flatly,
)
